"""Adaptive code-pile structuring — discovers projects, version families, and
dependencies from a heterogeneous code folder (`source/code/`).

Implements the 5-stage process from `doc/kb-code-material-optimization.md`:

  Stage 1 — universal signals (full-content hash dedup + import graph)
  Stage 2 — project clustering (co-location + import connectivity)
  Stage 3 — version-family grouping (frequency-aware name-similarity
             + content-overlap)
  Stage 4 — canonical / duplicate resolution
  Stage 5 — semantic labeling (curated-project reconciliation + LLM hook)

Produces `artifact/knowledge_bits/project_clusters.json` — a clustering with
confidence, where each cluster carries the signal(s) that produced it. The
CodeMap renderer is only a *view* over this intermediate: re-running Stages 1–3
after the folder changes yields a fresh clustering without touching the renderer.

Deliberately NOT hardcoded to this folder's conventions: no `{date}_{name}_{ver}`
regex, no `base/` special case (content-hash finds duplicates anywhere), no
person-dir name table (a *frequency-aware* generic-token detector strips owner /
asset-class / stage tokens wherever they appear). The only domain knowledge reused is the KB domain model (`model_data`),
not any folder layout convention.

Curated `source/project/<name>/*.py` files are the *authoritative* project code
and are NOT clustered here (Phase 2 already turns them into Project entities).
They are used only as reconciliation anchors: a pile cluster whose files match a
spec's `对应代码` list is linked back to that curated project.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from knowledge_extractor.util.paths import kb_dir, root
from knowledge_extractor.util import model_data

KB_DIR = kb_dir()
# TODO(搬移): 新 paths 未在改写规则中列出 root();若无此函数需按实际接口调整
ROOT = root()

# ── Universal generic tokens ──
# Version / duplicate / qualifier words present in ANY code folder (not specific
# to this one). Used to strip a directory name down to its "project" identity.
_GENERIC_TOKENS = {
    "dump", "old", "new", "copy", "backup", "bak", "base", "tmp", "test",
    "latest", "wip", "dev", "src", "main", "to", "for", "of", "and", "the",
}

_CODE_ROOT = "source/code"


# ── path helpers ──

def _load_json(name: str) -> list | dict:
    """Read a knowledge_bits JSON, degrading to [] on a missing/unreadable file
    (so a standalone run before Phase 1 writes its outputs is a clean no-op)."""
    p = KB_DIR / name
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"  ⚠ project_clusters: cannot read {p} ({e})", flush=True)
        return []


def _full_content_hash(path_str: str) -> str:
    """SHA-1 of the full file bytes — the universal 'is this a duplicate?' signal."""
    try:
        return hashlib.sha1((ROOT / path_str).read_bytes()).hexdigest()
    except OSError:
        return ""


def _stem(path_str: str) -> str:
    return Path(path_str).stem


def _pile_dir(path_str: str) -> str:
    """Directory of a `source/code/` file, relative to `source/code/`.

    Returns '' for a loose top-level script. For a non-pile path this returns the
    parent directory as a best effort — callers pre-filter to `source/code/` files.
    """
    parts = Path(path_str).parts
    # parts == ('source', 'code', <subdirs…>, <filename>)
    if len(parts) >= 3 and parts[0] == "source" and parts[1] == "code":
        return "/".join(parts[2:-1])
    return "/".join(parts[:-1])


def _group_key(f: dict) -> str:
    """Clustering key for a pile file: its relative dir, or `@<path>` for a loose
    top-level script (each loose script is its own unit, never merged)."""
    return f["dir"] if f["dir"] else "@" + f["path"]


def _group_key_of_path(path_str: str) -> str:
    """Clustering key for a pile file given only its full path."""
    d = _pile_dir(path_str)
    return d if d else "@" + path_str


def _display_label(c: dict) -> str:
    """Human-readable cluster name: LLM semantic label if present, else the
    deterministic normalized key (stable across runs)."""
    return c.get("llm_label") or c["label"]


# ── Stage 1: universal signals ──

def _load_code_files() -> list[dict]:
    """Load `code_file` sources under `source/code/` (the pile), excluding
    `.ipynb_checkpoints` (per-file notebook mirrors, not real code)."""
    sources = _load_json("sources.json")
    files = []
    for s in sources:
        if s["source_type"] != "code_file":
            continue
        p = s["path"]
        if ".ipynb_checkpoints" in p:
            continue
        if not p.startswith("source/code/"):
            continue  # curated source/project code is not part of the pile
        files.append({
            "source_id": s["id"],
            "path": p,
            "stem": _stem(p),
            "dir": _pile_dir(p),
            "hash": _full_content_hash(p),
        })
    return files


def _dedup(files: list[dict]) -> tuple[list[dict], list[dict]]:
    """Stage 1: collapse content-identical files. Returns (canonical, duplicates).

    Canonical selection prefers a non-`base/`, non-`_dump` path, then the shortest
    path (the "most primary" location). `base/` re-organizations and `_dump`
    snapshots resolve to their primary versioned directory without any hardcoded
    knowledge of what `base/` means — content identity alone decides.
    """
    readable = [f for f in files if f["hash"]]
    unreadable = [f for f in files if not f["hash"]]

    by_hash: dict[str, list[dict]] = defaultdict(list)
    for f in readable:
        by_hash[f["hash"]].append(f)

    canonical: list[dict] = []
    duplicates: list[dict] = []
    for h, group in by_hash.items():
        if len(group) == 1:
            canonical.extend(group)
            continue

        def _rank(f: dict) -> tuple[int, int, int, str]:
            p = f["path"]
            return (
                1 if "/base/" in p else 0,      # base/ is a re-org copy
                1 if "_dump" in p else 0,        # _dump is a snapshot
                len(p),                          # shorter = more primary
                p,                               # deterministic tiebreak
            )

        ordered = sorted(group, key=_rank)
        primary = ordered[0]
        canonical.append(primary)
        for dup in ordered[1:]:
            duplicates.append({
                "path": dup["path"],
                "canonical": primary["path"],
                "hash": h,
            })

    # Unreadable files (empty hash) are kept as canonical singletons: their
    # content is unknown so they can never be deduped, but they must not be
    # silently dropped from the clustering / stats.
    canonical.extend(unreadable)
    return canonical, duplicates


# ── Stage 1b / 2: import graph ──

def _build_import_edges(files: list[dict]) -> list[dict]:
    """Within-directory import edges (file A imports sibling file B).

    Built over the FULL pile file set (canonical + duplicates) so that a
    `→ utils` import still resolves when `utils.py` happens to be a duplicate of a
    copy elsewhere. Resolves each code unit's `imports` module name against the
    sibling file stems in the same directory — the only language-agnostic "these
    files form one thing" signal.
    """
    units = _load_json("code_units.json")
    imports_by_src: dict[str, set[str]] = defaultdict(set)
    for u in units:
        for imp in u.get("imports", []):
            module = imp.split(":", 1)[0].split(".", 1)[0]
            if module:
                imports_by_src[u["source_id"]].add(module.lower())

    stems_in_dir: dict[str, dict[str, str]] = defaultdict(dict)
    for f in files:
        stems_in_dir[f["dir"]][f["stem"].lower()] = f["path"]

    edges: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for f in files:
        d = f["dir"]
        for module in imports_by_src.get(f["source_id"], ()):
            target = stems_in_dir[d].get(module)
            if target and target != f["path"]:
                key = (f["path"], target)
                if key not in seen:
                    seen.add(key)
                    edges.append({"from": f["path"], "to": target, "type": "import"})
    return edges


# ── Stage 1c / P2: cross-project data flow ──

_DATA_MOUNT_PREFIXES = ("/mnt/", "/cache/", "/ext1/", "hdfs://")
_DATA_EXT = (".csv", ".npy", ".parquet", ".json", ".zst", ".txt")


def _is_data_path(s: str) -> bool:
    """A concrete data location (mounted HDFS path / config / data file), as
    opposed to a regex, glob, format-template (`%s`/`$VAR`), placeholder, or
    truncated prefix."""
    s = s.rstrip("/")
    if not s:
        return False
    if re.search(r"[\\{}*?()\[\]%$]", s):
        return False
    # template placeholders (config examples like `/my_path_hdfs.json`)
    if re.search(r"(my_path|example|placeholder|your_)", s.lower()):
        return False
    if s.startswith(_DATA_MOUNT_PREFIXES):
        # reject truncated prefixes (a trailing `_`/`-`/`.` from string
        # concatenation), keep real dirs/files ending in an alnum char or data ext
        return s[-1:].isalnum() or s.endswith(_DATA_EXT)
    return s.startswith("/") and s.endswith(_DATA_EXT)


def _data_paths_for_source(source_id: str, literals_by_source: dict) -> set[str]:
    paths: set[str] = set()
    for lit in literals_by_source.get(source_id, ()):
        s = lit.strip().strip("'\"`")
        if _is_data_path(s):
            paths.add(s.rstrip("/"))
    return paths


def _build_data_flow(canonical: list[dict],
                     cluster_by_group: dict[str, dict]) -> list[dict]:
    """Cross-project data flow: a data path referenced by files in ≥2 different
    clusters. This is the 'project A and project B touch the same data' signal
    (Stage 1 data references), independent of import structure."""
    units = _load_json("code_units.json")
    literals_by_source: dict[str, set[str]] = defaultdict(set)
    for u in units:
        for lit in u.get("literals", []):
            literals_by_source[u["source_id"]].add(lit)

    cluster_paths: dict[str, set[str]] = defaultdict(set)  # cluster id → data paths
    cluster_labels: dict[str, str] = {}
    for f in canonical:
        c = cluster_by_group.get(_group_key(f))
        if c is None:
            continue
        cluster_paths[c["id"]] |= _data_paths_for_source(f["source_id"], literals_by_source)
        cluster_labels[c["id"]] = c["label"]  # deterministic key; display name resolved at render

    path_clusters: dict[str, set[str]] = defaultdict(set)
    for cid, paths in cluster_paths.items():
        for p in paths:
            path_clusters[p].add(cid)

    data_flow: list[dict] = []
    for p, cids in sorted(path_clusters.items()):
        if len(cids) < 2:
            continue
        refs = sorted(
            ({"id": cid, "label": cluster_labels[cid]} for cid in cids),
            key=lambda r: r["label"],
        )
        data_flow.append({"path": p, "clusters": refs})
    return data_flow


def _distinctive_stem_sets(canonical: list[dict]) -> dict[str, set[str]]:
    """Per-directory set of *distinctive* stems (a stem shared by ≤2 directories).

    A stem that appears in many directories (utils, proc_1dnpy_hdfs, …) is a
    copy-pasted helper, not an identity of any one project — so it must not be
    used to merge two unrelated projects. Only rare stems link a project's versions.
    """
    stem_dirs: dict[str, set[str]] = defaultdict(set)
    for f in canonical:
        stem_dirs[f["stem"]].add(f["dir"])
    per_dir: dict[str, set[str]] = defaultdict(set)
    for stem, dirs in stem_dirs.items():
        if 1 <= len(dirs) <= 2:
            for d in dirs:
                per_dir[d].add(stem)
    return per_dir


# ── Stage 3: name-similarity (frequency-aware token stripping) ──

def _tokenize(name: str) -> list[str]:
    return [t for t in re.split(r"[^0-9a-zA-Z]+", name.lower()) if t]


def _is_numeric_qualifier(tok: str) -> bool:
    """Date / version / numeric markers (20240506, v8, 250219, 10k, 1m, v817x…)."""
    t = re.sub(r"^v+", "", tok)
    if not t:
        return True  # a bare 'v'
    if t.isdigit():
        return True
    if re.fullmatch(r"\d+[a-z]{1,3}", t):
        return True
    return False


def _compute_generic_tokens(dirs: list[str]) -> set[str]:
    """Tokens appearing in *many* directories are structural (owner / asset-class /
    stage), not project identity. Threshold scales with the number of directories
    so the rule is adaptive to folder size, not a magic constant for this folder."""
    token_dirs: dict[str, set[str]] = defaultdict(set)
    for d in dirs:
        for seg in d.split("/"):
            for tok in _tokenize(seg):
                token_dirs[tok].add(d)
    n = max(len(dirs), 1)
    threshold = max(4, math.ceil(0.12 * n))
    return {tok for tok, ds in token_dirs.items() if len(ds) >= threshold}


def _normalized_dir_key(dirname: str, generic_tokens: set[str]) -> str:
    """Strip a directory path to its 'project identity' tokens.

    Removes numeric/date/version tokens, universal generic tokens, asset-class
    tokens, frequency-generic tokens, and single-character tokens (a trailing
    `_y` variant suffix, etc.). The surviving tokens, joined in order, are the
    project key. Empty → fall back to the full path so every directory still has
    a distinct default key.
    """
    tokens: list[str] = []
    for seg in dirname.split("/"):
        for tok in _tokenize(seg):
            if _is_numeric_qualifier(tok):
                continue
            if tok in _GENERIC_TOKENS or tok in _ASSET_CLASS_TOKENS:
                continue
            if tok in generic_tokens:
                continue
            if len(tok) == 1:
                continue
            tokens.append(tok)
    key = "_".join(tokens)
    return key or dirname.lower()


# ── Stage 5: curated-project reconciliation ──

def _curated_code_refs() -> dict[str, str]:
    """Map canonical file stem → curated project name, from `对应代码` in
    `source/project/*.md` specs. The spec's 任务信息 block lists the code files a
    curated project owns; this is the authoritative project↔code anchor."""
    stem_to_project: dict[str, str] = {}
    project_dir = ROOT / "source" / "project"
    if not project_dir.exists():
        return stem_to_project
    for md in project_dir.rglob("*.md"):
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        m = re.search(r"对应代码[：:]\s*(.+?)(?=\n\s*\S+[：:]|\Z)", text, re.DOTALL)
        if not m:
            continue
        for token in re.split(r"[、，,;；\s]+", m.group(1)):
            name = token.strip().strip("`[]()").strip()
            if name.endswith(".py"):
                stem_to_project[Path(name).stem] = md.parent.name
    return stem_to_project


# ── Stage 5: semantic labeling (LLM hook, deterministic fallback) ──

_CLUSTER_LABEL_PROMPT = """You are labeling a cluster of Python files that belong together in a team's code pile. Given the directory name(s) and file names, produce a short ENGLISH label (≤4 words) naming this project/unit, plus a one-line description.

Input:
{evidence}

Return JSON: {{"label": "...", "description": "..."}}"""


def _llm_label(clusters: list[dict]) -> None:
    """Optional: enrich labels via one LLM call per cluster. Falls back silently."""
    # TODO(搬移): 新 util.llm 接口为 chat/extract_json/available,不再提供 LlmConfig/_call_llm;算法逻辑保留未改,后续按新接口适配
    from knowledge_extractor.util.llm import LlmConfig, _call_llm

    cfg = LlmConfig.from_env()
    if not cfg.available:
        return
    for c in clusters:
        evidence = "\n".join(
            f"- dir: {v['dir']} ({v['files']} files)"
            for v in c["versions"][:5]
        )
        try:
            raw = _call_llm(_CLUSTER_LABEL_PROMPT.format(evidence=evidence), cfg)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                continue
            out = json.loads(m.group())
            if isinstance(out.get("label"), str) and out["label"].strip():
                c["llm_label"] = out["label"].strip()
            if isinstance(out.get("description"), str):
                c["description"] = out["description"].strip()
        except Exception:
            continue


# ── Main builder ──

def build_project_clusters() -> dict:
    files = _load_code_files()
    unreadable = [f for f in files if not f["hash"]]
    if unreadable:
        print(f"  ⚠ project_clusters: {len(unreadable)} file(s) unreadable — "
              f"kept as canonical singletons (no dedup possible)", flush=True)
    canonical, duplicates = _dedup(files)
    import_edges = _build_import_edges(files)
    distinctive = _distinctive_stem_sets(canonical)

    # Stage 2: co-location. Group canonical files by directory. Loose top-level
    # scripts (dir == '') are NOT a coherent unit — each is its own singleton, so
    # give each a synthetic key (its own path) rather than lumping them together.
    by_dir: dict[str, list[dict]] = defaultdict(list)
    for f in canonical:
        by_dir[_group_key(f)].append(f)

    # Only real directories participate in version-family merging; loose scripts
    # never merge (they are independent one-off scripts).
    real_dirs = [k for k in by_dir if not k.startswith("@")]

    # Stage 3: union-find merge of real directories into clusters, driven by two
    # adaptive signals (frequency-aware name-similarity + content-overlap).
    dirs = sorted(real_dirs)
    parent = {d: d for d in dirs}

    def find(d: str) -> str:
        while parent[d] != d:
            parent[d] = parent[parent[d]]
            d = parent[d]
        return d

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # frequency-aware generic tokens, computed from the directories themselves.
    generic_tokens = _compute_generic_tokens(dirs)

    # signal A — name-similarity: same normalized project key.
    key_of = {d: _normalized_dir_key(d, generic_tokens) for d in dirs}
    by_key: dict[str, list[str]] = defaultdict(list)
    for d in dirs:
        by_key[key_of[d]].append(d)
    for key, group in by_key.items():
        if len(group) > 1:
            for d in group[1:]:
                union(group[0], d)

    # signal B — content-overlap: two directories share a distinctive stem.
    for d in dirs:
        for stem in distinctive.get(d, ()):
            for d2 in dirs:
                if d2 != d and stem in distinctive.get(d2, ()):
                    union(d, d2)

    # group into clusters (real dirs), then append each loose script separately.
    clusters: dict[str, list[str]] = defaultdict(list)
    for d in dirs:
        clusters[find(d)].append(d)

    result: list[dict] = []
    for root_dir in sorted(clusters.keys()):
        members = sorted(clusters[root_dir])
        member_files = [f for d in members for f in by_dir[d]]
        file_count = len(member_files)
        internal_edges = [
            e for e in import_edges
            if _pile_dir(e["from"]) in members and _pile_dir(e["to"]) in members
        ]

        versions = []
        for d in members:
            versions.append({
                "dir": _CODE_ROOT + "/" + d,
                "role": "canonical",
                "files": len(by_dir[d]),
                "dup_of": None,
            })

        # type + confidence + signals
        signals: list[str] = ["co-location"]
        if internal_edges:
            signals.append("import-connectivity")
        if len(members) > 1:
            signals.append("version-family")

        distinct_dirs = len(members)
        if distinct_dirs > 1:
            ctype = "version_family"
        elif internal_edges:
            ctype = "project"
        elif file_count >= 2:
            ctype = "work_collection"
        else:
            ctype = "standalone"  # a lone script in its own directory

        # confidence: base 0.4 for co-location, +0.3 connectivity, +0.2 family.
        confidence = 0.4
        if internal_edges:
            confidence += 0.3
        if distinct_dirs > 1:
            confidence += 0.2
        confidence = min(confidence, 1.0)

        label = _normalized_dir_key(members[0], generic_tokens)
        if not label or label == members[0].lower():
            label = members[0].replace("/", "_")

        result.append({
            "id": "pc-" + hashlib.sha1(("|".join(members)).encode()).hexdigest()[:12],
            "type": ctype,
            "label": label,
            "curated_refs": [],
            "confidence": round(confidence, 2),
            "source_signals": signals,
            "versions": versions,
            "file_count": file_count,
            "duplicate_count": 0,
            "internal_import_edges": len(internal_edges),
            "import_graph": internal_edges,
            "_group_keys": members,
        })

    # loose top-level scripts: each is a standalone cluster.
    for key in sorted(by_dir):
        if not key.startswith("@"):
            continue
        fl = by_dir[key]
        f = fl[0]
        result.append({
            "id": "pc-" + hashlib.sha1(f["path"].encode()).hexdigest()[:12],
            "type": "standalone",
            "label": f["stem"],
            "curated_refs": [],
            "confidence": 0.4,
            "source_signals": ["co-location"],
            "versions": [{"dir": f["path"], "role": "canonical",
                          "files": 1, "dup_of": None}],
            "file_count": 1,
            "duplicate_count": 0,
            "internal_import_edges": 0,
            "import_graph": [],
            "_group_keys": [key],
        })

    # Map every cluster's group key → cluster record, for reconciliation and
    # duplicate attachment.
    cluster_by_group: dict[str, dict] = {}
    for c in result:
        for gk in c["_group_keys"]:
            cluster_by_group[gk] = c

    # Stage 5 (deterministic part): reconcile against curated project specs.
    stem_to_project = _curated_code_refs()
    for f in canonical:
        proj = stem_to_project.get(f["stem"])
        if not proj:
            continue
        c = cluster_by_group.get(_group_key(f))
        if c is None or proj in c["curated_refs"]:
            continue
        c["curated_refs"].append(proj)
        if "curated" not in c["source_signals"]:
            c["source_signals"].append("curated")
        c["confidence"] = min(c["confidence"] + 0.1, 1.0)

    # Attach duplicate files to their canonical's cluster (role = duplicate).
    for dup in duplicates:
        c = cluster_by_group.get(_group_key_of_path(dup["canonical"]))
        if c is None:
            continue
        c["versions"].append({
            "dir": dup["path"],
            "role": "duplicate",
            "files": 1,
            "dup_of": dup["canonical"],
        })
        c["duplicate_count"] += 1

    # Drop the internal grouping keys — they are not part of the output contract.
    for c in result:
        c.pop("_group_keys", None)

    # Optional LLM labeling (P3).
    try:
        _llm_label(result)
    except Exception:
        pass

    # Re-round confidence (the curated boost may have introduced float drift).
    for c in result:
        c["confidence"] = round(c["confidence"], 2)

    # P2: cross-project data flow — data paths shared by ≥2 clusters (computed
    # after LLM labeling so the returned labels are final).
    data_flow = _build_data_flow(canonical, cluster_by_group)

    # order clusters: curated first, then by file_count desc, then stable id.
    result.sort(key=lambda c: (not c["curated_refs"], -c["file_count"], c["id"]))

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stats": {
            "code_files": len(files),
            "canonical_files": len(canonical),
            "duplicate_files": len(duplicates),
            "duplicate_groups": len({d["hash"] for d in duplicates}),
            "skipped_unreadable": len(unreadable),
            "clusters": len(result),
            "import_edges": len(import_edges),
            "data_flow_paths": len(data_flow),
        },
        "clusters": result,
        "import_edges": import_edges,
        "data_flow": data_flow,
        "duplicates": duplicates,
    }


def write_project_clusters() -> dict:
    data = build_project_clusters()
    KB_DIR.mkdir(parents=True, exist_ok=True)
    (KB_DIR / "project_clusters.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


if __name__ == "__main__":
    data = write_project_clusters()
    s = data["stats"]
    print(f"project_clusters: {s['clusters']} clusters, "
          f"{s['canonical_files']} canonical / {s['duplicate_files']} duplicate files, "
          f"{s['import_edges']} import edges")
    for c in data["clusters"]:
        cur = f" → {','.join(c['curated_refs'])}" if c["curated_refs"] else ""
        print(f"  [{c['type']:16s}] {_display_label(c):28s} "
              f"conf={c['confidence']:.2f} files={c['file_count']} "
              f"dup={c['duplicate_count']} sig={c['source_signals']}{cur}")
