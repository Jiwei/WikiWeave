"""Code map (Stage D) — deterministic map of the ingested code, replacing the
directory-mirror index with a functional map.

Produces knowledge_bits/code_map.json and renders artifact/index/code/index.md:

  - dedup: content-hash + path-pattern detection of duplicated files (base/
    prefix copies, .ipynb_checkpoints) → a dedup table
  - naming keys: a curated dictionary of team abbreviations (abc/def/ghi/
    tsf/v817/cx/stat_1m/…) so the index is self-explanatory
  - code domains: classify each file into a code domain (ETL stage) by filename + docstring
  - leaf summaries: one-line description per file (docstring → structure)
  - external repos: URLs / gitlab-style references found in literals

Naming keys are LLM-verified against READMEs + weekly reports; names that cannot
be verified are marked ⚠️ unverified.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from knowledge_extractor.util import model_data as _md
from knowledge_extractor.util.i18n import t
from knowledge_extractor.util.paths import INDEX_DIR, KB_DIR, ROOT, portable_code_path

# ── 命名键 / 代码域(code_domain):单一事实源 model_data/code/{naming_keys,code_domain}.json ──
# code_domain 是 domains 的一个子集/细化:每个 code_domain 关联一个 domain(+可选 sub_domain)。
NAMING_KEYS: dict[str, str] = _md.naming_keys()

_code_domain = _md.code_domains()
CODE_DOMAINS = _code_domain.get("code_domains", [])  # [{name, domain, sub_domain, keywords, description}]
CODE_DOMAIN_KEYWORDS = [(cd["name"], cd["keywords"]) for cd in CODE_DOMAINS]  # [(name, [kws]), ...]
CODE_DOMAIN_DESC = {cd["name"]: cd["description"] for cd in CODE_DOMAINS}
CODE_DOMAIN_LINK = {cd["name"]: (cd.get("domain"), cd.get("sub_domain")) for cd in CODE_DOMAINS}
UNCLASSIFIED_LABEL = _code_domain.get("unclassified_label", "Other / Unclassified")
CODE_DOMAIN_DESC[UNCLASSIFIED_LABEL] = _code_domain.get("unclassified_description", "doesn't fit any of the above")
CODE_DOMAIN_LABELS = [cd["name"] for cd in CODE_DOMAINS] + [UNCLASSIFIED_LABEL]


def _file_text(path_str: str) -> str:
    try:
        p = ROOT / path_str
        return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
    except Exception:
        return ""


def _content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _naming_key_matches(name: str) -> list[str]:
    tl = name.lower()
    hits = []
    for key, meaning in NAMING_KEYS.items():
        # split multi-alias keys like "v817 / v821"
        for alias in key.split("/"):
            alias = alias.strip().lower()
            if alias and (alias in tl or alias.replace("v", "") in tl):
                hits.append(f"{key} — {meaning}")
                break
    return hits


def _classify_code_domain(path_str: str, docstring: str, func_names: list[str]) -> str:
    text = f"{path_str} {docstring} {' '.join(func_names)}".lower()
    best, best_count = None, 0
    for label, kws in CODE_DOMAIN_KEYWORDS:
        c = sum(1 for k in kws if k.lower() in text)
        if c > best_count:
            best, best_count = label, c
    return best or UNCLASSIFIED_LABEL


def _classify_code_domains_llm(files: list[dict]) -> dict[str, str]:
    """Classify code files into code domains via LLM (filename + docstring).

    Returns {path: code_domain}. Returns {} if the LLM is unavailable or fails, so the
    caller falls back to the keyword classifier.
    """
    from knowledge_extractor.util.llm import LlmConfig, _call_llm
    import json as _json
    import re as _re

    cfg = LlmConfig.from_env()
    if not cfg.available:
        return {}

    code_domain_list = "\n".join(f"- {s}: {CODE_DOMAIN_DESC[s]}" for s in CODE_DOMAIN_LABELS)
    naming_hint = "\n".join(f"- {k}: {v}" for k, v in NAMING_KEYS.items())
    result: dict[str, str] = {}
    batch_size = 30
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        lines = []
        for j, f in enumerate(batch):
            doc = (f["docstring"] or "").replace("\n", " ")[:200]
            head = (f.get("source_head") or "").replace("\n", " ")[:200]
            funcs = ", ".join(f["funcs"][:8]) or "(none)"
            lines.append(f"[{j}] {f['name']} | doc: {doc or 'none'} | src: {head or 'none'} | funcs: {funcs}")
        prompt = (
            "Classify each script below into one code domain (ETL pipeline stage), using its "
            "filename, docstring, source head, and function names.\n\n"
            f"Abbreviations (team naming keys):\n{naming_hint}\n\n"
            f"Code domains:\n{code_domain_list}\n\n"
            "Return a JSON object mapping each [index] (as a string) to exactly one "
            "code-domain label from the list above.\n\n"
            f"Scripts:\n{chr(10).join(lines)}"
        )
        try:
            raw = _call_llm(prompt, cfg)
        except Exception:
            continue
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        if not m:
            continue
        try:
            parsed = _json.loads(m.group())
        except Exception:
            continue
        for j, f in enumerate(batch):
            cd = parsed.get(str(j), parsed.get(j))
            if isinstance(cd, str) and cd in CODE_DOMAIN_LABELS:
                result[f["path"]] = cd
    return result


def _one_line_summary(path_str: str, func_names: list[str], class_names: list[str],
                      docstring: str) -> str:
    fname = Path(path_str).name
    parts = []
    if docstring.strip():
        parts.append(docstring.strip().splitlines()[0][:100])
    if func_names:
        parts.append(f"{len(func_names)} functions: {', '.join(func_names[:4])}")
    if class_names:
        parts.append(f"{len(class_names)} classes: {', '.join(class_names[:2])}")
    return "; ".join(parts) or fname


NAMING_KEY_VERIFY_PROMPT = """You are verifying abbreviations used in a team's code pipeline. Here is the current abbreviation dictionary, where each key is an abbreviation and the value is my best-guess meaning:

{dict}

Here is relevant evidence text from the team's code READMEs and weekly reports (may be Chinese; may contain the abbreviations in context):

{evidence}

Task: For each abbreviation, confirm or correct its meaning based ONLY on the evidence, and write the meaning in concise ENGLISH. If the evidence is insufficient to verify an abbreviation, mark it "⚠️ unverified". Return a JSON object mapping each original abbreviation key → the verified meaning string (keep the key exactly as given).

Return: {{"abc": "verified English meaning or ⚠️ unverified", ...}}"""


def _gather_evidence(sources: list[dict], limit: int = 200) -> str:
    """Collect code README + weekly-report text fragments containing the abbreviations."""
    import re as _re
    abbrs = "|".join(_re.escape(k) for k in NAMING_KEYS)
    pat = _re.compile(rf".{{0,60}}(?:{abbrs}).{{0,80}}", _re.IGNORECASE)
    evidence = []
    for s in sources:
        if s["source_type"] not in ("code_readme", "weekly_report", "project", "doc"):
            continue
        try:
            text = _file_text(s["path"])
        except Exception:
            continue
        for m in pat.finditer(text):
            frag = " ".join(m.group().split())[:140]
            evidence.append(f"[{s['source_type']}] {frag}")
            if len(evidence) >= limit:
                return "\n".join(evidence)
    return "\n".join(evidence)


def _verify_naming_keys(sources: list[dict]) -> dict:
    """LLM-verify the abbreviation dictionary against code READMEs + weekly reports.

    Returns an updated {key: meaning} dict; unknown keys are marked ⚠️ unverified.
    Falls back to the curated dict if the LLM is unavailable.
    """
    from knowledge_extractor.util.llm import LlmConfig, _call_llm
    import json as _json
    import re as _re

    cfg = LlmConfig.from_env()
    if not cfg.available:
        return dict(NAMING_KEYS)

    evidence = _gather_evidence(sources)
    dict_str = "\n".join(f"- {k}: {v}" for k, v in NAMING_KEYS.items())
    prompt = NAMING_KEY_VERIFY_PROMPT.format(dict=dict_str, evidence=evidence or "(no evidence available)")
    try:
        raw = _call_llm(prompt, cfg)
    except Exception as e:
        print(f"  naming-key LLM verify failed ({e}) — keeping curated", flush=True)
        return dict(NAMING_KEYS)
    m = _re.search(r"\{.*\}", raw, _re.DOTALL)
    if not m:
        return dict(NAMING_KEYS)
    try:
        verified = _json.loads(m.group())
    except Exception:
        return dict(NAMING_KEYS)
    out = {}
    for k in NAMING_KEYS:
        v = verified.get(k)
        if isinstance(v, str) and v.strip():
            out[k] = v.strip() if v.strip() != "⚠️ unverified" else f"{NAMING_KEYS[k]} ⚠️ unverified"
        else:
            out[k] = NAMING_KEYS[k]
    return out


def classify_code_modules(code_modules: list, code_units: list) -> None:
    """把 code_domain 分类结果写回 CodeModule(原地)。

    code_domain 是 domains 的一个子集/细化:每个 CodeModule 的 code_domain 关联到
    code_domain.json 的 name,而后者又关联到 domains(见 CODE_DOMAIN_LINK)。关键字分类为主,
    LLM 可用时用 LLM 精化。CodeModule.code_domain 随 code_modules.json 落盘后由 build_code_map 消费。
    """
    if not code_modules:
        return
    funcs_by_source: dict[str, list[str]] = defaultdict(list)
    for cu in code_units:
        if getattr(cu, "kind", "") == "function":
            funcs_by_source[cu.source_id].append(cu.name)

    for cm in code_modules:
        cm.code_domain = _classify_code_domain(
            cm.path, getattr(cm, "doc", ""), funcs_by_source.get(cm.source_id, [])
        )

    # LLM 精化(可选;不可用/失败则保留关键字结果)
    try:
        llm = _classify_code_domains_llm([{
            "path": cm.path,
            "name": Path(cm.path).name,
            "docstring": getattr(cm, "doc", "") or "",
            "source_head": "",
            "funcs": funcs_by_source.get(cm.source_id, []),
        } for cm in code_modules])
        for cm in code_modules:
            if cm.path in llm:
                cm.code_domain = llm[cm.path]
    except Exception:
        pass


def _load_code_modules() -> list[dict]:
    p = KB_DIR / "code_modules.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def build_code_map() -> dict:
    sources = json.loads((KB_DIR / "sources.json").read_text(encoding="utf-8"))
    units = json.loads((KB_DIR / "code_units.json").read_text(encoding="utf-8"))
    units_by_source: dict[str, list] = defaultdict(list)
    for u in units:
        units_by_source[u["source_id"]].append(u)
    source_by_id = {s["id"]: s for s in sources}
    modules = _load_code_modules()
    mod_by_source = {m["source_id"]: m for m in modules}

    # ── canonical file set: code_file, excluding .ipynb_checkpoints ──
    files = []
    for sid, culist in units_by_source.items():
        src = source_by_id.get(sid)
        if not src or src["source_type"] != "code_file":
            continue
        if ".ipynb_checkpoints" in src["path"]:
            continue
        text = _file_text(src["path"])
        funcs = sorted({cu["name"] for cu in culist if cu["kind"] == "function"})
        classes = sorted({cu["name"] for cu in culist if cu["kind"] == "class"})
        docstrings = [cu.get("doc", "") for cu in culist if cu.get("doc")]
        doc = " ".join(docstrings)
        # code_domain 来自 CodeModule(文件级建模);未命中则回退关键字分类。
        cm = mod_by_source.get(sid)
        code_domain = (cm.get("code_domain") or "") if cm else _classify_code_domain(src["path"], doc, funcs)
        domain, sub_domain = CODE_DOMAIN_LINK.get(code_domain, (None, None))
        files.append({
            "source_id": sid,
            "path": src["path"],
            "name": Path(src["path"]).name,
            "funcs": funcs,
            "classes": classes,
            "docstring": doc[:200],
            "source_head": "\n".join(text.splitlines()[:6]),
            "summary": _one_line_summary(src["path"], funcs, classes, doc),
            "naming_keys": _naming_key_matches(src["path"]),
            "hash": _content_hash(text),
            "literals": sorted({lit for cu in culist for lit in cu.get("literals", [])})[:10],
            "code_domain": code_domain,
            "domain": domain,
            "sub_domain": sub_domain,
        })

    # ── dedup: same content hash → group; note base/ copies ──
    by_hash: dict[str, list] = defaultdict(list)
    for f in files:
        by_hash[f["hash"]].append(f)
    for group in by_hash.values():
        if len(group) > 1:
            for f in group:
                f["duplicate_of"] = sorted(f["path"] for f in group if f is not group[0])

    # external repos: URL / gitlab-style references in literals
    external = sorted({lit for f in files for lit in f["literals"]
                       if "://" in lit or "gitlab" in lit.lower()})[:20]

    # ── grouping: code_domain → files ──
    code_domains: dict[str, list] = defaultdict(list)
    for f in sorted(files, key=lambda x: x["path"]):
        code_domains[f["code_domain"]].append(f)

    # LLM-verify the abbreviation dictionary against READMEs + weekly reports.
    naming_keys = _verify_naming_keys(sources)

    code_map = {
        "files": files,
        "code_domains": dict(code_domains),
        "naming_keys": naming_keys,
        "external_repos": external,
        "dedup_count": sum(1 for f in files if f.get("duplicate_of")),
        "base_copies": [f["path"] for f in files if "/base/" in f["path"] and f.get("duplicate_of")],
    }
    (KB_DIR / "code_map.json").write_text(json.dumps(code_map, ensure_ascii=False, indent=2), encoding="utf-8")
    return code_map


def _load_project_clusters() -> dict | None:
    """Load the adaptive clustering produced by `project_clusters.py` (P0–P2).

    Returns None if the file is absent (e.g. the renderer was invoked standalone
    before the clustering stage ran), so the map degrades gracefully.
    """
    try:
        return json.loads((KB_DIR / "project_clusters.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _mermaid_id(name: str, used: set[str]) -> str:
    """A collision-free Mermaid node id: sanitize to `[0-9a-zA-Z_]`, then append a
    counter if two distinct files collapse to the same id (e.g. two `utils.py` in
    different subdirs of a version family)."""
    base = re.sub(r"[^0-9a-zA-Z_]", "_", name) or "node"
    nid, i = base, 1
    while nid in used:
        i += 1
        nid = f"{base}_{i}"
    used.add(nid)
    return nid


def _render_project_landscape(clusters: list[dict]) -> list[str]:
    type_order = ["version_family", "project", "work_collection", "standalone"]
    type_title = {
        "version_family": "Version families",
        "project": "Projects",
        "work_collection": "Work collections",
        "standalone": "Standalone scripts",
    }
    lines = [
        f"## {t('project_landscape')} 🗺️",
        "",
        f"**{len(clusters)}** clusters discovered from the ingested code by the adaptive "
        "clustering process (content-hash dedup → import connectivity → frequency-aware "
        "name similarity → curated reconciliation). Each cluster carries the signal(s) "
        "that produced it plus a confidence score.",
        "",
    ]
    for t in type_order:
        group = [c for c in clusters if c["type"] == t]
        if not group:
            continue
        lines.append(f"### {type_title[t]} ({len(group)})\n")
        for c in group:
            conf = ("high" if c["confidence"] >= 0.7
                    else "medium" if c["confidence"] >= 0.5 else "low")
            cur = f" → **{', '.join(c['curated_refs'])}**" if c["curated_refs"] else ""
            sig = ", ".join(c["source_signals"])
            label = c.get("llm_label") or c["label"]
            lines.append(
                f"- **{label}** `{c['type']}` "
                f"(conf {c['confidence']:.2f} · {conf} · {c['file_count']} files · "
                f"{c['duplicate_count']} dup · {sig}){cur}"
            )
            for v in c["versions"]:
                if v["role"] == "duplicate" and v.get("dup_of"):
                    role = f"⚠️ duplicate of `{Path(v['dup_of']).name}`"
                else:
                    role = v["role"]
                lines.append(f"    - `{portable_code_path(v['dir'])}` — {v['files']} file(s), {role}")
        lines.append("")
    return lines


def _render_import_graph(clusters: list[dict]) -> list[str]:
    lines = [f"## {t('project_import_graph')} 🔗", ""]
    with_edges = [c for c in clusters if c["import_graph"]]
    if not with_edges:
        lines.append("_No within-project import edges detected._")
        lines.append("")
        return lines
    lines.append(
        "Dependencies *inside* each project / version-family, extracted from module "
        "imports. Dual representation: plain text + Mermaid.\n"
    )
    for c in with_edges:
        label = c.get("llm_label") or c["label"]
        lines.append(f"### {label}\n")
        lines.append("```")
        for e in c["import_graph"]:
            lines.append(f"{Path(e['from']).name} → {Path(e['to']).name}")
        lines.append("```")
        lines.append("")
        lines.append("```mermaid")
        lines.append("graph TD")
        used: set[str] = set()
        node_ids: dict[str, str] = {}
        for e in c["import_graph"]:
            for p in (e["from"], e["to"]):
                if p not in node_ids:
                    node_ids[p] = _mermaid_id(Path(p).stem, used)
            lines.append(f"    {node_ids[e['from']]} --> {node_ids[e['to']]}")
        lines.append("```")
        lines.append("")
    return lines


def _render_data_flow(data_flow: list[dict], clusters: list[dict]) -> list[str]:
    lines = [f"## {t('cross_project_data_flow')} 🌐", ""]
    if not data_flow:
        lines.append("_No data paths shared by ≥2 clusters detected._")
        lines.append("")
        return lines
    lines.append(
        "Data paths (mounted HDFS dirs, configs, data files) referenced by files in "
        "**two or more** clusters — the 'who touches the same data' signal, "
        "independent of import structure.\n"
    )
    display = {c["id"]: (c.get("llm_label") or c["label"]) for c in clusters}
    lines.append("| Data path | Clusters |")
    lines.append("|---|---|")
    for df in data_flow:
        labels = ", ".join(display.get(c["id"], c["label"]) for c in df["clusters"])
        lines.append(f"| `{df['path']}` | {labels} |")
    lines.append("")
    return lines


def render_code_map() -> None:
    """Render artifact/index/code/index.md from code_map.json (replaces directory mirror).

    This is the high-level functional map: naming keys + code domains + dedup.
    Each file entry links to its per-file CodeModule page (`source_*.py.md`), which
    carries the function-level detail + doc cross-references. The raw flat registry
    is `code-units.md` — a grep-first surface that also links back to CodeModule pages.
    """
    code_map = build_code_map()
    files = code_map["files"]
    code_domains = code_map["code_domains"]
    total_funcs = sum(len(f["funcs"]) + len(f["classes"]) for f in files)
    pc = _load_project_clusters()

    lines = [
        "# Code Map",
        "",
        f"**{len(files)}** canonical Python files, **{total_funcs}** functions/classes, "
        f"**{code_map['dedup_count']}** duplicate copies deduplicated.",
        "",
        "> Some research-side code lives in an external repo — see [External repos](#external-repos).",
        "> Each file below links to its per-file Code page (function-level detail). "
        "The flat grep-first list is in [code-units.md](code-units.md).",
        "",
        f"## {t('naming_keys')} 🔑",
        "",
    ]
    for key, meaning in code_map["naming_keys"].items():
        lines.append(f"- **{key}** — {meaning}")
    lines.append("- Abbreviations that cannot be verified from evidence are marked ⚠️ unverified\n")

    lines.append(f"## {t('by_code_domain')}\n")
    lines.append("Each code domain is a code-specific refinement of a [domain](../../index.md#domains) "
                 "in the taxonomy (see the domain/sub-domain link after each heading).\n")
    for cd in sorted(code_domains.keys()):
        flist = code_domains[cd]
        dlink = CODE_DOMAIN_LINK.get(cd)
        suffix = ""
        if dlink and dlink[0]:
            dom_link = f"[{dlink[0]}](../../domains/{dlink[0]}/index.md)"
            suffix = f" — {dom_link}" + (f" · {dlink[1]}" if dlink[1] else "")
        lines.append(f"### {cd} ({len(flist)}){suffix}\n")
        for f in flist:
            page_name = f["path"].replace("/", "_") + ".md"
            dup = f" ⚠️ {len(f['duplicate_of'])} copies" if f.get("duplicate_of") else ""
            nk = f" `{'/'.join(f['naming_keys'])}`" if f["naming_keys"] else ""
            lines.append(f"- [{f['name']}]({page_name}) — {f['summary']}{nk}{dup}")
        lines.append("")

    if pc and pc.get("clusters"):
        lines.extend(_render_project_landscape(pc["clusters"]))
        lines.extend(_render_import_graph(pc["clusters"]))
        lines.extend(_render_data_flow(pc.get("data_flow", []), pc["clusters"]))

    lines.append(f"## {t('known_duplicates')}\n")
    dup_groups = defaultdict(list)
    for f in files:
        if f.get("duplicate_of"):
            dup_groups[f["name"]].append(f["path"])
    if dup_groups:
        lines.append("| File | Locations |")
        lines.append("|---|---|")
        for name, paths in sorted(dup_groups.items()):
            lines.append(f"| {name} | {', '.join(portable_code_path(p) for p in paths)} |")
    else:
        lines.append("_None_")
    lines.append("")

    lines.append(f"## {t('external_repos')} ⚠️\n")
    if code_map["external_repos"]:
        for r in code_map["external_repos"]:
            lines.append(f"- `{r}`")
    else:
        lines.append("_No explicit external URLs detected (can supplement from project docs' GitLab references)_")
    lines.append("")

    lines.append(f"## {t('see_also')}\n")
    lines.append("- [Knowledge index](../index.md)")
    lines.append("- [All entity types](../index.md#by-entity-type)")
    lines.append("- [Code analysis](analysis.md) — call graph, dependency graph, CFG, cyclomatic complexity")
    lines.append("- [Code units registry](code-units.md) — flat grep-first list of all functions/classes")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = f"---\ntype: CodeMap\ntitle: Code Map\ndescription: {len(files)} code files, functional map\n"
    out += f"timestamp: {ts}\n---\n\n" + "\n".join(lines) + "\n"
    code_out = INDEX_DIR / "code"
    code_out.mkdir(parents=True, exist_ok=True)
    (code_out / "index.md").write_text(out, encoding="utf-8")
    print(f"Code map → {code_out / 'index.md'} ({len(files)} files, {code_map['dedup_count']} dup)")

    # leaf pages: still rendered by render_code_index (kept), but index.md is now the map.


if __name__ == "__main__":
    render_code_map()
