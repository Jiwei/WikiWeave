"""Hub generator (Stage C, deterministic-first) — theme bundles + sub-domain labels.

Turns the flat domain page into a bundle-structured hub. A **theme bundle** is a
semantic cluster of content anchored on entities — primarily **Projects** (curated
工作线 from source/project/) plus generated task-family clusters for tasks that do
not map to a curated project. Every bundle carries two facet labels:

  - `sub_domains[]` — which sub-domain(s) of its domain the bundle belongs to
  - `types[]`       — which entity types anchor it (project / task / artifact /
                      workflow / protocol / error)

Deterministic-first: no LLM. Bundles are assembled from the structured data layer
(projects.json + phase2_*.json + doc_sections.json) with keyword signatures. The
result is written to knowledge_bits/hub_data.json and consumed by the renderer.

Usage:
    python -m indexing.hubgen
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from util import model_data
from util.i18n import prompt_language
from util.paths import kb_dir

KB_DIR = kb_dir()

# ── Sub-domain vocabulary (curated, deterministic) ──
# Each domain → list of (sub_domain_label, [keywords]). Keywords are lowercase
# substrings matched against section name + content. Derived from the domain
# definitions' typical_activities.
# Each domain → list of (zh_label, en_label, [keywords]). The English label is
# the display name used in hub headings; the Chinese label + keywords still drive
# deterministic matching against section content.
# ── 子领域词汇:单一事实源 model_data/domains/sub_domains.json ──
SUB_DOMAINS = model_data.load_or_empty("domains/sub_domains", {})
def _text(*parts: str) -> str:
    return " ".join(p for p in parts if p).lower()


def detect_sub_domains(domain: str, text: str) -> list[str]:
    """Return matched sub-domain English labels for content within a domain."""
    tl = text.lower()
    hits = []
    for _zh, en, kws in SUB_DOMAINS.get(domain, []):
        if any(k.lower() in tl for k in kws):
            hits.append(en)
    return hits


def _stable_id(prefix: str, key: str) -> str:
    import hashlib
    return f"{prefix}-{hashlib.sha1(key.encode()).hexdigest()[:12]}"


HOOK_PROMPT = """Output language: {lang}.

You are writing one-line navigation hooks for theme bundles in a knowledge base. Each bundle groups related sections (weekly reports, project specs, code) around a work line or deliverable.

For each bundle below, write a hook that answers "what will I learn by reading this bundle?" — a specific, non-generic one-liner (8-18 words). The hook must mention concrete entities/terms from the bundle (e.g. a model name, a factor set, a bug, a workflow), not just restate the title.

Return a JSON object mapping bundle_id → hook, e.g. {{"project:model_x": "How the pipeline ingests raw data and emits predictions."}}.

Bundles:
{items}"""


def _llm_hooks(bundles: list[dict], sections_by_id: dict[str, dict]) -> None:
    """LLM-write one-line hooks for substantive bundles (≥5 sections).

    Deterministic fallback: keeps the existing hook (task_type / kind text).
    Hooks carry no section links by default, but the bundle's `pointers` still
    carry entry section ids for provenance.
    """
    # TODO(搬移): 新 util.llm 接口为 chat/extract_json/available,不再提供 LlmConfig/_call_llm;算法逻辑保留未改,后续按新接口适配
    from util.llm import LlmConfig, _call_llm
    import json as _json
    import re as _re

    cfg = LlmConfig.from_env()
    if not cfg.available:
        return

    target = [b for b in bundles
              if b.get("anchor") != "misc" and len(b.get("member_section_ids", [])) >= 5]
    if not target:
        return

    items = []
    for b in target:
        # pick up to 5 representative section names as content signal
        names = []
        for sid in b.get("member_section_ids", []):
            ds = sections_by_id.get(sid, {})
            nm = ds.get("name", "")
            if nm:
                names.append(nm[:40])
            if len(names) >= 5:
                break
        item = (f"- id: {b['id']} | title: {b.get('title','')} | "
                f"types: {','.join(b.get('types',[]))} | sections: {' / '.join(names)}")
        items.append(item)

    prompt = HOOK_PROMPT.format(items="\n".join(items), lang=prompt_language())
    try:
        raw = _call_llm(prompt, cfg)
    except Exception as e:
        print(f"  hook LLM failed ({e}) — keeping deterministic hooks", flush=True)
        return
    m = _re.search(r"\{.*\}", raw, _re.DOTALL)
    if not m:
        return
    try:
        hooks = _json.loads(m.group())
    except Exception:
        return
    if not isinstance(hooks, dict):
        return
    n = 0
    for b in target:
        h = hooks.get(b["id"])
        if isinstance(h, str) and h.strip():
            b["hook"] = h.strip()[:140]
            n += 1
    if n:
        print(f"  hook LLM: wrote {n} bundle hooks", flush=True)


# ── Bundle construction ──

def build_hub_data(
    doc_sections: list[dict],
    entities: dict[str, list[dict]],
) -> dict:
    """Build per-domain theme bundles + sub-domain labels → hub_data structure.

    Deterministic: projects are the primary anchors (工作线). Tasks that do not
    clearly map to a curated project are clustered into generated task-family
    bundles by (domain × asset × person). Errors/workflows/protocols/artifacts
    are folded into the bundle whose member sections they share.

    实体列表统一从 entities({list_key: [dict]})取,不再按实体名逐一传参。
    """
    projects = entities.get("projects", [])
    tasks = entities.get("tasks", [])
    errors = entities.get("errors", [])
    workflows = entities.get("workflows", [])
    protocols = entities.get("protocols", [])
    artifacts = entities.get("artifacts", [])

    sections_by_id = {s["id"]: s for s in doc_sections}

    # ── C0 helper: bridge protocol cu- ids to doc sections via code_units' source ──
    # Protocols anchor to code units (src-xxx-cu-N), not doc sections. To give
    # them a doc-section footprint, map each cu → its source file → that source's
    # sections. Fall back to the source_id prefix's sections when the cu is absent.
    def _bridge_cu_ids(cu_ids: list[str]) -> list[str]:
        out = set()
        for cid in cu_ids:
            if "-sec-" in cid and cid in sections_by_id:
                out.add(cid)
                continue
            if "-cu-" in cid:
                src_prefix = cid.rsplit("-cu-", 1)[0]  # src-xxx
                # find all sections of that source
                for dsid, ds in sections_by_id.items():
                    if ds["source_id"] == src_prefix:
                        out.add(dsid)
        return sorted(out)

    # ── 1. Seed bundles (C2): substantive anchors seed; thin entities fold in ──
    # Bundle types = entity types. But not every entity instance should seed a
    # bundle — the data shows errors (572/674 with 1 section), workflows (all
    # 1-2 sections) and protocols (0 bridged doc sections) are too thin to be
    # primary anchors. So we seed PROJECT + ARTIFACT bundles (substantive),
    # then FOLD thin error/workflow/protocol records into the best-overlap
    # bundle with a types[] tag. C3 then merges overlapping seeds into
    # multi-type bundles (e.g. project + artifact → a work line plus its deliverable).
    tasks_by_id = {t["id"]: t for t in tasks}
    bundles: list[dict] = []
    task_bundle_links: dict[str, str] = {}  # task_id → bundle_id

    def _seed(seed_type: str, name: str, secs: list[str], owner: str = "",
              hook: str = "", source: str = "curated") -> dict:
        b = {
            "id": f"{seed_type}:{name}",
            "kind": "entity",
            "anchor": seed_type,
            "title": name,
            "owner": owner,
            "sub_domains": [],   # filled below per domain
            "types": [seed_type],
            "member_section_ids": sorted(set(secs)),
            "time_span": "",
            "hook": hook,
            "source": source,
        }
        bundles.append(b)
        return b

    # 1a. project seeds (curated + generated) — members = spec + rolled-up task sections
    for p in projects:
        pname = p["name"]
        psecs = set(p.get("source_sections", []))
        for tid in p.get("related_tasks", []):
            t = tasks_by_id.get(tid)
            if t:
                psecs.update(t.get("source_sections", []))
                task_bundle_links[tid] = f"project:{pname}"
        origin = p.get("origin", "curated")
        # bundle title = semantic title (LLM-derived for curated), fallback codename
        btitle = p.get("title") or pname
        b = _seed("project", btitle, sorted(psecs), owner=p.get("owner", ""),
                  hook=(p.get("task_type", "") or p.get("output_positioning", "")
                        or f"{len(p.get('related_tasks', []))} tasks"),
                  source=origin)
        b["name"] = pname  # codename for linking to the project page

    # 1b. artifact seeds — members = artifact's source_sections (+ its project's sections)
    for a in artifacts:
        asecs = set(a.get("source_sections", []))
        for pid in a.get("related_projects", []):
            for p in projects:
                if p.get("id") == pid:
                    asecs.update(p.get("source_sections", []))
        if not asecs:
            continue  # skip artifacts with no reachable sections
        _seed("artifact", a.get("name", "?"), sorted(asecs),
              owner=a.get("owner", ""), hook=f"{a.get('kind','')} artifact",
              source="artifact")

    # 1c'. source-doc seeds (C2: reference docs / code readmes are anchors)
    # Every doc/readme source is a stable identity + domain anchor → seed it.
    # C3 merge and the hub's top-N rendering keep the hub from fragmenting even
    # though many small doc bundles exist in the data layer. This guarantees
    # their sections are reachable (not orphaned into misc).
    sections_by_source: dict[str, list[str]] = {}
    for sec in doc_sections:
        sections_by_source.setdefault(sec["source_id"], []).append(sec["id"])
    doc_sources = sorted({sec["source_id"] for sec in doc_sections
                          if sec.get("kind") in ("heading_block", "font_heading_block", "text_block")})
    for src_id in doc_sources:
        secs = sections_by_source.get(src_id, [])
        if not secs:
            continue
        name = next((sec["name"] for sec in doc_sections if sec["id"] == secs[0]), src_id)
        _seed("document", f"{name[:60]}", secs, owner="", hook="参考文档/readme",
              source="document")

    # 1c. fold thin entity records into best-overlap bundle (types[] tag)
    # Errors/workflows are thin (1-2 sections) and protocols are code-anchored;
    # fold each into the bundle whose member sections it overlaps most, tagging
    # its type. Protocols' cu- ids are bridged to their source's sections first.
    def _fold(records: list[dict], etype: str, label_fn, bridge=False):
        for r in records:
            raw = r.get("source_sections", [])
            rsecs = _bridge_cu_ids(raw) if bridge else [sid for sid in raw if sid in sections_by_id]
            if not rsecs:
                continue
            best_b, best_overlap = None, 0
            for b in bundles:
                ov = len(set(b["member_section_ids"]) & set(rsecs))
                if ov > best_overlap:
                    best_b, best_overlap = b, ov
            if best_b and best_overlap > 0:
                best_b["member_section_ids"] = sorted(set(best_b["member_section_ids"]) | set(rsecs))
                if etype not in best_b["types"]:
                    best_b["types"].append(etype)
                if r.get("id"):
                    best_b.setdefault("entities", []).append(f"{etype}:{label_fn(r)}")

    # ── fold 薄实体(由 entities.json 的 render.hub_role=fold 声明驱动;seed 体见上方实体专属 hook) ──
    from util.ontology import all_entities
    for _name, _cls in all_entities().items():
        _rcfg = getattr(_cls, "_render", {})
        if _rcfg.get("hub_role") != "fold":
            continue
        _lk = getattr(_cls, "_list_key", "")
        _etype = _rcfg.get("kind_label", _name).lower()
        _tf = _rcfg.get("title_field", "name")
        _bridge = _rcfg.get("hub_bridge", False)
        _fold(entities.get(_lk, []), _etype,
              lambda r, tf=_tf: (r.get(tf) or r.get("id", ""))[:40], bridge=_bridge)

    # ── 2. C3 merge: overlapping seeds → multi-type bundles ──
    # Merge smaller seed B into larger A when B is SUBSTANTIALLY contained in A:
    #   overlap ≥ K_ABS sections (3) AND overlap ≥ K_REL * min(|A|,|B|) (50%).
    # The relative term prevents a big project bundle from absorbing every
    # artifact/workflow that merely shares a few task sections — that produced
    # one mega-bundle (with 2337 sections). Keeps theme bundles cohesive.
    K_ABS = 3
    K_REL = 0.5
    changed = True
    while changed:
        changed = False
        for i in range(len(bundles)):
            bi = bundles[i]
            if bi is None:
                continue
            best_j, best_score = -1, 0.0
            ai = set(bi["member_section_ids"])
            for j in range(len(bundles)):
                if i == j or bundles[j] is None:
                    continue
                bj = bundles[j]
                aj = set(bj["member_section_ids"])
                overlap = len(ai & aj)
                if overlap < K_ABS:
                    continue
                rel = overlap / max(len(ai), len(aj))
                if rel < K_REL:
                    continue
                # prefer merging into the larger bundle
                score = overlap + 0.001 * min(len(ai), len(aj))
                if score > best_score:
                    best_j, best_score = j, score
            if best_j >= 0:
                bj = bundles[best_j]
                bj["member_section_ids"] = sorted(ai | set(bj["member_section_ids"]))
                for t in bi["types"]:
                    if t not in bj["types"]:
                        bj["types"].append(t)
                if bi["owner"] and not bj["owner"]:
                    bj["owner"] = bi["owner"]
                if bi["hook"] and not bj["hook"]:
                    bj["hook"] = bi["hook"]
                if bi.get("entities"):
                    bj.setdefault("entities", []).extend(bi["entities"])
                bundles[i] = None
                changed = True
    bundles = [b for b in bundles if b is not None]

    # ── 4. Compute per-bundle domain membership ──
    # A bundle can span multiple domains (its sections are multi-labeled). For
    # each domain it belongs to, assign sub_domain labels using ONLY that
    # domain's vocabulary, so labels never leak across domains.
    for b in bundles:
        b["domains"] = sorted({d for sid in b["member_section_ids"]
                               for d in (sections_by_id.get(sid, {}).get("domains") or []) if d != "others"})
        bundle_text = _text(*[sections_by_id.get(sid, {}).get("name", "") for sid in b["member_section_ids"]])
        b["sub_domains_by_domain"] = {
            d: detect_sub_domains(d, bundle_text) or ["Uncategorized"]
            for d in b["domains"]
        }

    # ── 3 (after domains). C4 full assignment (zero orphans at section level) ──
    # Any section not yet in any bundle → assign to the best-overlap bundle.
    # Score = same-source sections already in the bundle (primary) + shared
    # domain (secondary). This captures thin sections that share a source file
    # with a bundled section (e.g. remaining sections of a weekly report).
    # Sections with neither signal stay as "杂项" (appendix, not hub).
    covered_secs = set()
    for b in bundles:
        covered_secs.update(b["member_section_ids"])
    # index: source_id → which bundles already contain sections of that source
    source_in_bundle: dict[str, dict[str, int]] = {}
    for b in bundles:
        for sid in b["member_section_ids"]:
            src = sections_by_id.get(sid, {}).get("source_id")
            if not src:
                continue
            if src not in source_in_bundle:
                source_in_bundle[src] = {}
            source_in_bundle[src][b["id"]] = source_in_bundle[src].get(b["id"], 0) + 1

    orphan_secs = [sid for sid in sections_by_id if sid not in covered_secs]
    unassignable = 0
    for sid in orphan_secs:
        ds = sections_by_id.get(sid, {})
        ds_doms = set(ds.get("domains") or [])
        src = ds.get("source_id")
        best_b, best_score = None, -1.0
        for b in bundles:
            score = 0.0
            # primary: how many sections of the same source are already in b
            if src:
                score += 2.0 * source_in_bundle.get(src, {}).get(b["id"], 0)
            # secondary: shared domain
            score += len(ds_doms & set(b.get("domains", [])))
            if score > best_score:
                best_b, best_score = b, score
        if best_b and best_score > 0:
            best_b["member_section_ids"].append(sid)
            best_b["member_section_ids"] = sorted(set(best_b["member_section_ids"]))
            if src:
                if src not in source_in_bundle:
                    source_in_bundle[src] = {}
                source_in_bundle[src][best_b["id"]] = source_in_bundle[src].get(best_b["id"], 0) + 1
        else:
            unassignable += 1
    if unassignable:
        _MISC = {"id": "misc", "kind": "entity", "anchor": "misc",
                 "title": "Misc (uncategorized sections)", "owner": "",
                 "sub_domains": ["Uncategorized"], "types": ["misc"],
                 "member_section_ids": [sid for sid in orphan_secs],
                 "time_span": "", "hook": "无法归入任何主题束的散件", "source": "misc",
                 "domains": sorted({d for sid in orphan_secs
                                    for d in (sections_by_id.get(sid, {}).get("domains") or []) if d != "others"}),
                 "sub_domains_by_domain": {"Uncategorized": ["Uncategorized"]}}
        bundles.append(_MISC)

    # ── C6: per-bundle time_span + section pointers (deterministic) ──
    for b in bundles:
        weeks = [sections_by_id.get(sid, {}).get("iso_week", "")
                 for sid in b["member_section_ids"] if sections_by_id.get(sid, {}).get("iso_week")]
        weeks = [w for w in weeks if w]
        if weeks:
            b["time_span"] = f"{min(weeks)} – {max(weeks)}"
        else:
            b["time_span"] = ""
        # pointers: entry = the largest/most-cited section; others = first few
        member = b["member_section_ids"]
        if member:
            # prefer sections with people (weekly reports) or code mentions as entry
            def _weight(sid):
                ds = sections_by_id.get(sid, {})
                return (1 if ds.get("people") else 0) + (1 if ds.get("code_mentions") else 0)
            entry = max(member, key=lambda sid: (_weight(sid), sid))
            others = [sid for sid in member if sid != entry][:4]
            b["pointers"] = {"entry": entry, "sections": others}
        else:
            b["pointers"] = {"entry": "", "sections": []}

    # ── C6 (LLM): human-readable one-line hooks for substantive bundles ──
    _llm_hooks(bundles, sections_by_id)

    # ── 5. Per-domain hub payloads ──
    domains = sorted({d for b in bundles for d in b["domains"]})
    # entry_points = top substantive bundles per domain; failure_modes = top errors
    err_by_domain: dict[str, list[dict]] = defaultdict(list)
    for e in errors:
        for d in (e.get("domains") or []):
            if d != "others":
                err_by_domain[d].append(e)
    hub = {}
    for dom in domains:
        dom_bundles = [b for b in bundles if dom in b["domains"]]
        # sort by size (desc) as a deterministic "cognitive value" proxy
        dom_bundles.sort(key=lambda b: len(b["member_section_ids"]), reverse=True)
        # failure modes: top 5 errors in this domain, by section count
        dom_errs = sorted(err_by_domain.get(dom, []),
                          key=lambda e: -len(e.get("source_sections", [])))[:5]
        # entry points: top substantive bundles (id + title)
        entry_points = [
            {"id": b["id"], "title": b["title"]}
            for b in dom_bundles if b.get("anchor") != "misc" and b.get("member_count", 0) >= 5
        ][:5]
        hub[dom] = {
            "sub_domains": [en for _zh, en, _kws in SUB_DOMAINS.get(dom, [])],
            "failure_modes": [{"id": e.get("id"), "title": e.get("title", "")[:60]} for e in dom_errs],
            "entry_points": entry_points,
            "theme_bundles": [
                {
                    "id": b["id"],
                    "title": b["title"],
                    "name": b.get("name", ""),   # codename for project links
                    "kind": b["kind"],
                    "anchor": b["anchor"],
                    "owner": b["owner"],
                    "domains": b["domains"],
                    # this domain's sub-domain labels only
                    "sub_domains": b["sub_domains_by_domain"].get(dom, ["Uncategorized"]),
                    "types": b["types"],
                    "hook": b["hook"],
                    "member_count": len(b["member_section_ids"]),
                    "time_span": b["time_span"],
                    "pointers": b["pointers"],
                    "source": b["source"],
                }
                for b in dom_bundles
            ],
            # distribution: sub_domain → bundle ids (this domain's labels only)
            "sub_domain_bundles": defaultdict(list),
        }
        for b in dom_bundles:
            for sd in b["sub_domains_by_domain"].get(dom, ["Uncategorized"]):
                hub[dom]["sub_domain_bundles"][sd].append(b["id"])
        hub[dom]["sub_domain_bundles"] = dict(hub[dom]["sub_domain_bundles"])

    return hub


def main() -> None:
    def _load(name: str) -> list:
        p = KB_DIR / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

    hub = build_hub_data(
        doc_sections=_load("doc_sections.json"),
        entities={
            "projects": _load("projects.json"),
            "tasks": _load("phase2_tasks.json"),
            "errors": _load("phase2_errors.json"),
            "workflows": _load("phase2_workflows.json"),
            "protocols": _load("phase2_protocols.json"),
            "artifacts": _load("artifacts.json"),
        },
    )
    KB_DIR.mkdir(parents=True, exist_ok=True)
    (KB_DIR / "hub_data.json").write_text(json.dumps(hub, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"hub_data.json written — {len(hub)} domains")
    for dom, payload in hub.items():
        n = len(payload["theme_bundles"])
        nproj = sum(1 for b in payload["theme_bundles"] if b["anchor"] == "project")
        nart = sum(1 for b in payload["theme_bundles"] if b["anchor"] == "artifact")
        nmisc = sum(1 for b in payload["theme_bundles"] if b["anchor"] == "misc")
        print(f"  {dom}: {n} bundles ({nproj} project, {nart} artifact, {nmisc} misc) · sub-domains: {len(payload['sub_domains'])}")


if __name__ == "__main__":
    main()
