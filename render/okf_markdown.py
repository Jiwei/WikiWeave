"""Render the knowledge index from knowledge_bits — all pages follow Google OKF format."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from util import model_data
from util.i18n import t
from util.ontology import (
    CodeUnitRecord, DocSectionRecord, DomainRecord, SourceRecord, get as get_entity,
)
from util.paths import index_dir, kb_dir, portable_code_path

INDEX_DIR = index_dir()
KB_DIR = kb_dir()

_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fm(typ: str, title: str = "", description: str = "", tags: list[str] | None = None) -> str:
    lines = ["---", f"type: {typ}"]
    if title: lines.append(f"title: {title}")
    if description: lines.append(f"description: {description}")
    if tags: lines.append(f"tags: [{', '.join(tags)}]")
    lines.append(f"timestamp: {_TIMESTAMP}")
    lines.append("---\n")
    return "\n".join(lines)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _rel_link(target: str, depth: int) -> str:
    """Generate a relative markdown link, URL-encoding problematic characters in the URL."""
    import urllib.parse
    # URL-encode parentheses and spaces in the path portion
    encoded = target
    # Only encode if the path has problematic characters for markdown links
    if any(c in target for c in "() "):
        parts = target.split("/")
        encoded = "/".join(urllib.parse.quote(p, safe="") if any(c in p for c in "() ") else p for p in parts)
    prefix = "/".join([".."] * depth)
    return f"[{target}]({prefix}/{encoded})"


def _source_link_or_text(path: str, depth: int) -> str:
    """Render a source reference: portable plain text for code sources (not bundled in
    the artifact), a relative markdown link otherwise."""
    if path.startswith("code/"):
        return portable_code_path(path)
    return _rel_link(path, depth)


def _count_json(name: str) -> int:
    """Read a knowledge_bits JSON and return its record count (0 if absent)."""
    import json as _json
    p = KB_DIR / name
    if not p.exists():
        return 0
    try:
        return len(_json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return 0


def _source_type_labels() -> dict:
    """source_type → 人类可读标签(单一事实源:model_data/sources/source_types.json 的 labels)。"""
    return model_data.source_type_rules().get("labels", {})


def _source_type_reading_guides() -> dict:
    """source_type → 阅读指南(单一事实源:model_data/sources/source_types.json 的 reading_guides)。"""
    return model_data.source_type_rules().get("reading_guides", {})


def _load_json(name: str) -> list:
    """Read a knowledge_bits JSON as a list ([] if absent)."""
    import json as _json
    p = KB_DIR / name
    if not p.exists():
        return []
    try:
        data = _json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _render_domain_f1(dname: str, dlabel: str) -> str:
    """Render F1 hub sections (30-sec panorama, model lineage, failure modes,
    by-entity-type browse) from hub_data.json + artifacts + errors.

    Deterministic summaries (the design's LLM versions are a later enhancement).
    Returns "" when no hub_data for this domain.
    """
    import json as _json
    hub_path = KB_DIR / "hub_data.json"
    if not hub_path.exists():
        return ""
    try:
        hub = _json.loads(hub_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    payload = hub.get(dname)
    if not payload:
        return ""
    lines = []

    # ── 30-second panorama (deterministic from sub-domain + top bundles) ──
    bundles = payload.get("theme_bundles", [])
    n_proj = sum(1 for b in bundles if b.get("anchor") == "project")
    n_art = sum(1 for b in bundles if b.get("anchor") == "artifact")
    sds = payload.get("sub_domains", [])
    top = sorted(bundles, key=lambda b: -b.get("member_count", 0))[:3]
    top_desc = "; ".join(f"{b['title']} ({b.get('member_count',0)} sections)" for b in top if b.get("anchor") != "misc")
    lines.append("## 30-Second Overview\n")
    lines.append(
        f"This domain has **{len(bundles)}** theme bundles (**{n_proj}** work lines / "
        f"**{n_art}** deliverables) across sub-domains: {', '.join(sds) or '—'}. "
        f"Top work lines: {top_desc or '—'}.\n"
    )

    # ── Model lineage (artifact kind=model) ──
    arts = _load_json("artifacts.json")
    models = [a for a in arts if a.get("kind") == "model"]
    # match by domain via asset classes is weak; show models whose sections are in this domain
    sections_by_id = {}
    if (KB_DIR / "doc_sections.json").exists():
        for ds in _json.loads((KB_DIR / "doc_sections.json").read_text()):
            sections_by_id[ds["id"]] = ds
    dom_models2 = []
    for m in models:
        msecs = m.get("source_sections", [])
        if any(dname in (sections_by_id.get(s, {}).get("domains") or []) for s in msecs):
            dom_models2.append(m)
    if dom_models2:
        lines.append("## Model Lineage\n")
        for m in sorted(dom_models2, key=lambda x: x["name"]):
            slug = _slug(m["name"])
            owner = f" · {m.get('owner','')}" if m.get("owner") else ""
            lines.append(f"- [{m['name']}](../../artifacts/model/{slug}.md){owner} — {m.get('summary','')[:60]}")
        lines.append("")

    # ── Common failure modes (from hub_data.json — single writer) ──
    dom_errs = payload.get("failure_modes", [])
    if dom_errs:
        lines.append("## Common Failure Modes\n")
        for e in dom_errs:
            lines.append(f"- [{e.get('title','')[:60]}](../../infos/errors/{e['id']}.md)")
        lines.append("")

    # ── By-entity-type browse (compact index) ──
    by_type = {}
    for b in bundles:
        if b.get("anchor") == "misc":
            continue
        for tp in b.get("types", []):
            by_type[tp] = by_type.get(tp, 0) + 1
    if by_type:
        links = []
        # entity type → page dir(由 entities.json 的 render 驱动;document 为特殊锚点)
        from util.ontology import all_entities
        type_map = {}
        for _n, _c in all_entities().items():
            _r = getattr(_c, "_render", {})
            if _r:
                type_map[_r.get("kind_label", _n).lower()] = f"{_r.get('page_dir', '')}/index.md"
        type_map["document"] = "sections-all.md"
        for tp in sorted(by_type):
            rel = type_map.get(tp, "infos/index.md")
            # sections-all.md is a sibling of the domain index, not under root
            prefix = "" if rel == "sections-all.md" else "../../"
            links.append(f"[{tp}×{by_type[tp]}]({prefix}{rel})")
        lines.append("## Browse by Entity Type\n")
        lines.append(" · ".join(links) + "\n")

    return "\n".join(lines) + "\n"


def _render_domain_bundles(dname: str) -> str:
    """Render the Theme Bundles section for a domain hub, from hub_data.json.

    Bundles are the design's "主题束": content clusters anchored on entities —
    primarily curated **Projects** (工作线) plus generated task-family clusters.
    Each bundle row carries its sub-domain label(s) and entity-type tags. Tasks
    live inside bundles (as members), so the hub no longer lists raw task dumps.
    Returns "" if hub_data.json is absent for this domain.
    """
    import json as _json
    hub_path = KB_DIR / "hub_data.json"
    if not hub_path.exists():
        return ""
    try:
        hub = _json.loads(hub_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    payload = hub.get(dname)
    if not payload or not payload.get("theme_bundles"):
        return ""

    lines = ["", "## Theme Bundles", ""]
    anchors = {}
    for b in payload["theme_bundles"]:
        anchors[b.get("anchor", "?")] = anchors.get(b.get("anchor", "?"), 0) + 1
    # only count substantive bundles (exclude tiny doc bundles and misc) for the header
    substantive = [b for b in payload["theme_bundles"]
                   if b.get("anchor") != "misc" and b.get("member_count", 0) >= 5]
    anchor_desc = " · ".join(f"{k}={v}" for k, v in sorted(anchors.items()) if k != "misc")
    lines.append(f"Bundles = entity-anchored clusters (multi-type mergeable). Distribution: {anchor_desc} "
                 f"({len(substantive)} substantive bundles ≥5 sections)\n")

    # Group bundles by sub-domain — but each bundle is listed only under its
    # PRIMARY sub-domain (first in sub_domains[], the highest-scoring match).
    # Cross-domain bundles (spanning >1 sub-domain) go to a single "Cross-cutting"
    # section so they're not duplicated across every sub-domain group.
    declared = payload.get("sub_domains", [])
    grouped: dict[str, list] = {}
    cross_cutting: list[dict] = []
    for b in payload["theme_bundles"]:
        sds = b.get("sub_domains", ["Uncategorized"])
        if b.get("anchor") == "misc":
            continue
        if len(sds) > 1:
            cross_cutting.append(b)
        else:
            grouped.setdefault(sds[0], []).append(b)

    def _bundle_row(b: dict) -> str:
        types = ",".join(b.get("types", [])) or "?"
        owner = b.get("owner") or ""
        hook = (b.get("hook") or "").strip()
        cnt = b.get("member_count", 0)
        anchor = b.get("anchor", b.get("kind", "?"))
        span = b.get("time_span", "")
        # link to the project page when the bundle is project-anchored
        if anchor == "project":
            link = f"[**{b['title']}**](../../projects/{b.get('name', b['title'])}.md)"
        else:
            link = f"**{b['title']}**"
        meta = f"`[{types}]`"
        if owner:
            meta += f" {owner}"
        if span:
            meta += f" [{span}]"
        if hook:
            meta += f" — {hook}"
        return f"- {link} {meta} · {cnt} sections"

    for sd in declared + [s for s in grouped if s not in declared]:
        blist = grouped.get(sd, [])
        if not blist:
            continue
        lines.append(f"### {sd} ({len(blist)})\n")
        # project/artifact work lines first, then docs, by size desc (misc excluded from hub)
        order = [b for b in blist if b.get("anchor") != "misc"]
        order.sort(key=lambda b: (b.get("anchor") not in ("project", "artifact"), -b.get("member_count", 0)))
        for b in order[:10]:
            lines.append(_bundle_row(b))
        if len(order) > 10:
            lines.append(f"- ... and {len(order) - 10} more bundles in this sub-domain")
        lines.append("")

    # Cross-cutting bundles: span >1 sub-domain, listed once (not duplicated per group).
    if cross_cutting:
        lines.append(f"### Cross-cutting ({len(cross_cutting)})\n")
        cc = sorted(cross_cutting, key=lambda b: -b.get("member_count", 0))
        for b in cc[:10]:
            lines.append(_bundle_row(b))
        if len(cc) > 10:
            lines.append(f"- ... and {len(cc) - 10} more cross-cutting bundles")
        lines.append("")

    lines.append("_Raw flat entity lists (tasks/errors/workflows/protocols) are in [infos/](../../infos/index.md)._  "
                 "_Uncategorized sections are in [sections-all](sections-all.md)._\n")
    return "\n".join(lines)


def _render_domain_topics(dname: str, ddir: Path, appendix_link: str) -> None:
    """Render topics/{sub_domain}.md cross-view pages + big-bundle detail pages.

    Each per-sub-domain page shows all bundles in that sub-domain grouped by
    entity type (the design's "double-dimension cross view"), and links each
    bundle. Large bundles (> 40 sections) also get a topics/{bundle}.md detail
    page. Deterministic — all from hub_data.json.
    """
    import json as _json
    hub_path = KB_DIR / "hub_data.json"
    if not hub_path.exists():
        return
    try:
        payload = _json.loads(hub_path.read_text(encoding="utf-8")).get(dname)
    except Exception:
        return
    if not payload or not payload.get("theme_bundles"):
        return

    topics_dir = ddir / "topics"
    bundles = payload["theme_bundles"]
    # group bundles by sub_domain → type
    sub_bundles: dict[str, list] = defaultdict(list)
    for b in bundles:
        for sd in b.get("sub_domains", ["Uncategorized"]):
            sub_bundles[sd].append(b)

    # topics/index.md — list sub-domains
    tlines = [f"# {dname} — Bundle Cross-Views", "",
              "Each sub-domain page lists every bundle in that sub-domain, grouped by entity type.",
              "", "## Sub-domains", ""]
    for sd in sorted(sub_bundles):
        tlines.append(f"- [{sd}]({_slug(sd)}.md) — {len(sub_bundles[sd])} bundles")
    # topics/ is one level deeper than the domain index, so the appendix link needs ../.
    tlines += ["", f"- [All sections appendix](../{appendix_link})", ""]
    _write(topics_dir / "index.md",
           _fm("TopicIndex", f"{dname} topics", f"{len(bundles)} bundles") + "\n".join(tlines) + "\n")

    for sd, blist in sub_bundles.items():
        by_type: dict[str, list] = defaultdict(list)
        for b in blist:
            for tp in b.get("types", ["?"]):
                by_type[tp].append(b)
        plines = [f"# {dname} — {sd}", "",
                  f"**{len(blist)}** bundles in sub-domain `{sd}`.",
                  "", "## By Entity Type\n"]
        for tp in sorted(by_type):
            plines.append(f"### {tp} ({len(by_type[tp])})\n")
            for b in sorted(by_type[tp], key=lambda x: -x.get("member_count", 0)):
                anchor = b.get("anchor", b.get("kind"))
                if anchor == "project":
                    link = f"[**{b['title']}**](../../../projects/{b.get('name', b['title'])}.md)"
                else:
                    link = f"**{b['title']}**"
                owner = f" · {b.get('owner','')}" if b.get("owner") else ""
                hook = f" — {b.get('hook','')}" if b.get("hook") else ""
                span = f" [{b.get('time_span','')}]" if b.get("time_span") else ""
                plines.append(f"- {link}{owner}{span}{hook} · {b.get('member_count',0)} sections")
            plines.append("")
        plines.append(f"- [返回 {dname} hub](index.md)")
        _write(topics_dir / f"{_slug(sd)}.md",
               _fm("Topic", f"{dname} · {sd}", f"{len(blist)} bundles") + "\n".join(plines) + "\n")

    # big-bundle detail pages (>40 sections)
    for b in bundles:
        if b.get("member_count", 0) <= 40:
            continue
        slug = _slug(b["title"])
        anchor = b.get("anchor", b.get("kind"))
        if anchor == "project":
            head = f"[项目工作线](../../../projects/{b.get('name', b['title'])}.md) · {b.get('owner','')}"
        else:
            head = f"{b.get('owner','')} · {b.get('hook','')}"
        blines = [f"# {dname} — {b['title']}", "",
                  f"- **{len(b.get('sub_domains',[]))} sub-domains**: {', '.join(b.get('sub_domains',[]))}",
                  f"- **types**: {', '.join(b.get('types',[]))}",
                  f"- **anchor**: {anchor}",
                  f"- **sections**: {b.get('member_count',0)}  ·  **span**: {b.get('time_span','')}",
                  f"- **source**: {b.get('source','')} · {head}", ""]
        blines.append(f"- [返回 {dname} 交叉视图](index.md)")
        _write(topics_dir / f"{slug}.md",
               _fm("Bundle", b["title"], f"{b.get('member_count',0)} sections") + "\n".join(blines) + "\n")

    # 清理历史残留:bundle title 每次重跑会变,旧 detail 页会残留;删除非当前 hub_data 生成的 .md
    expected = {"index.md"}
    for sd in sub_bundles:
        expected.add(f"{_slug(sd)}.md")
    for b in bundles:
        if b.get("member_count", 0) > 40:
            expected.add(f"{_slug(b['title'])}.md")
    if topics_dir.exists():
        for f in topics_dir.glob("*.md"):
            if f.name not in expected:
                f.unlink()




def _slug(name: str) -> str:
    """Produce a filename-safe slug from an entity name (no spaces/parens).

    Artifact names with parentheses/spaces must be slugged so the
    generated page filename and its links survive markdown/HTML link parsing.
    """
    out = []
    for ch in name:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_").rstrip(". ")
    return slug or "_"


def _render_domain_page(dname: str, dlabel: str, ds_list: list, cu_list: list,
                        pr_list: list, source_by_id: dict, is_auto: bool,
                        sec_link_fn=None, source_depth: int = 4,
                        domain_def: str = "") -> None:
    """Render a domain hub page plus a complete sections appendix."""
    if is_auto:
        ddir = INDEX_DIR / "domains" / "others"
        path = ddir / f"{dname}.md"
        appendix_path = ddir / f"{dname}-sections-all.md"
        appendix_link = f"{dname}-sections-all.md"
    else:
        ddir = INDEX_DIR / "domains" / dname
        path = ddir / "index.md"
        appendix_path = ddir / "sections-all.md"
        appendix_link = "sections-all.md"

    seen_srcs: set[str] = set()
    ds_lines = []
    for ds in ds_list:
        src = source_by_id.get(ds.source_id)
        if src:
            pp = f" ({','.join(ds.people[:3])})" if ds.people else ""
            link = sec_link_fn(ds) if sec_link_fn else _source_link_or_text(src.path, source_depth)
            ds_lines.append(f"- {link} → **{ds.name}**{pp}")
            seen_srcs.add(src.id)

    cu_lines = [f"- [`{cu.name}` ({cu.kind})](../../code/{source_by_id[cu.source_id].path.replace('/', '_')}.md#{cu.id})"
                for cu in cu_list if cu.source_id in source_by_id]
    people_lines = [f"- {pr['name']} ({', '.join(pr.get('aliases', []))})" for pr in pr_list]
    src_lines = [f"- {_source_link_or_text(source_by_id[sid].path, source_depth)}" for sid in seen_srcs if sid in source_by_id]

    # Render structured definition (from domain_definitions.json) or flat text
    def_section = ""
    if isinstance(domain_def, dict):
        sections = []
        field_labels = {
            "concept": "Concept",
            "typical_activities": "Typical Activities",
            "not": "What it is NOT",
            "discriminating_signals": "Discriminating Signals",
        }
        for field, heading in field_labels.items():
            if domain_def.get(field):
                sections.append(f"### {heading}\n\n{domain_def[field]}\n")
        if sections:
            def_section = "\n## Definition\n\n" + "\n".join(sections)
    elif isinstance(domain_def, str) and domain_def:
        def_section = f"\n## Definition\n\n{domain_def}\n"

    top_sections = ds_lines[:12]
    top_code = cu_lines[:8]
    top_people = people_lines[:8]
    top_sources = src_lines[:8]

    hub_lines = [
        f"# {dname} — {dlabel}" if dlabel else f"# {dname}",
        "",
        "## Overview",
        f"[{len(ds_list)} doc sections](#document-sections) · [{len(cu_list)} code units](#code-units) · [{len(pr_list)} people](#people) · [full section appendix]({appendix_link})",
    ]
    if def_section:
        hub_lines.append(def_section.rstrip())
    # ── F1 hub sections: 30-sec panorama / model lineage / failure modes / entity browse ──
    f1 = _render_domain_f1(dname, dlabel)
    if f1:
        hub_lines.append(f1)
    # ── Theme bundles (Stage C, from hub_data.json) ──
    bundle_section = _render_domain_bundles(dname)
    if bundle_section:
        hub_lines.append(bundle_section)
    # ── topics/ cross-view pages (per sub-domain) + bundle detail pages ──
    _render_domain_topics(dname, ddir, appendix_link)
    hub_lines.extend([
        "",
        "## Document Sections",
        *(top_sections or ["_None_"]),
    ])
    if len(ds_lines) > len(top_sections):
        hub_lines.append(f"- ... and {len(ds_lines) - len(top_sections)} more in [{appendix_link}]({appendix_link})")
    hub_lines.extend([
        "",
        "## Code Units",
        *(top_code or ["_None_"]),
    ])
    if len(cu_lines) > len(top_code):
        hub_lines.append(f"- ... and {len(cu_lines) - len(top_code)} more in the code and entity indexes")
    hub_lines.extend([
        "",
        "## People",
        *(top_people or ["_None_"]),
    ])
    if len(people_lines) > len(top_people):
        hub_lines.append(f"- ... and {len(people_lines) - len(top_people)} more in [people index](../../indexes/by-person.md)")
    hub_lines.extend([
        "",
        "## Sources",
        *(top_sources or ["_None_"]),
    ])
    if len(src_lines) > len(top_sources):
        hub_lines.append(f"- ... and {len(src_lines) - len(top_sources)} more represented in [{appendix_link}]({appendix_link})")
    # ── 你在找什么 (per-domain decision tree, F1 principle 4) ──
    decision_lines = [
        "",
        "## What are you looking for?",
        f"- **Work lines** in this domain → [Theme Bundles](#theme-bundles) / [Projects](../../projects/index.md)",
        f"- **Deliverables** (models/factors/data/servers) → [Artifacts](../../artifacts/index.md) / [registries](../../registries/models.md)",
        f"- **Rules / terms / metrics** → [TERMS](../../TERMS.md) / [Protocols](../../infos/protocols/index.md)",
        f"- **Common failures & fixes** → [Errors](../../infos/errors/index.md)",
        f"- **All sections (exhaustive grep)** → [{appendix_link}]({appendix_link})",
    ]
    if (ddir / "topics").exists():
        decision_lines.append(f"- **Cross-view (sub-domain × entity type)** → [topics/](topics/index.md)")
    hub_lines.extend(decision_lines)
    hub_lines.extend([
        "",
        "## See also",
        "- [All domains](../../index.md#domains)",
        "- [All entity types](../../index.md#by-entity-type)",
        "- [By source type](../../indexes/by-source-type/index.md)",
    ])

    fm_title = f"{dname} — {dlabel}" if dlabel else dname
    _write(
        path,
        _fm("Domain", fm_title, f"{len(ds_list)} doc sections, {len(cu_list)} code units") + "\n".join(hub_lines) + "\n",
    )

    appendix_lines = [
        f"# {dname} — All Sections Appendix",
        "",
        f"Complete section listing for `{dname}`. Return to the [domain hub](index.md).",
        "",
        f"## All Document Sections ({len(ds_list)})",
        *(ds_lines or ["_None_"]),
        "",
        "## Source Pages",
        *(src_lines or ["_None_"]),
    ]
    _write(
        appendix_path,
        _fm("DomainAppendix", f"{fm_title} appendix", f"{len(ds_list)} doc sections") + "\n".join(appendix_lines) + "\n",
    )


def render_index(
    domains: list[DomainRecord],
    sources: list[SourceRecord],
    doc_sections: list[DocSectionRecord],
    code_units: list[CodeUnitRecord],
) -> None:
    # 发现型实体(dicts)从 knowledge_bits 加载,避免硬编码 resource/person
    people = _load_json("people.json")
    resources = _load_json("resources.json")
    source_by_id = {s.id: s for s in sources}
    predefined_names = {d.name for d in domains}
    total_ds = len(doc_sections)

    # index sections by domain
    domain_docs: dict[str, list] = defaultdict(list)
    domain_code: dict[str, list] = defaultdict(list)
    domain_people: dict[str, list] = defaultdict(list)
    dc = Counter()
    for ds in doc_sections:
        for d in (ds.domains or []):
            domain_docs[d].append(ds)
            dc[d] += 1
    for cu in code_units:
        src = source_by_id.get(cu.source_id)
        if src and ".ipynb_checkpoints" in src.path:
            continue  # checkpoint copies have no CodeModule page (byte-identical duplicates)
        for d in (cu.domains or []):
            domain_code[d].append(cu)
    for pr in people:
        for d in (pr.get("domains") or []):
            domain_people[d].append(pr)

    auto_names = sorted(set(dc.keys()) - predefined_names)
    kc = Counter(ds.kind for ds in doc_sections)

    # ── root index ──
    all_domain_names = sorted(predefined_names)

    # Load domain definitions for brief descriptions (single source: model_data)
    domain_defs: dict = model_data.domain_definitions()

    def _brief_def(domain_name: str) -> str:
        """Brief = first sentence of the structured concept (from model_data definitions)."""
        full = domain_defs.get(domain_name, "")
        if isinstance(full, dict):
            full = full.get("concept") or full.get("label") or ""
        if not full:
            return ""
        first = full.split(". ")[0].strip()
        if len(first) > 200:
            first = first[:197] + "..."
        return first

    # Domain section with brief definitions
    domain_entries = []
    for d in all_domain_names:
        brief = _brief_def(d)
        count = dc.get(d, 0)
        entry = f"- **[{d}](domains/{d}/index.md)** ({count} sections)"
        if brief:
            entry += f" — {brief}"
        domain_entries.append(entry)
    domain_section = "\n".join(domain_entries)

    # Source type summary
    source_type_counts = Counter(s.source_type for s in sources)
    SD_LABELS = _source_type_labels()
    source_type_lines = []
    for st, count in source_type_counts.most_common():
        label = SD_LABELS.get(st, st)
        source_type_lines.append(f"- **{st}** ({count} sources) — {label}")
    source_type_section = "\n".join(source_type_lines)

    kind_stats = "\n".join(f"  {k}: {c}" for k, c in kc.most_common())

    # what_looking_for 实体行(由 entities.json 的 nav_hint + render 驱动)
    from util.ontology import all_entities as _all_entities
    _what_lines = []
    for _n, _c in _all_entities().items():
        _r = getattr(_c, "_render", {})
        _hint = _r.get("nav_hint")
        if _hint:
            _what_lines.append(f"- **{_hint}** → [{_r.get('kind_label', _n)}s]({_r.get('page_dir', '')}/index.md)")
    what_entity_lines = "\n".join(_what_lines)

    _write(INDEX_DIR / "index.md", _fm("KnowledgeIndex", t('index_title'),
        f"{len(sources)} sources, {total_ds} sections, {len(code_units)} code units, "
        f"{len(resources)} resources, {len(people)} people") + f"""# {t('index_title')}

## {t('overview')}

This knowledge base indexes **{len(sources)}** source materials,
covering chat logs, reports, code repositories, technical documentation, and configuration files.
Content is segmented into **{total_ds}** individual sections, each classified into one or more of the
**{len(all_domain_names)}** domains below. Use this index to discover what the organization knows about any
business area, find relevant source materials, and trace relationships between code, documentation,
and people.

**{t('query_guidance')}**: To answer a question about a specific business area, start by
navigating to the relevant domain page (e.g., `domains/<domain>/index.md`). Each domain page
is a bounded hub linking to a full `sections-all.md` appendix. Entity indexes provide complementary
lookup paths — find all sections mentioning a person, all resources of a given type, or all sources
of a particular format.

## {t('what_looking_for')}

- **A term / acronym / codename you don't recognize** → [TERMS.md](TERMS.md) — global glossary, grep first stop
{what_entity_lines}
- **The code that does X** → [Code map](code/index.md)

## {t('statistics')}

- **{len(sources)}** {t('sources')} · **{total_ds}** {t('doc_sections')} · **{len(code_units)}** {t('code_units')}
- **{len(resources)}** {t('resources')} (storage paths, file references, URLs) · **{len(people)}** {t('people')}

### {t('section_types')}
{kind_stats}

## {t('domains')}

Each domain represents a distinct business activity area. Sections are classified into one or more
domains via LLM-based semantic classification. The count reflects the number of sections where
this domain applies (multi-label: a section may belong to multiple domains).

{domain_section}

## {t('entity_indexes')}

Entity indexes provide alternate navigation paths through the knowledge base, organized by entity
type rather than by domain.

### {t('by_entity_type')}
- **[person](indexes/by-person.md)** ({len(people)} items) — Team members with domain, section, and task links
- **[project](projects/index.md)** ({_count_json('projects.json')}) — Long-running work items (curated from source/project/)
- **[artifact](artifacts/index.md)** ({_count_json('artifacts.json')}) — Named deliverables (model/factor_set/dataset/server/script) by kind
- **[task](infos/tasks/index.md)** ({_count_json('phase2_tasks.json')}) — Week-bounded work threads
- **[error](infos/errors/index.md)** ({_count_json('phase2_errors.json')}) — Error→diagnosis→fix records
- **[workflow](infos/workflows/index.md)** ({_count_json('phase2_workflows.json')}) — Procedural step sequences
- **[protocol](infos/protocols/index.md)** ({_count_json('phase2_protocols.json')}) — Named conventions and rules
- **[doc_section](sections/)** ({total_ds} items) — All text sections by kind; full listing per domain in `sections-all.md`
- **[code_unit](code/code-units.md)** ({len(code_units)} items) — Functions/classes/modules, cross-referenced to the doc sections that mention them (code↔docs view)
- **[resource](indexes/by-resource.md)** ({len(resources)} items) — HDFS paths, file references, URLs (appendix registry)

### {t('registries')}
- **[models](registries/models.md)** · **[factor-sets](registries/factor-sets.md)** · **[servers](registries/servers.md)** · **[datasets](registries/datasets.md)** — single-file grep-first tables
- **[TERMS.md](TERMS.md)** — global glossary (alias → canonical, grep first stop)

### {t('by_source_type')}
- **[chat](indexes/by-source-type/chat.md)** — Sections from chat logs (1-on-1 and group conversations)
- **[weekly_report](indexes/by-source-type/weekly_report.md)** — Sections from weekly progress reports (DOCX and PDF)
- **[doc](indexes/by-source-type/doc.md)** — Sections from technical documentation (markdown specs, design docs)
- **[code_file](indexes/by-source-type/code_file.md)** — Sections extracted from Python source files
- **[code_readme](indexes/by-source-type/code_readme.md)** — Sections from code README documentation files
- **[config](indexes/by-source-type/config.md)** — Sections from configuration files (CSV registries, config tables)
- **[project](indexes/by-source-type/project.md)** — Project task specifications (source/project)

## Code Structure

Browsable code structure pages generated from AST analysis — imports, functions with docstrings, path literals, and doc cross-references.

- **[Code map](code/index.md)** — functional map of the ingested Python code (naming keys, code domains, dedup)

## Source type summary

{source_type_section}
""")

    # ── doc section pages (self-contained, depth 3) — 仅当语料含文档分段时渲染 ──
    sec_dir = INDEX_DIR / "sections"
    if doc_sections:
        sec_dir.mkdir(parents=True, exist_ok=True)
    for ds in doc_sections:
        src = source_by_id.get(ds.source_id)
        # Source code files (source/code/…) aren't bundled in the artifact — show the
        # path as plain text instead of a dead link.
        src_link = _source_link_or_text(src.path, 3) if src else "?"
        _write(sec_dir / f"{ds.id}.md",
            _fm("DocSection", ds.name, f"kind: {ds.kind}, source: {portable_code_path(src.path) if src else '?'}") +
            f"""# {ds.name[:120]}

- **Kind**: {ds.kind}
- **Source**: {src_link}
- **Domains**: {', '.join(ds.domains) if ds.domains else 'others'}
- **People**: {', '.join(ds.people) if ds.people else '_none_'}
- **Messages**: {ds.message_count}  |  **Time**: {ds.time_range}

```text
{ds.content}
```
""")

    # ── all domain pages (predefined + others) ──
    # Domain pages are at depth 2 (domains/{name}/index.md), sections at depth 1.
    def _sec_link(ds):
        return f"[{ds.name[:80]}](../../sections/{ds.id}.md)"

    # Load domain definitions (structured format from model_data)
    domain_defs_raw: dict = model_data.domain_definitions()

    # Build domain name → display label map
    domain_labels = {d.name: d.label for d in domains}

    for dname in sorted(predefined_names):
        _render_domain_page(dname, domain_labels.get(dname, dname),
                            domain_docs.get(dname, []), domain_code.get(dname, []),
                            domain_people.get(dname, []), source_by_id, False, _sec_link,
                            domain_def=domain_defs_raw.get(dname, ""))
    _render_domain_page("others", "Uncategorized",
                        domain_docs.get("others", []), domain_code.get("others", []),
                        domain_people.get("others", []), source_by_id, False, _sec_link)

    # ── per-type appendix registries (relocated from the removed by-entity-type/) ──
    # Each entity type now lives at its canonical page (projects/, artifacts/,
    # infos/, code/, indexes/by-person.md). Only the raw registries that have no
    # other home are kept: code_unit → code/code-units.md, resource →
    # indexes/by-resource.md. person → indexes/by-person.md (Person hub).
    res_section_map: dict[str, list] = defaultdict(list)
    for ds in doc_sections:
        for rid in (ds.resources or []):
            res_section_map[rid].append(ds)

    # resource appendix registry (was by-entity-type/resource.md)
    res_by_kind: dict[str, list] = defaultdict(list)
    for r in resources:
        res_by_kind[r["kind"]].append(r)
    res_lines = []
    for kind in sorted(res_by_kind):
        res_lines.append(f"## {kind} ({len(res_by_kind[kind])})")
        for r in res_by_kind[kind]:
            res_lines.append(f"- `{r['name']}`")
            linked = res_section_map.get(r["id"], [])
            for ds in linked:
                res_lines.append(f"    - [{ds.name[:80]}](../../sections/{ds.id}.md)")
    _write(INDEX_DIR / "indexes" / "by-resource.md",
        _fm("ResourceRegistry", "resource", f"{len(resources)} resources (appendix)") +
        f"# resource ({len(resources)})\n\n" + "\n".join(res_lines))

    # code-units.md = the code↔docs cross-reference, grouped by code domain
    # (mirroring the Code Map) → code module → code units. Each code unit links to
    # its CodeModule page's function section and lists the doc sections that mention
    # it. Checkpoint (.ipynb_checkpoints) units are excluded (byte-identical
    # duplicates already covered by the dedup table in code/index.md).
    from collections import defaultdict as _defaultdict
    # TODO(搬移): code_map 未在本次搬移范围内,目标位置待定(疑似 exploration/segmentation.code)
    from segmentation.code.code_map import _classify_code_domain
    sections_by_id = {ds.id: ds for ds in doc_sections}
    canonical_units = [cu for cu in code_units
                       if source_by_id.get(cu.source_id)
                       and ".ipynb_checkpoints" not in source_by_id[cu.source_id].path]
    cu_by_source: dict[str, list] = _defaultdict(list)
    for cu in canonical_units:
        cu_by_source[cu.source_id].append(cu)

    # code_domain → source_id → units (same code_domain labels as the Code Map)
    code_domain_modules: dict[str, dict[str, list]] = _defaultdict(lambda: _defaultdict(list))
    for sid, culist in cu_by_source.items():
        src = source_by_id.get(sid)
        if not src:
            continue
        funcs = [cu.name for cu in culist if cu.kind == "function"]
        doc = " ".join(cu.doc for cu in culist if cu.doc)
        code_domain = _classify_code_domain(src.path, doc, funcs)
        code_domain_modules[code_domain][sid] = culist

    cu_lines = [
        f"# Code Units ↔ Doc Sections ({len(canonical_units)})",
        "",
        f"**{len(canonical_units)}** canonical code units, grouped by code domain "
        f"(matching the [Code map](index.md)), then by code module. Each code unit links "
        f"to its CodeModule page's function section and lists the doc sections that "
        f"mention it. Checkpoint duplicates are excluded.",
        "",
    ]
    for code_domain in sorted(code_domain_modules.keys()):
        modules = code_domain_modules[code_domain]
        cu_lines.append(f"\n## {code_domain} ({len(modules)} modules)\n")
        for sid in sorted(modules, key=lambda s: Path(source_by_id[s].path).name):
            src = source_by_id[sid]
            page_name = src.path.replace("/", "_") + ".md"
            culist = modules[sid]
            cu_lines.append(f"\n### [{Path(src.path).name}]({page_name})\n")
            for cu in sorted(culist, key=lambda c: c.name):
                anchor = cu.id
                cu_lines.append(f"- **[{cu.name} ({cu.kind})]({page_name}#{anchor})**")
                # doc sections that mention this code unit (dedup + cap)
                seen = set()
                for dsid_raw in (cu.doc_mentions or []):
                    dsid = dsid_raw.split(":")[0]
                    if dsid in seen:
                        continue
                    seen.add(dsid)
                    sec = sections_by_id.get(dsid)
                    if sec:
                        cu_lines.append(f"    - [{sec.name[:80]}](../sections/{dsid}.md)")
                    if len(seen) >= 8:
                        break
    cu_lines.append("\n## See also\n- [Code map](index.md)\n- [Knowledge index](../index.md)")
    _write(INDEX_DIR / "code" / "code-units.md",
        _fm("CodeUnitRegistry", "code_unit", f"{len(canonical_units)} code units ↔ doc sections") +
        "\n".join(cu_lines) + "\n")

    # person page — grouped by confidence bucket with per-person headings
    person_section_map: dict[str, list] = defaultdict(list)
    for ds in doc_sections:
        for pname in (ds.people or []):
            person_section_map[pname].append(ds)

    # 反向引用:凡 ref=PersonRecord 的字段,建立 person → 实体 映射(由 entities.json 声明驱动)
    from util.ontology import all_entities as _all_entities
    _schema = model_data.load("ontology/entities")
    person_entity_map: dict[str, list] = defaultdict(list)  # person -> [(label, page_dir, slug_field, title_field, rec)]
    for _name, _cls in _all_entities().items():
        _r = getattr(_cls, "_render", {})
        if not _r:
            continue
        _person_fields = [f["name"] for f in _schema.get(_name, {}).get("fields", [])
                          if f.get("ref") == "PersonRecord"]
        if not _person_fields:
            continue
        _label = _r.get("kind_label", _name)
        _page_dir = _r.get("page_dir", "")
        _slug_field = _r.get("slug_field", "id")
        _title_field = _r.get("title_field", _slug_field)
        for _rec in _load_entity_records(_cls):
            for _pf in _person_fields:
                _vals = _rec.get(_pf)
                if isinstance(_vals, str):
                    _vals = [_vals]
                for _p in (_vals or []):
                    if _p and _p != "unknown":
                        person_entity_map[_p].append((_label, _page_dir, _slug_field, _title_field, _rec))

    pr_lines = [f"# person ({len(people)})\n"]

    # Quick-nav list at top
    for bucket, label in [("HIGH", "HIGH"), ("MEDIUM", "MEDIUM"), ("LOW", "LOW")]:
        bucket_people = [p for p in people if p.get('confidence', 'LOW') == bucket]
        if bucket_people:
            names = " · ".join(f"[{p['name']}](#{p['name'].replace(' ', '-')})" for p in bucket_people)
            pr_lines.append(f"**{label}**: {names}\n")

    for bucket, label in [("HIGH", "HIGH Confidence"), ("MEDIUM", "MEDIUM Confidence"), ("LOW", "LOW Confidence")]:
        bucket_people = [p for p in people if p.get('confidence', 'LOW') == bucket]
        if not bucket_people:
            continue
        pr_lines.append(f"## {label} ({len(bucket_people)})\n")

        for pr in bucket_people:
            # Person heading with explicit anchor for reliable linking
            alias_str = f" ({', '.join(pr.get('aliases', []))})" if pr.get('aliases') else ""
            anchor = pr["name"].replace(" ", "-")
            pr_lines.append(f'<a id="{anchor}"></a>\n\n### {pr["name"]}{alias_str}\n')
            stats = []
            if pr.get('count_chat', 0) > 0:
                stats.append(f"💬{pr.get('count_chat', 0)}")
            if pr.get('count_files', 0) > 0:
                stats.append(f"📄{pr.get('count_files', 0)}")
            if pr.get('count_tasks', 0) > 0:
                stats.append(f"✅{pr.get('count_tasks', 0)}")
            if stats:
                pr_lines.append(f"{' '.join(stats)}\n")

            # Section links (compact, all) — page lives at indexes/by-person.md (depth 3)
            linked = person_section_map.get(pr["name"], [])
            if linked:
                pr_lines.append(f"**Sections** ({len(linked)}): " + " · ".join(
                    f"[{ds.name}](../sections/{ds.id}.md)" for ds in linked
                ) + "\n")

            # Entity links (compact, all;由 ref=PersonRecord 声明驱动)
            _by_label = defaultdict(list)
            for _label, _page_dir, _slug_field, _title_field, _rec in person_entity_map.get(pr["name"], []):
                _by_label[_label].append((_page_dir, _slug_field, _title_field, _rec))
            for _label, _items in sorted(_by_label.items()):
                _links = " · ".join(
                    f"[{(_rec.get(_title_field) or _rec.get(_slug_field, ''))[:80]}](../{_page_dir}/{_slug(str(_rec.get(_slug_field, '')))}.md)"
                    for _page_dir, _slug_field, _title_field, _rec in _items
                )
                pr_lines.append(f"**{_label}s** ({len(_items)}): {_links}\n")

    # Person hub (was by-entity-type/person.md; now at indexes/by-person.md)
    nav = ("## See also\n\n"
           f"- [All domains](../index.md#domains)\n"
           f"- [All entity types](../index.md#by-entity-type)\n"
           f"- [By source type](by-source-type/index.md)")

    _write(INDEX_DIR / "indexes" / "by-person.md",
        _fm("PersonIndex", "People", f"{len(people)} people") +
        "\n".join(pr_lines) + nav)

    # ── source type index — reading guides + links to sections, not just sources ──
    SD_LABELS = _source_type_labels()
    # Reading guide per source type: how an agent should read this class of source.
    READING_GUIDES = _source_type_reading_guides()
    sdoms = defaultdict(list)
    for s in sources: sdoms[s.source_type].append(s)
    for sd, srcs in sdoms.items():
        desc = SD_LABELS.get(sd, "Other materials")
        sd_docs = [ds for ds in doc_sections if ds.source_id in {s.id for s in srcs}]
        lines = [f"*{desc} — {len(srcs)} sources, {len(sd_docs)} sections*\n"]
        guide = READING_GUIDES.get(sd)
        if guide:
            lines.append("## How to Read This Source\n")
            for label, body in guide:
                lines.append(f"- **{label}**：{body}")
            lines.append("")
        if sd_docs:
            lines.append(f"## Sections ({len(sd_docs)})")
            for ds in sd_docs:
                pp = f" ({','.join(ds.people[:3])})" if ds.people else ""
                lines.append(f"- [{ds.name[:80]}](../../sections/{ds.id}.md){pp}")
        lines.append(f"\n## Sources ({len(srcs)})")
        for s in srcs:
            lines.append(f"- {_source_link_or_text(s.path, 3)}")
        _write(INDEX_DIR / "indexes" / "by-source-type" / f"{sd}.md",
            _fm("SourceTypeIndex", sd, f"{desc} — {len(srcs)} sources") +
            f"# {sd}\n\n" + "\n".join(lines))

    # by-source-type index page
    st_lines = []
    for sd, srcs in sdoms.items():
        desc = SD_LABELS.get(sd, "Other materials")
        st_lines.append(f"- [{sd}]({sd}.md) ({len(srcs)} sources) — {desc}")
    st_nav = ("\n## See also\n\n"
              f"- [All domains](../../index.md#domains)\n"
              f"- [All entity types](../../index.md#by-entity-type)")

    _write(INDEX_DIR / "indexes" / "by-source-type" / "index.md",
        _fm("SourceTypeIndex", "Source Type Index") +
        "# Source Type Index\n\n" + "\n".join(st_lines) + st_nav)

    # ── Clean stale section pages ──
    expected_ids = {ds.id for ds in doc_sections}
    stale = 0
    for page in sec_dir.glob("*.md"):
        if page.stem not in expected_ids:
            page.unlink()
            stale += 1
    if stale:
        print(f"  cleaned {stale} stale section pages", flush=True)

    # ── 通用实体页渲染(由 entities.json 的 render 声明驱动) ──
    from util.ontology import all_entities
    entity_records = {}
    for _name, _cls in all_entities().items():
        _lk = getattr(_cls, "_list_key", "")
        if _lk:
            entity_records[_lk] = _load_entity_records(_cls)
    render_ctx = {
        "sections_by_id": sections_by_id,
        "entities": entity_records,
        "code_units_by_id": {cu.id: cu for cu in code_units},
        "source_by_id": source_by_id,
    }
    for _name, _cls in all_entities().items():
        if not getattr(_cls, "_render", {}):
            continue
        _lk = getattr(_cls, "_list_key", "")
        _render_entity_pages(_name, _cls, entity_records.get(_lk, []), render_ctx)
    _render_artifact_registries(entity_records.get("artifacts", []))
    _link_phase2_pages(sources, doc_sections)

    # ── Self-indexing: TERMS.md + Search Tags + backlinks + frontmatter aliases ──
    from render.selfindex import render_terms, render_selfindex
    render_terms()
    render_selfindex()

    # ── AGENTS.md schema file ──
    _write_agents_md(
        sources, doc_sections, code_units, resources, people,
        domain_defs, all_domain_names, dc, domain_docs, domain_code,
        kc, total_ds,
    )

    # ── Code structure index(仅当语料含代码时渲染;纯文档语料跳过) ──
    if code_units:
        from render.code_index import render_code_index
        render_code_index()
        # The code map renders the index page from code_map.json, replacing the
        # directory-mirror grouping. Leaf pages stay from render_code_index.
        from segmentation.code.code_map import render_code_map
        render_code_map()


def _write_agents_md(
    sources: list,
    doc_sections: list,
    code_units: list,
    resources: list,
    people: list,
    domain_defs: dict,
    all_domain_names: list,
    dc: Counter,
    domain_docs: dict,
    domain_code: dict,
    kc: Counter,
    total_ds: int,
) -> None:
    """Generate AGENTS.md — 泛化模板 + 具体数据(entities.json 的 schema + render)动态组装。

    实体清单、字段表、目录树、导航列表都由 all_entities() 的 render/doc/fields 驱动;
    仅保留与实体无关的通用约定(领域分类、来源类型、链接约定、阅读约定)作为静态文案。
    """
    from util.ontology import all_entities
    from util import model_data as _md

    # ── 收集实体信息(单一事实源:entities.json 的 schema + render) ──
    schema = _md.load("ontology/entities")
    ents = []
    for name, cls in all_entities().items():
        rcfg = getattr(cls, "_render", {})
        if not rcfg:
            continue
        e_schema = schema.get(name, {})
        ents.append({
            "label": rcfg.get("kind_label", name),
            "page_dir": rcfg.get("page_dir", ""),
            "count": len(_load_entity_records(cls)),
            "doc": (e_schema.get("doc") or "").strip(),
            "fields": e_schema.get("fields", []),
        })

    # ── 实体表 / 字段表 / 目录树实体行 / 导航列表 ──
    entity_rows = [f"| [{e['label']}]({e['page_dir']}/index.md) | {e['count']} | {e['doc']} |" for e in ents]
    entity_table = "\n".join(entity_rows) if entity_rows else "_none_"

    field_sections = []
    for e in ents:
        rows = "\n".join(f"| {f['name']} | {f.get('type', 'str')} | {f.get('doc', '')} |" for f in e["fields"])
        field_sections.append(f"### {e['label']} Fields\n\n| Field | Type | Description |\n|---|---|---|\n{rows}\n")
    field_tables = "\n".join(field_sections)

    dir_entity_lines = "\n".join(f"├── {e['page_dir']}/ ← {e['doc']}" for e in ents)
    nav_entity_lines = "\n".join(f"- [{e['label']}s]({e['page_dir']}/index.md) — {e['doc']}" for e in ents)

    # ── Source type summary ──
    source_types = Counter(s.source_type for s in sources)
    st_lines = "\n".join(f"| {st} | {c} |" for st, c in source_types.most_common())

    # ── Domain table ──
    domain_rows = []
    for dname in sorted(all_domain_names):
        dd = domain_defs.get(dname, {})
        label = (dd.get("label", dname) if isinstance(dd, dict) else dname)
        count = dc.get(dname, 0)
        concept = ""
        if isinstance(dd, dict):
            concept = dd.get("concept", "")[:120]
        domain_rows.append(f"| [{dname}](domains/{dname}/index.md) | {label} | {count} | {concept} |")
    domain_table = "\n".join(domain_rows) if domain_rows else "_none_"

    content = _fm("AgentGuide", "Knowledge Base Schema",
                  "How this knowledge base is structured, its conventions, and navigation recipes for AI agents.") + f"""# AGENTS.md — Knowledge Base Schema

> Auto-generated schema file. Tells AI agents how this knowledge base is structured,
> what conventions to follow, and how to navigate and query it effectively.
> Co-evolve this file as the knowledge base grows.

## Directory Map

```
artifact/index/
├── index.md              ← Main entry point (you are here)
├── AGENTS.md             ← This schema file
├── TERMS.md              ← Global glossary (unknown term? grep this first)
├── domains/              ← One page per business domain
│   ├── {{domain}}/index.md   ← hub
│   ├── {{domain}}/sections-all.md  ← full section appendix
│   └── {{domain}}/topics/    ← sub-domain × entity-type cross-views
├── indexes/              ← by-person / by-resource / by-source-type / by-asset-class
{dir_entity_lines}
├── sections/             ← Individual document section pages (Phase 1)
│   └── {{section-id}}.md
├── code/                 ← Code pages (structure map + code↔docs cross-reference)
└── registries/           ← Single-file grep-first registries per artifact kind
```

## Domain Taxonomy

The knowledge base uses a {len(all_domain_names)}-domain taxonomy. Every document section, code unit,
and entity is classified into one or more of these domains.
**Always use these exact domain labels when querying or classifying.**

| Domain | Label | Sections | Description |
|---|---|---|---|
{domain_table}

**Classification rule**: Assign the domain that best matches the PRIMARY concern
of the content. Multi-label when the content genuinely spans multiple concerns.

## Entity Types

Structured entities extracted/determined from Phase 1 document sections.
Each entity has its own detail page, index, and cross-references.

| Entity | Count | Description |
|---|---|---|
{entity_table}

## Entity Fields

{field_tables}

## Source Types

| Type | Count |
|---|---|
{st_lines}

## Reading Conventions

When reading section pages to answer questions:

- **Read the full content** — sections may have supplemental rules, amendments,
  or clarifications appended at the end. Do not stop at the first matching rule.
- **Check adjacent sections** — related context may span multiple sections from
  the same source.
- **Prefer entity pages for structured answers** — entity info pages contain
  consolidated narratives extracted from multiple sections.
- **When the "why" is missing, say so** — many source materials document what
  was done but not why.

## Navigation Quick Reference

{nav_entity_lines}

- **By person** → `indexes/by-person.md` (sections + entities per person)
- **By domain** → `domains/{{domain}}/index.md` (hub)
- **By source type** → `indexes/by-source-type/index.md`

## Conventions

### ISO Weeks
All temporal references use ISO week format: `YYYY-Www` (e.g., `2025-W42`).

### Person Names
Chinese names (2-4 characters), normalized across aliases in the People index;
the `unknown` person label indicates sections where attribution failed.

### Status Values
- `completed` — work finished this week
- `planned` — intended for next week
- `in_progress` — ongoing work
- `blocked` — halted by an error or dependency
- `abandoned` — work discontinued

### Confidence Scores
Every extracted entity has a `_confidence` field (0.0–1.0) indicating extraction quality.

### Domain Label Rules
- Use exact labels from the taxonomy — no variants, no inventions
- Multi-label when content genuinely spans concerns
- Prefer the domain of the PRIMARY activity when ambiguous
"""

    _write(INDEX_DIR / "AGENTS.md", content)
    print(f"  AGENTS.md → {INDEX_DIR / 'AGENTS.md'}")

    html_dir = INDEX_DIR.parent / "index_html"
    html_dir.mkdir(parents=True, exist_ok=True)
    _write(html_dir / "AGENTS.md", content)
    print(f"  AGENTS.md → {html_dir / 'AGENTS.md'}")

    import re as _re
    plain = _re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
    _write(html_dir / "AGENTS.txt", plain)


def _load_entity_records(cls) -> list:
    """Load an entity's records from knowledge_bits (tries phase2_{lk}.json then {lk}.json)."""
    import json as _json
    lk = getattr(cls, "_list_key", "")
    for cand in (f"phase2_{lk}.json", f"{lk}.json"):
        if lk and (KB_DIR / cand).exists():
            return _json.loads((KB_DIR / cand).read_text())
    return []


def _ref_links(cls_name: str, field_name: str):
    """从字段的 ref 声明推导 links 的 (entity, page_dir, slug_field, title_field)。

    跨实体链接的 entity/target/label_field 不再手写,由 ref(目标实体类型)+ 目标实体的 render 推导。
    """
    schema = model_data.load("ontology/entities")
    fref = None
    for f in schema.get(cls_name, {}).get("fields", []):
        if f["name"] == field_name:
            fref = f.get("ref")
            break
    if not fref:
        return None
    tcls = get_entity(fref)
    tr = getattr(tcls, "_render", {})
    return (getattr(tcls, "_list_key", ""), tr.get("page_dir", ""),
            tr.get("slug_field", "id"), tr.get("title_field") or tr.get("slug_field", "id"))


def _render_entity_pages(entity_name: str, cls, records: list, context: dict) -> None:
    """Generic OKF page renderer driven by cls._render (见 entity_schema.json.render_schema).

    生成详情页(按 detail 分节)+ 索引页(按 index),并做 stale 清理。跨实体 links 由
    context["entities"]({list_key: [dict]})解析;相对链接深度由 page_dir + subdir_field 推导。
    """
    rcfg = getattr(cls, "_render", {})
    if not rcfg or not records:
        return
    page_dir = rcfg.get("page_dir")
    if not page_dir:
        return

    kind_label = rcfg.get("kind_label", entity_name)
    slug_field = rcfg.get("slug_field", "id")
    title_field = rcfg.get("title_field", slug_field)
    subdir_field = rcfg.get("subdir_field")
    sections_by_id = context["sections_by_id"]
    linked = {}
    for lk, recs in (context.get("entities") or {}).items():
        linked[lk] = {r.get("id") or r.get("name"): r for r in recs}

    adir = INDEX_DIR / page_dir
    depth = len(Path(page_dir).parts) + (1 if subdir_field else 0)
    prefix = "../" * depth

    # stale cleanup
    expected = {(r.get(subdir_field) if subdir_field else None, _slug(str(r.get(slug_field, "")))) for r in records}
    if adir.exists():
        for sub in adir.iterdir():
            if sub.is_dir():
                for page in sub.glob("*.md"):
                    if page.name != "index.md" and (sub.name, page.stem) not in expected:
                        page.unlink()
            elif sub.name.endswith(".md") and sub.name != "index.md" and (None, sub.stem) not in expected:
                sub.unlink()

    # detail pages
    for r in records:
        slug = _slug(str(r.get(slug_field, "")))
        subdir = r.get(subdir_field) if subdir_field else None
        odir = adir / subdir if subdir else adir
        odir.mkdir(parents=True, exist_ok=True)

        h1 = r.get(title_field) or r.get(slug_field, "")
        parts = [f"# {h1}", ""]

        for sec in rcfg.get("detail", []):
            stype = sec.get("type")
            heading = sec.get("heading")
            if stype == "bullets":
                for it in sec.get("items", []):
                    v = r.get(it["field"])
                    if v in (None, "", []):
                        continue
                    if isinstance(v, list):
                        v = ", ".join(map(str, v))
                    if it.get("code"):
                        v = f"`{v}`"
                    parts.append(f"- **{it.get('label', it['field'])}**: {v}")
            elif stype == "text":
                v = r.get(sec.get("field")) or r.get(sec.get("fallback_field", ""), "")
                if v:
                    parts.append(f"\n## {heading}\n\n{v}\n")
            elif stype == "list":
                v = r.get(sec.get("field")) or []
                if v:
                    body = "\n".join(f"- `{x}`" if sec.get("code") else f"- {x}" for x in v)
                    parts.append(f"\n## {heading}\n\n{body}\n")
            elif stype == "steps":
                steps = r.get(sec.get("field")) or []
                if steps:
                    parts.append(f"\n## {heading} ({len(steps)})\n\n")
                    for st in steps:
                        line = f"{st.get('order','?')}. **{st.get('action','')}**"
                        if st.get("function"):
                            line += f" — `{st['function']}`"
                        parts.append(line + "  \n")
                        if st.get("validation"):
                            parts.append(f"   - Validation: {st['validation']}\n")
            elif stype == "links":
                ids = r.get(sec.get("field")) or []
                out = []
                if sec.get("entity"):
                    lrecs = linked.get(sec.get("entity"), {})
                    for rid in ids:
                        rec = lrecs.get(rid)
                        if not rec:
                            continue
                        target = sec["target"].format(id=rid, name=rec.get("name", rid))
                        label = rec.get(sec.get("label_field", "name"), rid)
                        out.append(f"- [{str(label)[:80]}]({prefix}{target})")
                else:
                    info = _ref_links(entity_name, sec.get("field"))
                    if info:
                        entity, page_dir, slug_field, title_field = info
                        lrecs = linked.get(entity, {})
                        for rid in ids:
                            rec = lrecs.get(rid)
                            if not rec:
                                continue
                            slug = _slug(str(rec.get(slug_field, rid)))
                            label = rec.get(title_field) or rec.get(slug_field, rid)
                            out.append(f"- [{str(label)[:80]}]({prefix}{page_dir}/{slug}.md)")
                if out:
                    parts.append(f"\n## {heading}\n\n" + "\n".join(out) + "\n")

        # extra artifact-specific sections (related_code / related_resources / server_access)
        for extra in rcfg.get("extra", []):
            if extra == "related_code" and r.get("related_code"):
                code_links = []
                for cuid in r.get("related_code", [])[:10]:
                    cu = context.get("code_units_by_id", {}).get(cuid)
                    if not cu:
                        continue
                    csrc = context.get("source_by_id", {}).get(cu.source_id)
                    if csrc:
                        page = csrc.path.replace("/", "_") + ".md"
                        if (INDEX_DIR / "code" / page).exists():
                            code_links.append(f"- [`{cu.name}` ({cu.kind})]({prefix}code/{page})")
                        else:
                            code_links.append(f"- `{cu.name}`")
                    else:
                        code_links.append(f"- `{cu.name}`")
                if code_links:
                    parts.append("\n## Related code\n" + "\n".join(code_links) + "\n")
            elif extra == "related_resources" and r.get("related_resources"):
                res_links = [f"- `{rid}`" for rid in r.get("related_resources", [])[:10]]
                if res_links:
                    parts.append("\n## Related resources\n" + "\n".join(res_links) + "\n")
            elif extra == "server_access" and r.get("kind") == "server":
                access = _server_access_section(r, sections_by_id)
                if access:
                    parts.append(access)

        # source sections (uniform)
        src = []
        for sid in r.get("source_sections", []):
            sec = sections_by_id.get(sid)
            if sec:
                src.append(f"- [{sec.name[:80]}]({prefix}sections/{sid}.md)")
        if src:
            parts.append("\n## Source sections\n" + "\n".join(src) + "\n")

        # See also
        parts.append(f"\n## See also\n- [All {kind_label.lower()}s](index.md)\n")
        if subdir_field:
            parts.append(f"- [All {kind_label.lower()}s](../index.md)\n")

        desc = r.get("description") or r.get("summary") or r.get("progress", "") or ""
        tags = r.get("domains", [])
        _write(odir / f"{slug}.md", _fm(kind_label, h1, desc, tags) + "\n".join(parts) + "\n")

    # index page(s)
    if subdir_field:
        by_sub = defaultdict(list)
        for r in records:
            by_sub[r.get(subdir_field)].append(r)
        for sub in sorted(by_sub):
            lines = [f"# {kind_label} kind: {sub}", "", f"**{len(by_sub[sub])}** {sub}.", ""]
            for r in sorted(by_sub[sub], key=lambda x: str(x.get(slug_field))):
                lines.append(f"- **[{r.get(slug_field)}]({_slug(str(r.get(slug_field)))}.md)**")
            _write(adir / sub / "index.md",
                   _fm(f"{kind_label}Kind", sub, f"{len(by_sub[sub])} {sub}") + "\n".join(lines) + "\n")
        top = [f"# {kind_label}s", "", f"**{len(records)}** {kind_label.lower()}s.", "", "| Kind | Count |", "|---|---|"]
        for sub in sorted(by_sub):
            top.append(f"| [{sub}]({sub}/index.md) | {len(by_sub[sub])} |")
        _write(adir / "index.md",
               _fm(f"{kind_label}Index", f"{kind_label}s", f"{len(records)} {kind_label.lower()}s") + "\n".join(top) + "\n")
    else:
        idx = rcfg.get("index", {})
        gb = idx.get("group_by")
        lines = [f"# {kind_label} ({len(records)})\n"]
        if gb:
            grouped = defaultdict(list)
            for r in records:
                v = r.get(gb)
                if isinstance(v, list):
                    for x in v:
                        grouped[x].append(r)
                elif v:
                    grouped[v].append(r)
                else:
                    grouped["unknown"].append(r)
            keys = list(grouped.keys())
            if gb == "week":
                keys = sorted(keys, reverse=True)
            elif gb == "scope":
                order = ["pipeline", "procedure", "convention"]
                keys = [k for k in order if k in grouped] + [k for k in sorted(keys) if k not in order]
            else:
                keys = sorted(keys)
            lines.append(f"\nOrdered by {gb}.\n")
            for k in keys:
                lines.append(f"\n## {k} ({len(grouped[k])})\n")
                for r in grouped[k]:
                    label = r.get(title_field) or r.get(slug_field, "")
                    lines.append(f"- [{str(label)[:80]}]({_slug(str(r.get(slug_field)))}.md)")
        else:
            rows = idx.get("rows", [])
            lines.append("| " + " | ".join(c["field"] for c in rows) + " |")
            lines.append("|" + "|".join("---" for _ in rows) + "|")
            for r in records:
                cells = []
                for col in rows:
                    f = r.get(col["field"])
                    mode = col.get("mode")
                    if mode == "count":
                        cells.append(str(len(f or [])))
                    elif mode == "join":
                        cells.append(",".join(map(str, (f or []))) or "—")
                    elif col.get("link"):
                        cells.append(f"[{f}]({_slug(str(f))}.md)")
                    else:
                        cells.append(str(f or "—")[:50])
                lines.append("| " + " | ".join(cells) + " |")
        _write(adir / "index.md",
               _fm(f"{kind_label}Index", kind_label, f"{len(records)} records") + "\n".join(lines) + "\n")


def _render_artifact_registries(records: list) -> None:
    """Single-file grep-first registries for high-value artifact kinds.

    复数映射暂内联(后续与 SD_LABELS 一并外置到 model_data)。
    """
    by_kind = defaultdict(list)
    for a in records:
        by_kind[a["kind"]].append(a)
    plural = model_data.load_or_empty("ontology/enums", {}).get("artifact_kind_plural",
        {"factor_set": "factor-sets", "server": "servers", "model": "models", "dataset": "datasets"})
    rdir = INDEX_DIR / "registries"
    for kind in ("factor_set", "server", "model", "dataset"):
        items = sorted(by_kind.get(kind, []), key=lambda a: a["name"])
        if not items:
            continue
        lines = [f"# Registry: {plural[kind]}", "", f"**{len(items)}** {kind} artifacts (grep-first registry).", ""]
        for a in items:
            owner = a.get("owner", "") or "—"
            lines.append(f"- **{a['name']}** owner={owner} → [details](../artifacts/{kind}/{_slug(a['name'])}.md)")
        _write(rdir / f"{plural[kind]}.md",
               _fm("Registry", plural[kind], f"{len(items)} {kind}") + "\n".join(lines) + "\n")


def _server_access_section(a: dict, sections_by_id: dict) -> str:
    """Extract SSH access evidence (Host / ProxyJump / HostName / User) for a server
    from its source sections, so server pages carry real access info (design §6.1,
    report test 1). Returns markdown or ''.
    """
    import re as _re
    lines = ["\n## Access\n"]
    found = False
    for sid in a.get("source_sections", []):
        sec = sections_by_id.get(sid)
        if not sec:
            continue
        text = sec.content
        # strip speaker prefixes (e.g. "李四: ") then collect ssh config tokens
        for line in text.splitlines():
            clean = _re.sub(r"^[一-鿿A-Za-z]+[:：]\s*", "", line).strip()
            if _re.search(r"Host\s|ProxyJump|HostName|User\s|ServerAliveInterval", clean):
                lines.append(f"- `{clean[:140]}`")
                found = True
        if found and "ProxyJump" in text:
            break
    if not found:
        return ""
    lines.append("")
    return "\n".join(lines)


def _link_phase2_pages(sources, doc_sections) -> None:
    """Cross-cut entity links into domain/section pages(由 entities.json 的 render 声明驱动)。

    遍历所有带 _render 的实体,按 domains 归组到领域页、按 source_sections 归组到章节页。
    (原「main index 加 Phase 2 段」是死代码:主索引的 entity_indexes 段已含这些链接,guard 恒 false,已移除。)
    """
    import re as _re
    from util.ontology import all_entities

    sections_by_id = {ds.id: ds for ds in doc_sections}

    # 收集各实体:记录 + 渲染元信息(页目录 / slug / 标题 / 标签)
    ents = []
    for name, cls in all_entities().items():
        rcfg = getattr(cls, "_render", {})
        if not rcfg:
            continue
        recs = _load_entity_records(cls)
        if not recs:
            continue
        ents.append({
            "lk": getattr(cls, "_list_key", ""),
            "label": rcfg.get("kind_label", name),
            "page_dir": rcfg.get("page_dir", ""),
            "slug_field": rcfg.get("slug_field", "id"),
            "title_field": rcfg.get("title_field") or rcfg.get("slug_field", "id"),
            "records": recs,
        })
    if not ents:
        return

    # ── 1. Domain pages: add entity-type counts + links ──
    dom_counts = defaultdict(lambda: defaultdict(int))
    for ent in ents:
        for r in ent["records"]:
            for d in (r.get("domains") or []):
                dom_counts[d][ent["lk"]] += 1
    for dname, counts in dom_counts.items():
        domain_page = INDEX_DIR / "domains" / dname / "index.md"
        if not domain_page.exists():
            continue
        current = domain_page.read_text(encoding="utf-8")
        for stale in ["## Phase 2 entities", "## Extracted Knowledge", "## Phase 2"]:
            current = _re.split(rf"\n{_re.escape(stale)}.*?(?=\n## |\Z)", current, flags=_re.DOTALL)[0]
        parts = [f"[{counts[e['lk']]} {e['label']}s](../../{e['page_dir']}/index.md)"
                 for e in ents if counts.get(e['lk'])]
        if parts:
            block = "\n## Phase 2 entities\n\n" + " · ".join(parts) + "\n"
            if "## See also" in current:
                current = current.replace("\n## See also", block + "\n## See also")
            else:
                current = current.rstrip() + "\n" + block
        _write(domain_page, current)

    # ── 2. Section pages: add entity links ──
    sec_refs = defaultdict(list)
    for ent in ents:
        for r in ent["records"]:
            for sid in r.get("source_sections", []):
                sec_refs[sid].append((ent, r))
    for sid, refs in sec_refs.items():
        section_page = INDEX_DIR / "sections" / f"{sid}.md"
        if not section_page.exists():
            continue
        current = section_page.read_text(encoding="utf-8")
        if "## Phase 2 Extractions" in current:
            current = _re.split(r"\n## Phase 2 Extractions.*", current, flags=_re.DOTALL)[0]
        by_lk = defaultdict(list)
        for ent, r in refs:
            by_lk[ent["lk"]].append(r)
        lines = ["\n## Phase 2 Extractions\n"]
        for ent in ents:
            rs = by_lk.get(ent["lk"], [])
            if not rs:
                continue
            lines.append(f"\n### {ent['label']}s ({len(rs)})\n")
            for r in rs:
                slug = _slug(str(r.get(ent["slug_field"], "")))
                title = str(r.get(ent["title_field"]) or r.get(ent["slug_field"], ""))[:80]
                lines.append(f"- [{title}](../{ent['page_dir']}/{slug}.md)")
        _write(section_page, current.rstrip() + "\n" + "\n".join(lines))

    # ── 3. Remove the obsolete by-entity-type/ directory ──
    etype_dir = INDEX_DIR / "indexes" / "by-entity-type"
    if etype_dir.exists():
        import shutil as _shutil
        _shutil.rmtree(etype_dir)
        print("  removed obsolete indexes/by-entity-type/", flush=True)

    print(f"cross-cut {len(ents)} entity types into domain/section pages", flush=True)


# ── CLI ──

if __name__ == "__main__":
    # Full render from scratch — reads everything in knowledge_bits
    import json as _json
    from util.ontology import build_domains

    sources = [SourceRecord(**s) for s in _json.loads((KB_DIR / "sources.json").read_text())]
    doc_sections = [DocSectionRecord(**ds) for ds in _json.loads((KB_DIR / "doc_sections.json").read_text())]
    code_units = [CodeUnitRecord(**cu) for cu in _json.loads((KB_DIR / "code_units.json").read_text())]

    render_index(build_domains(), sources, doc_sections, code_units)
