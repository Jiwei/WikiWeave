"""Self-indexing layer — TERMS.md + Search Tags + backlink baking (Stage E).

Renders:
  - artifact/index/TERMS.md — global glossary (grep first stop), aggregated from
    Protocol / Project / Artifact records.
  - Search Tags block on every leaf page — aliases + question forms + related
    links, derived deterministically from the page's own metadata (people,
    domains, asset classes, entities) so `grep` finds it via many spellings.
  - "Referenced by" backlinks on section pages — built from the inbound link
    graph (code units' doc_mentions + Phase 2 entities' source_sections), with
    noise filtered (checkpoint/jupyter refs dropped).
  - semantic aliases in section frontmatter — `aliases` field (display name /
    codename) so identity stays stable while search finds it by alias.

All deterministic — no LLM. LLM alias generation is a possible later enhancement.

Usage:
    python -m indexing.selfindex            (reads knowledge_bits, writes index/)
    python -m indexing.render_index         (calls render_terms() + render_selfindex())
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from util.paths import index_dir, kb_dir

INDEX_DIR = index_dir()
KB_DIR = kb_dir()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _fm(title: str, description: str) -> str:
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"---\ntype: Glossary\ntitle: {title}\ndescription: {description}\ntimestamp: {ts}\n---\n\n"


ALIAS_PROMPT = """You are building a glossary alias table for a knowledge base. For each canonical term below, list the alternative names / abbreviations / shorthands that might be used for it (e.g. a person's full name → their initials, a Chinese term → its English acronym or pinyin). Focus on aliases that would actually appear in the source materials (chat logs, reports, or code).

Return a JSON object mapping canonical term → array of alias strings. Return at most 4 aliases per term, and an empty array if a term has no common alias.

Terms:
{items}"""


def _derive_term_aliases(canonical: list[str]) -> dict[str, list[str]]:
    """LLM-derive aliases for canonical terms; returns {canonical: [aliases]}."""
    # TODO(搬移): 新 util.llm 接口为 chat/extract_json/available,不再提供 LlmConfig/_call_llm;
    # 调用点算法逻辑按要求保留未改,后续需按新接口适配
    from util.llm import LlmConfig, _call_llm
    import re as _re

    cfg = LlmConfig.from_env()
    if not cfg.available:
        return {}

    # batch in chunks of ~80 to keep the prompt bounded
    aliases: dict[str, list[str]] = {}
    chunk_size = 80
    for i in range(0, len(canonical), chunk_size):
        chunk = canonical[i:i + chunk_size]
        prompt = ALIAS_PROMPT.format(items="\n".join(f"- {c}" for c in chunk))
        try:
            raw = _call_llm(prompt, cfg)
        except Exception:
            continue
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        if not m:
            continue
        import json as _json
        try:
            parsed = _json.loads(m.group())
        except Exception:
            continue
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                if isinstance(v, list):
                    aliases[k] = [str(a).strip() for a in v if str(a).strip()]
    return aliases


def render_terms() -> None:
    """Render TERMS.md from protocols + projects + artifacts + LLM aliases."""
    def _load(name: str) -> list:
        p = KB_DIR / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return []

    protocols = _load("phase2_protocols.json")
    projects = _load("projects.json")
    artifacts = _load("artifacts.json")

    # LLM aliases for canonical terms (pinyin abbrevs, acronyms, cross-language)
    canonical = []
    for p in protocols:
        n = (p.get("name", "") or "").strip()
        if n:
            canonical.append(n)
    for p in projects:
        n = (p.get("title") or p.get("name") or "").strip()
        if n:
            canonical.append(n)
    for a in artifacts:
        n = (a.get("name", "") or "").strip()
        if n:
            canonical.append(n)
    aliases = _derive_term_aliases(canonical[:400])
    # persist terms.json for downstream use
    (KB_DIR / "terms.json").write_text(
        json.dumps(aliases, ensure_ascii=False, indent=2), encoding="utf-8")

    parts = [
        "# TERMS — Global Glossary",
        "",
        "**First stop for unknown terms.** Any unfamiliar name, abbreviation, or "
        "jargon you hit should be `grep`-able in this file. Each entry maps a "
        "canonical term to a one-line definition and the entity page that explains it.",
        "",
        f"- {len(protocols)} protocol terms (conventions, rules, metrics)",
        f"- {len(projects)} project codenames",
        f"- {len(artifacts)} named deliverables (models, factor sets, datasets, servers, scripts)",
        "",
    ]

    def _aliases_str(name: str) -> str:
        al = aliases.get(name, [])
        # also match by substring (project title ↔ codename variants)
        if not al:
            for cand, lst in aliases.items():
                if name and (name in cand or cand in name):
                    al = lst
                    break
        return f" `≡ {' / '.join(al[:4])}`" if al else ""

    # ── Protocols (the richest term source) ──
    parts.append(f"## Protocols ({len(protocols)})\n")
    for p in sorted(protocols, key=lambda x: x.get("name", "").lower()):
        name = p.get("name", "").strip()
        if not name:
            continue
        definition = (p.get("definition", "") or "").strip().replace("\n", " ")[:140]
        dom = ",".join(p.get("domains", [])) or "—"
        link = f"[{name}](infos/protocols/{p['id']}.md)"
        parts.append(f"- **{link}**{_aliases_str(name)} [{dom}] — {definition}")
    parts.append("")

    # ── Projects ──
    if projects:
        parts.append(f"## Projects ({len(projects)})\n")
        for p in sorted(projects, key=lambda x: x.get("name", "").lower()):
            name = p.get("name", "")
            title = p.get("title") or name
            task_type = (p.get("task_type", "") or "").strip().replace("\n", " ")[:120]
            link = f"[{name}](projects/{name}.md)"
            parts.append(f"- **{link}**{_aliases_str(title)} — {task_type or p.get('output_positioning','')}")
        parts.append("")

    # ── Artifacts ──
    if artifacts:
        parts.append(f"## Artifacts ({len(artifacts)})\n")
        for a in sorted(artifacts, key=lambda x: x.get("name", "").lower()):
            name = a.get("name", "")
            kind = a.get("kind", "")
            summary = (a.get("summary", "") or "").strip().replace("\n", " ")[:120]
            slug = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in name).strip("_") or "_"
            link = f"[{name}](artifacts/{kind}/{slug}.md)"
            parts.append(f"- **{link}**{_aliases_str(name)} ({kind}) — {summary}")
        parts.append("")

    # ── Grep hint ──
    parts.append("## How to use\n")
    parts.append("- Unknown acronym / code name? `grep <name> TERMS.md`.\n")
    parts.append("- Cross-language alias (e.g. pinyin abbrev)? Search Tags on entity pages will cover that in a later pass.\n")
    parts.append("- Not here? Try [Protocols](infos/protocols/index.md) or the root [decision tree](index.md).\n")

    _write(INDEX_DIR / "TERMS.md", _fm("Glossary", f"{len(protocols)} terms") + "\n".join(parts))
    print(f"  TERMS.md → {INDEX_DIR / 'TERMS.md'} ({len(protocols)} protocol terms)", flush=True)


# ── Search Tags + backlinks + frontmatter aliases (P4) ──

_NOISE_RE = re.compile(r"checkpoint|\.ipynb", re.IGNORECASE)


def _section_alias_terms(section: dict) -> list[str]:
    """Deterministic alias terms for a section: name + people + asset classes."""
    terms = []
    name = section.get("name", "")
    if name:
        terms.append(name[:60])
    terms.extend(section.get("people", [])[:3])
    terms.extend(d for d in (section.get("domains") or [])[:3] if d != "others")
    # dedupe preserving order
    seen, out = set(), []
    for t in terms:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _load(name: str) -> list:
    p = KB_DIR / name
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return d if isinstance(d, list) else []
        except Exception:
            return []
    return []


def _load_json(name: str) -> dict:
    p = KB_DIR / name
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}
    return {}


def _build_inbound_links() -> dict[str, list[tuple[str, str]]]:
    """Build inbound-link graph for sections: {section_id: [(type, label)]}.

    Sources: code_units.doc_mentions (filtered noise) + Phase 2 entity
    source_sections. Returns section_id → [(kind, human-label)].
    """
    inbound: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for cu in _load("code_units.json"):
        for m in cu.get("doc_mentions", []):
            sid = m.split(":")[0]
            if "-sec-" not in sid:
                continue
            cu_name = cu.get("name", "")
            if _NOISE_RE.search(cu_name):
                continue
            inbound[sid].append(("code", cu_name[:50]))
    for etype, fname in [("task", "phase2_tasks.json"), ("error", "phase2_errors.json"),
                         ("workflow", "phase2_workflows.json"), ("protocol", "phase2_protocols.json")]:
        for r in _load(fname):
            label = r.get("title") or r.get("name") or ""
            for sid in r.get("source_sections", []):
                inbound[sid].append((etype, str(label)[:50]))
    # dedupe per (sid, label)
    out = {}
    for sid, refs in inbound.items():
        seen, kept = set(), []
        for kind, label in refs:
            key = f"{kind}:{label}"
            if key not in seen:
                seen.add(key)
                kept.append((kind, label))
        out[sid] = kept
    return out


def render_selfindex() -> None:
    """Write Search Tags + Referenced-by blocks and frontmatter aliases on leaf pages."""
    sections = _load("doc_sections.json")
    sections_by_id = {s["id"]: s for s in sections}
    inbound = _build_inbound_links()

    sec_dir = INDEX_DIR / "sections"
    updated = 0
    for ds in sections:
        sid = ds["id"]
        page = sec_dir / f"{sid}.md"
        if not page.exists():
            continue
        text = page.read_text(encoding="utf-8")
        body = text.split("---", 2)[2] if text.startswith("---") else text

        # ── Search Tags block ──
        aliases = _section_alias_terms(ds)
        tags_block = "\n\n---\n## Search Tags\n\n"
        if aliases:
            tags_block += "别名: " + ", ".join(aliases) + "\n"
        # question forms
        name = ds.get("name", "")[:60]
        q = []
        if name:
            q.append(f'"{name}"')
        for p in (ds.get("people") or [])[:2]:
            q.append(f'"{p}" 做了什么')
        if q:
            tags_block += "问题: " + " · ".join(q) + "\n"
        # related links: same source / same domain
        related = []
        same_src = [x for x in sections if x["source_id"] == ds.get("source_id") and x["id"] != sid][:3]
        for x in same_src:
            related.append(f"[[{x['id']}]]")
        if related:
            tags_block += "相关: " + " ".join(related) + "\n"

        # ── Referenced by (dedup, cap 15) ──
        refs = inbound.get(sid, [])[:15]
        if refs:
            tags_block += "\n## Referenced by\n\n"
            for kind, label in refs:
                if kind == "code":
                    tags_block += f"- code: `{label}`\n"
                else:
                    tags_block += f"- {kind}: {label}\n"

        # strip any existing blocks, then append
        body = re.split(r"\n---\n## Search Tags", body)[0]
        body = re.split(r"\n## Referenced by", body)[0]
        # add frontmatter aliases
        new_fm = _inject_frontmatter_aliases(text, aliases)
        page.write_text(new_fm + body.rstrip() + "\n" + tags_block, encoding="utf-8")
        updated += 1

    print(f"  selfindex: Search Tags + backlinks on {updated}/{len(sections)} section pages", flush=True)


def _inject_frontmatter_aliases(text: str, aliases: list[str]) -> str:
    """Add an `aliases: [...]` field to existing frontmatter (idempotent)."""
    if not text.startswith("---"):
        return text
    end = text.find("\n---\n", 4)
    if end == -1:
        return text
    fm = text[4:end]
    body_rest = text[end:]
    # strip existing aliases line
    fm_lines = [ln for ln in fm.splitlines() if not ln.strip().startswith("aliases:")]
    if aliases:
        fm_lines.append(f"aliases: [{', '.join(a for a in aliases)}]")
    return "---\n" + "\n".join(fm_lines) + body_rest


if __name__ == "__main__":
    render_terms()
    render_selfindex()
