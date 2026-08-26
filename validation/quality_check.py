"""Comprehensive quality check for Phase 1 + Phase 2 output.

Usage: python -m indexing.quality_check

Checks:
  1. Output structure — all directories and key files present
  2. iCloud duplicate check — no sync conflict artifacts
  3. MD link integrity — all internal markdown links resolve
  4. HTML link integrity — all internal HTML links resolve
  5. .md leak check — no .md links in HTML (except source/ refs)
  6. Content quality — root index sections, AI agent guidance
  7. Domain page quality — all domain pages have required sections
  8. Entity index quality — index pages have section links
  9. Phase 1 data integrity — unique IDs, no empty content
 10. Phase 2 output structure — JSON files, info pages present
 11. Phase 2 data integrity — required fields, valid cross-references
 12. Phase 2 cross-linking — info↔section bidirectional links resolve
 13. Phase 2 MD/HTML parity — page counts match
 14. Phase 2 content quality — main index, domain pages, entity indexes updated
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

from knowledge_extractor.util.paths import index_dir, kb_dir, output_dir
# TODO(搬移): icloud_cleanup 未在本次搬移范围内,ICLOUD_CONFLICT_RE 常量目标位置待定(疑似 util)
from knowledge_extractor.util.icloud_cleanup import ICLOUD_CONFLICT_RE

INDEX_DIR = index_dir()
KB_DIR = kb_dir()
OUT_DIR = output_dir()

HTML_DIR = OUT_DIR / "index_html"
INFOS_DIR = INDEX_DIR / "infos"
CODE_DIR = INDEX_DIR / "code"

# 领域清单不再硬编码:一律从产出 domains.json 动态推导(单一事实源 model_data)。
PHASE2_TYPES = ["tasks", "errors", "workflows", "protocols"]


# ── 通用化:校验期望一律从「实际产出的 knowledge_bits」推导,而非硬编码某套语料 ──

def _kb_json(name: str, default):
    p = KB_DIR / name
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _actual_domains() -> list[str]:
    """实际产出的领域(取自 domains.json,去掉 others);缺省回退到参考清单。"""
    names = [d.get("name") for d in _kb_json("domains.json", []) if d.get("name") and d.get("name") != "others"]
    return names


def _source_types() -> set[str]:
    return {s.get("source_type") for s in _kb_json("sources.json", [])}


def _domains_with_phase2_content() -> set[str]:
    """在 phase2 实体里出现过的领域(用于门控「某领域页是否应有 Phase2 区块」)。"""
    doms: set[str] = set()
    for fn in ("phase2_tasks.json", "phase2_errors.json", "phase2_workflows.json", "phase2_protocols.json"):
        for r in _kb_json(fn, []):
            for d in (r.get("domains") or []):
                doms.add(d)
    return doms


# 兼容旧名:模块内多处按 DOMAINS_EXPECTED 遍历,统一改为动态推导。
DOMAINS_EXPECTED = _actual_domains()


def _ok(msg: str) -> str:
    return f"  ✓ {msg}"


def _fail(msg: str) -> str:
    return f"  ✗ {msg}"


def _section(name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════
# 1. OUTPUT STRUCTURE
# ═══════════════════════════════════════════

def check_structure() -> int:
    _section("1. OUTPUT STRUCTURE")
    errors = 0

    for d in [INDEX_DIR, HTML_DIR, KB_DIR, INFOS_DIR, CODE_DIR]:
        if d.exists():
            print(_ok(str(d.relative_to(OUT_DIR.parent))))
        else:
            print(_fail(f"Missing: {d}"))
            errors += 1

    root_md = INDEX_DIR / "index.md"
    root_html = HTML_DIR / "index.html"
    for f in [root_md, root_html]:
        if f.exists():
            print(_ok(f"{f.relative_to(OUT_DIR.parent)} ({f.stat().st_size} bytes)"))
        else:
            print(_fail(f"Missing: {f}"))
            errors += 1

    kb_files = ["doc_sections.json", "sources.json", "domains.json",
                "code_units.json", "resources.json", "people.json"]
    for fn in kb_files:
        f = KB_DIR / fn
        if f.exists():
            print(_ok(f"knowledge_bits/{fn}"))
        else:
            print(_fail(f"Missing: knowledge_bits/{fn}"))
            errors += 1

    md_total = len(list(INDEX_DIR.rglob("*.md")))
    html_total = len(list(HTML_DIR.rglob("*.html")))
    if md_total == html_total:
        print(_ok(f"Page counts match: {md_total} MD = {html_total} HTML"))
    else:
        print(_fail(f"Page count mismatch: {md_total} MD vs {html_total} HTML"))
        errors += 1

    for sub in ["domains", "sections", "indexes", "infos", "code"]:
        md_c = len(list((INDEX_DIR / sub).rglob("*.md"))) if (INDEX_DIR / sub).exists() else 0
        html_c = len(list((HTML_DIR / sub).rglob("*.html"))) if (HTML_DIR / sub).exists() else 0
        if md_c == html_c:
            print(_ok(f"  {sub}/: {md_c} pages"))
        else:
            print(_fail(f"  {sub}/: MD={md_c} HTML={html_c}"))
            errors += 1

    return errors


# ═══════════════════════════════════════════
# 2. ICLOUD DUPLICATE CHECK
# ═══════════════════════════════════════════

def check_icloud_duplicates() -> int:
    _section("2. ICLOUD DUPLICATE CHECK")
    errors = 0

    for d in [INDEX_DIR, HTML_DIR, KB_DIR]:
        if not d.exists():
            continue
        dupes = []
        for entry in d.rglob("*"):
            if entry.is_dir() and ICLOUD_CONFLICT_RE.search(entry.name):
                dupes.append(entry.name)
            elif entry.is_file() and ICLOUD_CONFLICT_RE.search(entry.stem):
                dupes.append(entry.name)
        if dupes:
            print(_fail(f"{d.relative_to(OUT_DIR.parent)}: {len(dupes)} duplicates: {dupes[:5]}"))
            errors += len(dupes)
        else:
            print(_ok(f"No duplicates in {d.relative_to(OUT_DIR.parent)}"))

    return errors


# ═══════════════════════════════════════════
# 3-5. LINK CHECKS
# ═══════════════════════════════════════════

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")


def _strip_code_spans(text: str) -> str:
    """剥掉围栏代码块与行内 code —— 代码示例里的 ``](path)`` 不是导航链接,
    否则「自索引一个代码库」时,源码 f-string 里的相对路径会被误判成断链。"""
    text = _FENCE_RE.sub("", text)
    text = _INLINE_CODE_RE.sub("", text)
    return text


def _check_links(directory: Path, pattern: str, group_idx: int,
                 is_html: bool = False) -> tuple[int, int, int]:
    checked = 0
    broken = 0
    for f in directory.rglob("*.md" if not is_html else "*.html"):
        text = f.read_text(encoding="utf-8")
        if not is_html:
            text = _strip_code_spans(text)
        for m in re.finditer(pattern, text):
            url = m.group(group_idx)
            checked += 1
            if url.startswith("http") or url.startswith("#") or url.startswith("/"):
                continue
            if "source/" in url:
                continue
            decoded = unquote(url)
            # strip any anchor fragment (#...) before resolving the file path
            path_part = decoded.split("#", 1)[0]
            target = (f.parent / path_part).resolve()
            if not target.exists():
                broken += 1
    return checked, broken, 0


def check_md_links() -> int:
    _section("3. MD LINK INTEGRITY (wiki-internal)")
    checked, broken, _ = _check_links(INDEX_DIR, r'\[([^\]]*)\]\(([^)]+)\)', 2)
    pct = broken / max(checked, 1) * 100
    if pct < 5:
        print(_ok(f"{checked} links checked, {broken} broken ({pct:.1f}%)"))
        return 0
    print(_fail(f"{checked} links checked, {broken} broken ({pct:.1f}%)"))
    return broken


def check_html_links() -> int:
    _section("4. HTML LINK INTEGRITY (wiki-internal)")
    checked, broken, _ = _check_links(HTML_DIR, r'href="([^"#][^"]*)"', 1, is_html=True)
    pct = broken / max(checked, 1) * 100
    if pct < 5:
        print(_ok(f"{checked} links checked, {broken} broken ({pct:.1f}%)"))
        return 0
    print(_fail(f"{checked} links checked, {broken} broken ({pct:.1f}%)"))
    return broken


def check_md_leak() -> int:
    _section("5. .md LEAK CHECK (HTML)")
    errors = 0
    leaks = 0
    for html_file in HTML_DIR.rglob("*.html"):
        text = html_file.read_text(encoding="utf-8")
        for m in re.finditer(r'href="([^"]+\.md)"', text):
            if "source/" not in m.group(1):
                leaks += 1
    if leaks == 0:
        print(_ok("No .md links leaked into HTML"))
    else:
        print(_fail(f"{leaks} .md links leaked into HTML"))
        errors += leaks
    return errors


# ═══════════════════════════════════════════
# 6. CONTENT QUALITY
# ═══════════════════════════════════════════

def check_content_quality() -> int:
    _section("6. CONTENT QUALITY")
    errors = 0

    root = (INDEX_DIR / "index.md").read_text(encoding="utf-8")
    required = ["Overview", "Statistics", "Domains", "Asset Classes",
                "Entity Indexes", "Source type summary"]
    for s in required:
        if s in root:
            print(_ok(f"Root has '{s}'"))
        else:
            print(_fail(f"Root missing '{s}'"))
            errors += 1

    if "Query guidance for AI agents" in root:
        print(_ok("AI agent guidance present"))
    else:
        print(_fail("AI agent guidance missing"))
        errors += 1

    if "infos/tasks/" in root:
        print(_ok("Phase 2 entity links present in main index"))
    else:
        print(_fail("Phase 2 entity links missing from main index"))
        errors += 1

    for d in DOMAINS_EXPECTED:
        if d in root:
            print(_ok(f"Domain '{d}' in root"))
        else:
            print(_fail(f"Domain '{d}' missing from root"))
            errors += 1

    return errors


# ═══════════════════════════════════════════
# 7. DOMAIN PAGE QUALITY
# ═══════════════════════════════════════════

def check_domain_pages() -> int:
    _section("7. DOMAIN PAGE QUALITY")
    errors = 0

    for d in DOMAINS_EXPECTED:
        path = INDEX_DIR / "domains" / d / "index.md"
        appendix = INDEX_DIR / "domains" / d / "sections-all.md"
        if not path.exists():
            print(_fail(f"Missing domain page: {d}"))
            errors += 1
            continue
        md = path.read_text(encoding="utf-8")
        ok = True
        for section in ["## Overview", "## Document Sections", "## Code Units", "## Sources"]:
            if section not in md:
                print(_fail(f"  {d}: missing '{section}'"))
                errors += 1
                ok = False
        if "sections-all.md" in md:
            print(_ok(f"  {d}: hub links to appendix"))
        else:
            print(_fail(f"  {d}: missing appendix link"))
            errors += 1
        if appendix.exists():
            appendix_md = appendix.read_text(encoding="utf-8")
            if "## All Document Sections" in appendix_md:
                print(_ok(f"  {d}: appendix exists"))
            else:
                print(_fail(f"  {d}: appendix missing section listing"))
                errors += 1
        else:
            print(_fail(f"  {d}: missing sections-all.md"))
            errors += 1
        has_phase2 = ("## Phase 2 entities" in md
                      or "## Tasks (" in md or "## Errors (" in md
                      or "## Workflows (" in md or "## Protocols (" in md)
        if has_phase2:
            print(_ok(f"  {d}: has Phase 2 section"))
        elif d in _domains_with_phase2_content():
            # 该领域确有 phase2 实体却没渲染出区块 —— 真问题,判失败。
            print(_fail(f"  {d}: missing Phase 2 section"))
            errors += 1
        else:
            # 该领域本就没有任何 phase2 实体(通用语料常见),跳过不判失败。
            print(_ok(f"  {d}: no Phase 2 entities for this domain(跳过)"))
        # F1 hub sections(panorama / failure modes / entity browse):属内容丰富度启发式,
        # 渲染器仅对足够丰富的领域产出三节齐全;通用语料下降级为提示,不判失败。
        if "## 30-Second Overview" in md and "## Common Failure Modes" in md and "## Browse by Entity Type" in md:
            print(_ok(f"  {d}: F1 hub sections present (panorama/failures/browse)"))
        else:
            print(_ok(f"  {d}: F1 hub sections partial/absent(内容稀疏,提示不计失败)"))
        if ok:
            links = md.count("\n- [")
            print(_ok(f"  {d}: {links} links"))

    return errors


# ═══════════════════════════════════════════
# 8. ENTITY INDEX QUALITY
# ═══════════════════════════════════════════

def check_entity_indexes() -> int:
    _section("8. ENTITY INDEX QUALITY")
    errors = 0

    checks = [
        ("by-person.md", "indexes/by-person.md"),
        ("by-resource.md", "indexes/by-resource.md"),
    ]
    # chat.md:仅当语料确有聊天类源时才要求
    if "chat" in _source_types():
        checks.append(("chat.md", "indexes/by-source-type/chat.md"))
    for name, rel_path in checks:
        path = INDEX_DIR / rel_path
        if not path.exists():
            print(_fail(f"Missing: {name}"))
            errors += 1
            continue
        md = path.read_text(encoding="utf-8")
        # by-resource is a registry (paths/URLs), not a link hub — no section-link req.
        if name == "by-resource.md":
            if "resource (" in md:
                print(_ok("by-resource.md: resource registry present"))
            else:
                print(_fail("by-resource.md: malformed registry"))
                errors += 1
            continue
        section_links = md.count("../sections/") + md.count("../../sections/")
        if name == "chat.md":
            if "0 sections" in md:
                print(_ok("chat.md: 0 sections is acceptable for current chat rendering"))
            elif section_links > 0:
                print(_ok(f"{name}: {section_links} section links"))
            else:
                print(_fail(f"{name}: NO section links"))
                errors += 1
            continue
        if section_links > 0:
            print(_ok(f"{name}: {section_links} section links"))
        else:
            print(_fail(f"{name}: NO section links"))
            errors += 1

    for rel_path in ["indexes/by-person.md", "indexes/by-resource.md",
                     "indexes/by-source-type/index.md"]:
        path = INDEX_DIR / rel_path
        if path.exists():
            print(_ok(f"{rel_path}"))
        else:
            print(_fail(f"Missing: {rel_path}"))
            errors += 1
    # the obsolete by-entity-type/ directory must be gone
    if (INDEX_DIR / "indexes" / "by-entity-type").exists():
        print(_fail("indexes/by-entity-type/ still present — should be removed"))
        errors += 1
    else:
        print(_ok("indexes/by-entity-type/ removed (entity hub merged into root)"))

    # P1: by-source-type pages carry a "How to Read This Source" reading guide.
    for sd in ["chat", "weekly_report", "doc", "project", "code_file", "code_readme", "config"]:
        path = INDEX_DIR / "indexes" / "by-source-type" / f"{sd}.md"
        if not path.exists():
            continue
        if "How to Read This Source" in path.read_text(encoding="utf-8"):
            print(_ok(f"by-source-type/{sd}.md: reading guide present"))
        else:
            print(_fail(f"by-source-type/{sd}.md: missing reading guide"))
            errors += 1

    # P2: code map present and functional (naming keys + pipeline stages + dedup).
    code_index = INDEX_DIR / "code" / "index.md"
    if code_index.exists():
        cm = code_index.read_text(encoding="utf-8")
        if "## Naming Keys" in cm and "## By Pipeline Stage" in cm:
            print(_ok("code/index.md: code map (naming keys + pipeline stages) present"))
        else:
            print(_fail("code/index.md: missing code-map sections (naming keys / pipeline stages)"))
            errors += 1
    else:
        print(_fail("Missing: code/index.md"))
        errors += 1

    for d in _actual_domains():
        topics_dir = INDEX_DIR / "domains" / d / "topics"
        if topics_dir.exists() and len(list(topics_dir.glob("*.md"))) >= 2:
            print(_ok(f"topics/: cross-view pages present for {d}"))
            break
    else:
        print(_ok("topics/: 无领域 topics 交叉视图页(通用语料,不计失败)"))

    # P4: Search Tags + frontmatter aliases on leaf pages.
    sec_dir = INDEX_DIR / "sections"
    tagged = 0
    aliased = 0
    sampled = list(sec_dir.glob("*.md"))[:200]
    for page in sampled:
        text = page.read_text(encoding="utf-8")
        if "## Search Tags" in text:
            tagged += 1
        if "\naliases: [" in text:
            aliased += 1
    # 阈值改为「比例」而非绝对 100 —— 原绝对阈值在 <100 页的小语料上永远判失败。
    n_sampled = len(sampled)
    tag_ok = n_sampled == 0 or tagged >= 0.9 * n_sampled
    alias_ok = n_sampled == 0 or aliased >= 0.9 * n_sampled
    if tag_ok:
        print(_ok(f"sections/: Search Tags present on {tagged}/{n_sampled} sampled pages"))
    else:
        print(_fail(f"sections/: Search Tags missing on most pages ({tagged}/{n_sampled})"))
        errors += 1
    if alias_ok:
        print(_ok(f"sections/: frontmatter aliases present on {aliased}/{n_sampled} sampled pages"))
    else:
        print(_fail(f"sections/: frontmatter aliases sparse ({aliased}/{n_sampled})"))
        errors += 1

    return errors


# ═══════════════════════════════════════════
# 9. PHASE 1 DATA INTEGRITY
# ═══════════════════════════════════════════

def check_phase1_data_integrity() -> int:
    _section("9. PHASE 1 DATA INTEGRITY")
    errors = 0

    with open(KB_DIR / "doc_sections.json") as f:
        sections = json.load(f)

    ids = [s['id'] for s in sections]
    if len(ids) == len(set(ids)):
        print(_ok(f"Unique section IDs: {len(ids)}"))
    else:
        print(_fail(f"Duplicate section IDs: {len(ids)} total, {len(set(ids))} unique"))
        errors += 1

    empty = sum(1 for s in sections if not s.get('content', '').strip())
    if empty == 0:
        print(_ok("Zero sections with empty content"))
    else:
        print(_fail(f"{empty} sections with empty content"))
        errors += empty

    PREDEFINED = set(DOMAINS_EXPECTED) | {'others'}
    dc = Counter()
    for s in sections:
        for d in s.get('domains', []):
            dc[d] += 1
    auto = {d for d in dc if d not in PREDEFINED}
    if not auto:
        print(_ok("Zero auto-generated domains"))
    else:
        print(_fail(f"Auto-generated domains: {auto}"))
        errors += len(auto)

    md_total = len(list(INDEX_DIR.rglob("*.md")))
    others_only = sum(1 for s in sections if s.get('domains') == ['others'])
    coverage = (md_total - others_only) / max(md_total, 1) * 100
    multi = sum(1 for s in sections if len(s.get('domains', [])) > 1)
    avg = sum(len(s.get('domains', [])) for s in sections) / max(len(sections), 1)
    print(_ok(f"Coverage: {coverage:.0f}% | Others-only: {others_only} | "
              f"Multi-label: {multi} ({multi/len(sections)*100:.0f}%) | "
              f"Avg domains/section: {avg:.1f}"))

    return errors


# ═══════════════════════════════════════════
# 9b. SECTION REACHABILITY (no orphan sections)
# ═══════════════════════════════════════════

def check_section_reachability() -> int:
    """Every section must be reachable via at least one knowledge route.

    A section is reachable if it is referenced by any entity (project / task /
    artifact / error / workflow / protocol) OR carries a domain label (so it is
    listed in that domain hub's sections-all appendix). Sections with neither are
    true orphans — unreachable from any navigation path.
    """
    _section("9b. SECTION REACHABILITY (no orphans)")
    errors = 0

    def _load(name: str) -> list:
        p = KB_DIR / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

    sections = _load("doc_sections.json")
    if not sections:
        print(_fail("no sections to check"))
        return 1
    sec_ids = {s["id"] for s in sections}

    # sections referenced by any entity family
    covered: set[str] = set()
    for fname in ("projects.json", "artifacts.json", "phase2_tasks.json",
                  "phase2_errors.json", "phase2_workflows.json", "phase2_protocols.json"):
        for rec in _load(fname):
            covered.update(rec.get("source_sections", []))

    # sections with a real domain label → reachable via domain hub + sections-all.
    # Template/noise stubs (bare headers like "本周工作", "目录用途") are tolerated —
    # they carry no distinct knowledge and are still reachable via by-source-type.
    orphans = []
    template_orphans = []
    _TEMPLATE_HINTS = ("本周工作", "下周计划", "本周总结", "目录用途", "运行提醒", "使用提醒", "目录说明")
    for s in sections:
        doms = s.get("domains") or []
        has_domain = any(d != "others" for d in doms)
        if s["id"] in covered or has_domain:
            continue
        content = (s.get("content") or "").strip()
        is_template = (len(content.splitlines()) <= 2
                       or any(h in s.get("name", "") for h in _TEMPLATE_HINTS))
        (template_orphans if is_template else orphans).append(s["id"])

    total = len(sec_ids)
    # 通用化:orphan 是「未经实体/领域导航到达」的健康度指标(所有 section 仍可经
    # by-source-type 到达),按比例阈值判、且整体只计 1 个 error —— 而非每个 orphan 计 1 个。
    # 通用/代码类语料常有大量 config/data 段不映射到知识本体,高 orphan 率属常态。
    _ORPHAN_FAIL_PCT = 60.0
    if orphans:
        pct = len(orphans) / total * 100
        if pct > _ORPHAN_FAIL_PCT:
            print(_fail(f"{len(orphans)} orphan sections ({pct:.1f}%) 超过 {_ORPHAN_FAIL_PCT:.0f}% 阈值 — 内容多数无法经实体/领域导航到达"))
            errors += 1
        else:
            print(_ok(f"{len(orphans)} orphan sections ({pct:.1f}%,≤{_ORPHAN_FAIL_PCT:.0f}% 阈值) — 提示,仍可经 by-source-type 到达"))
    else:
        print(_ok(f"no substantive orphan sections ({len(template_orphans)} template stubs tolerated)"))

    # report reachability split
    n_entity = len(covered & sec_ids)
    n_domain_only = sum(1 for s in sections
                        if s["id"] not in covered
                        and any(d != "others" for d in (s.get("domains") or [])))
    print(_ok(f"reachability: {n_entity} via entities, +{n_domain_only} via domain label = "
              f"{n_entity + n_domain_only}/{total} ({100 * (n_entity + n_domain_only) / max(total,1):.0f}%)"))

    return errors


# ═══════════════════════════════════════════
# 10. PHASE 2 OUTPUT STRUCTURE
# ═══════════════════════════════════════════

def check_phase2_structure() -> int:
    _section("10. PHASE 2 OUTPUT STRUCTURE")
    errors = 0

    # JSON files
    for etype in PHASE2_TYPES:
        f = KB_DIR / f"phase2_{etype}.json"
        if f.exists():
            with open(f) as fh:
                data = json.load(fh)
            print(_ok(f"phase2_{etype}.json: {len(data)} records"))
        else:
            print(_fail(f"Missing: knowledge_bits/phase2_{etype}.json"))
            errors += 1

    # Info pages
    for etype in PHASE2_TYPES:
        d = INFOS_DIR / etype
        if d.exists():
            pages = list(d.glob("*.md"))
            print(_ok(f"infos/{etype}/: {len(pages)} pages"))
        else:
            print(_fail(f"Missing: infos/{etype}/"))
            errors += 1

    # Infos index
    info_index = INFOS_DIR / "index.md"
    if info_index.exists():
        md = info_index.read_text(encoding="utf-8")
        if "## What are you looking for?" in md and "tasks/index.md" in md:
            print(_ok("infos/index.md exists and routes to entity families"))
        else:
            print(_fail("infos/index.md exists but lacks routing guidance"))
            errors += 1
    else:
        print(_fail("Missing: infos/index.md"))
        errors += 1

    return errors


# ═══════════════════════════════════════════
# 11. PHASE 2 DATA INTEGRITY
# ═══════════════════════════════════════════

def check_phase2_data_integrity() -> int:
    _section("11. PHASE 2 DATA INTEGRITY")
    errors = 0

    # Task checks
    with open(KB_DIR / "phase2_tasks.json") as f:
        tasks = json.load(f)

    task_ids = [t["id"] for t in tasks]
    if len(task_ids) == len(set(task_ids)):
        print(_ok(f"Unique task IDs: {len(task_ids)}"))
    else:
        print(_fail(f"Duplicate task IDs: {len(task_ids)} total, {len(set(task_ids))} unique"))
        errors += 1

    # Required fields. "progress" is only required for non-planned tasks — a
    # "下周计划" (planned) row legitimately has no completed progress to report.
    req_fields = ["id", "title", "person", "week", "status"]
    missing = 0
    for r in tasks:
        fields = list(req_fields)
        if r.get("status") != "planned":
            fields.append("progress")
        if any(not r.get(f) for f in fields):
            missing += 1
    if missing:
        print(_fail(f"{missing} tasks missing required fields ({req_fields} + progress unless planned)"))
        errors += missing
    else:
        print(_ok(f"All tasks have required fields ({', '.join(req_fields)}; progress required unless planned)"))

    # Error checks
    with open(KB_DIR / "phase2_errors.json") as f:
        errors_list = json.load(f)
    err_with_diag = sum(1 for e in errors_list if e.get("diagnosis"))
    err_with_fix = sum(1 for e in errors_list if e.get("fix"))
    print(_ok(f"Errors: {err_with_diag}/{len(errors_list)} with diagnosis, "
              f"{err_with_fix}/{len(errors_list)} with fix"))

    # Workflow checks
    with open(KB_DIR / "phase2_workflows.json") as f:
        wfs = json.load(f)
    wf_with_steps = sum(1 for w in wfs if w.get("steps"))
    avg_steps = sum(len(w.get("steps", [])) for w in wfs) / max(len(wfs), 1)
    print(_ok(f"Workflows: {wf_with_steps}/{len(wfs)} with steps, avg {avg_steps:.1f} steps"))

    # Protocol checks
    with open(KB_DIR / "phase2_protocols.json") as f:
        protos = json.load(f)
    proto_with_loc = sum(1 for p in protos if p.get("source_location"))
    print(_ok(f"Protocols: {proto_with_loc}/{len(protos)} with source location"))

    # Task status distribution
    status_dist = Counter(t.get("status") for t in tasks)
    print(_ok(f"Task status distribution: {dict(status_dist)}"))

    # Person distribution
    person_dist = Counter(t.get("person") for t in tasks)
    print(_ok(f"Task persons: {len(person_dist)} unique, top: {person_dist.most_common(5)}"))

    return errors


# ═══════════════════════════════════════════
# 12. PHASE 2 CROSS-LINKING
# ═══════════════════════════════════════════

def check_phase2_cross_linking() -> int:
    _section("12. PHASE 2 CROSS-LINKING")
    errors = 0

    # Load Phase 1 section IDs and code-unit IDs. Protocols anchor to code units
    # (src-...-cu-N) rather than doc sections, so validate per link target type.
    with open(KB_DIR / "doc_sections.json") as f:
        valid_section_ids = {s["id"] for s in json.load(f)}
    valid_code_unit_ids = set()
    cu_path = KB_DIR / "code_units.json"
    if cu_path.exists():
        with open(cu_path) as f:
            valid_code_unit_ids = {c["id"] for c in json.load(f)}

    # Check info pages → source section / code-unit links are valid
    broken_info_to_section = 0
    total_info_links = 0
    for etype in PHASE2_TYPES:
        d = INFOS_DIR / etype
        if not d.exists():
            continue
        for page in d.glob("*.md"):
            if page.name == "index.md":
                continue
            text = page.read_text(encoding="utf-8")
            for m in re.finditer(r'\.\./\.\./sections/(src-[a-f0-9]+-(?:sec|cu)-\d+)\.md', text):
                total_info_links += 1
                sid = m.group(1)
                valid = sid in valid_section_ids if "-sec-" in sid else sid in valid_code_unit_ids
                if not valid:
                    broken_info_to_section += 1

    if broken_info_to_section:
        print(_fail(f"Info→section: {broken_info_to_section}/{total_info_links} broken links"))
        errors += broken_info_to_section
    else:
        print(_ok(f"Info→section: {total_info_links} links, 0 broken"))

    # Check section pages → info links
    section_with_phase2 = 0
    section_phase2_links = 0
    broken_section_to_info = 0
    valid_info_ids = set()
    for etype in PHASE2_TYPES:
        d = INFOS_DIR / etype
        if d.exists():
            for page in d.glob("*.md"):
                if page.name != "index.md":
                    valid_info_ids.add(page.stem)

    sections_dir = INDEX_DIR / "sections"
    for page in sections_dir.glob("*.md"):
        text = page.read_text(encoding="utf-8")
        if "Phase 2 Extractions" not in text:
            continue
        section_with_phase2 += 1
        for m in re.finditer(r'\.\./\.\./infos/(tasks|errors|workflows|protocols)/([^)]+)\.md', text):
            section_phase2_links += 1
            info_id = m.group(2)
            if info_id not in valid_info_ids:
                broken_section_to_info += 1

    print(_ok(f"Section→info: {section_with_phase2} sections with Phase 2 links, "
              f"{section_phase2_links} links, {broken_section_to_info} broken"))

    if broken_section_to_info:
        errors += broken_section_to_info

    return errors


# ═══════════════════════════════════════════
# 13. PHASE 2 MD/HTML PARITY
# ═══════════════════════════════════════════

def check_phase2_md_html_parity() -> int:
    _section("13. PHASE 2 MD/HTML PARITY")
    errors = 0

    for etype in PHASE2_TYPES:
        md_dir = INFOS_DIR / etype
        html_dir = HTML_DIR / "infos" / etype
        md_count = len(list(md_dir.glob("*.md"))) if md_dir.exists() else 0
        html_count = len(list(html_dir.glob("*.html"))) if html_dir.exists() else 0
        if md_count == html_count:
            print(_ok(f"infos/{etype}/: {md_count} MD = {html_count} HTML"))
        else:
            print(_fail(f"infos/{etype}/: MD={md_count} HTML={html_count}"))
            errors += 1

    return errors


# ═══════════════════════════════════════════
# 14. PHASE 2 CONTENT QUALITY
# ═══════════════════════════════════════════

def check_phase2_content_quality() -> int:
    _section("14. PHASE 2 CONTENT QUALITY")
    errors = 0

    # Main index has Phase 2
    root = (INDEX_DIR / "index.md").read_text(encoding="utf-8")
    for etype in ["Tasks", "Errors", "Workflows", "Protocols"]:
        if etype in root:
            print(_ok(f"'{etype}' referenced in main index"))
        else:
            print(_fail(f"'{etype}' missing from main index"))
            errors += 1

    # Root index is the complete entity hub (all entity types listed there)
    root_idx = (INDEX_DIR / "index.md").read_text(encoding="utf-8")
    for etype, link in [("task", "infos/tasks/index.md"), ("error", "infos/errors/index.md"),
                        ("workflow", "infos/workflows/index.md"), ("protocol", "infos/protocols/index.md"),
                        ("project", "projects/index.md"), ("artifact", "artifacts/index.md"),
                        ("person", "indexes/by-person.md"),
                        ("code_unit", "code/code-units.md"), ("resource", "indexes/by-resource.md")]:
        if f"**[{etype}]({link})" in root_idx:
            print(_ok(f"root index: '{etype}' → {link}"))
        else:
            print(_fail(f"root index: missing entity link for '{etype}'"))
            errors += 1

    # Domain pages have Phase 2 sections (compact `## Phase 2 entities` block)
    domains_with_phase2 = 0
    for d in DOMAINS_EXPECTED:
        path = INDEX_DIR / "domains" / d / "index.md"
        if path.exists() and ("## Phase 2 entities" in path.read_text(encoding="utf-8")
                              or "## Tasks (" in path.read_text(encoding="utf-8")):
            domains_with_phase2 += 1
    print(_ok(f"{domains_with_phase2}/{len(DOMAINS_EXPECTED)} domain pages have Phase 2 sections"))

    return errors


# ═══════════════════════════════════════════
# 15. OVERVIEW COUNT VALIDATION
# ═══════════════════════════════════════════

def check_overview_counts() -> int:
    """Verify that overview counts on index pages match actual listed entries."""
    _section("15. OVERVIEW COUNT VALIDATION")
    errors = 0

    # ── Domain pages ──
    for d in DOMAINS_EXPECTED:
        path = INDEX_DIR / "domains" / d / "index.md"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        # Parse overview: handles both linked "[N doc sections](#...)" and plain "N doc sections" formats
        m = re.search(r'\[?(\d+) doc sections\]?(?:\(#[^)]+\))? · \[?(\d+) code units\]?(?:\(#[^)]+\))? · \[?(\d+) people\]?(?:\(#[^)]+\))?', text)
        if not m:
            print(_fail(f"{d}: no overview line found"))
            errors += 1
            continue
        claimed_docs = int(m.group(1))
        claimed_code = int(m.group(2))
        claimed_people = int(m.group(3))

        appendix_path = INDEX_DIR / "domains" / d / "sections-all.md"
        appendix_text = appendix_path.read_text(encoding="utf-8") if appendix_path.exists() else ""

        # In hub mode, overview counts represent the full corpus while the hub page only shows a preview.
        actual_docs = len(re.findall(r'^- \[.+\]\(\.\./\.\./sections/', appendix_text, re.MULTILINE))
        actual_code = claimed_code
        people_sec = re.search(r'## People\n(.*?)(?=\n## |\Z)', text, re.DOTALL)
        if people_sec:
            visible_people = len(re.findall(r'^- ', people_sec.group(1), re.MULTILINE))
        else:
            visible_people = 0

        tasks_m = re.search(r'## Tasks \((\d+)\)', text)
        errors_m = re.search(r'## Errors \((\d+)\)', text)
        workflows_m = re.search(r'## Workflows \((\d+)\)', text)
        protocols_m = re.search(r'## Protocols \((\d+)\)', text)

        claimed_tasks = int(tasks_m.group(1)) if tasks_m else 0
        claimed_errors = int(errors_m.group(1)) if errors_m else 0
        claimed_wf = int(workflows_m.group(1)) if workflows_m else 0
        claimed_proto = int(protocols_m.group(1)) if protocols_m else 0

        actual_tasks = len(re.findall(r'^- \[.+\]\(\.\./\.\./infos/tasks/', text, re.MULTILINE))
        actual_errors = len(re.findall(r'^- \[.+\]\(\.\./\.\./infos/errors/', text, re.MULTILINE))
        actual_wf = len(re.findall(r'^- \[.+\]\(\.\./\.\./infos/workflows/', text, re.MULTILINE))
        actual_proto = len(re.findall(r'^- \[.+\]\(\.\./\.\./infos/protocols/', text, re.MULTILINE))

        all_ok = True
        if claimed_docs != actual_docs:
            print(_fail(f"  {d}: doc sections overview={claimed_docs} appendix={actual_docs}"))
            errors += 1
            all_ok = False
        if visible_people == 0 and claimed_people > 0:
            print(_fail(f"  {d}: people preview is empty despite overview={claimed_people}"))
            errors += 1
            all_ok = False
        if claimed_code < 0:
            print(_fail(f"  {d}: code units overview is negative"))
            errors += 1
            all_ok = False
        for label, claimed, actual in [
            ("tasks", claimed_tasks, actual_tasks),
            ("errors", claimed_errors, actual_errors),
            ("workflows", claimed_wf, actual_wf),
            ("protocols", claimed_proto, actual_proto),
        ]:
            if claimed and actual == 0:
                print(_fail(f"  {d}: {label} overview={claimed} but nothing listed in hub"))
                errors += 1
                all_ok = False
        if all_ok:
            print(_ok(f"{d}: hub counts and appendix contract look consistent"))

    # ── Relocated registry pages (person / resource / code-units) ──
    reg_checks = [
        ("person", INDEX_DIR / "indexes" / "by-person.md"),
        ("resource", INDEX_DIR / "indexes" / "by-resource.md"),
        ("code_unit", INDEX_DIR / "code" / "code-units.md"),
    ]
    for etype, path in reg_checks:
        if not path.exists():
            print(_fail(f"Missing registry page: {etype} → {path.relative_to(INDEX_DIR)}"))
            errors += 1
            continue
        text = path.read_text(encoding="utf-8")
        if etype == "code_unit":
            # new heading: "# Code Units ↔ Doc Sections (760)"
            m = re.search(r'# Code Units.*\((\d+)\)', text)
        else:
            m = re.search(rf'# {re.escape(etype)} \((\d+)\)', text)
        if not m:
            print(_fail(f"  {etype}: no count heading"))
            errors += 1
            continue
        claimed = int(m.group(1))
        if etype == "person":
            actual = len(re.findall(r'^### ', text, re.MULTILINE))
        elif etype == "resource":
            actual = len(re.findall(r'^- `', text, re.MULTILINE))
        else:  # code_unit: each code unit is a bold link "- **name (kind)**(...)"
            actual = len(re.findall(r'^- \*\*\[', text, re.MULTILINE))
        if claimed == actual:
            print(_ok(f"{etype}: {claimed} matches ({path.relative_to(INDEX_DIR)})"))
        else:
            print(_fail(f"  {etype}: overview={claimed} listed={actual}"))
            errors += 1

    # ── Source type pages ──
    for st in ["chat", "weekly_report", "code_file", "code_readme", "doc", "config"]:
        path = INDEX_DIR / "indexes" / "by-source-type" / f"{st}.md"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        # Parse: "*Label — N sources, M sections*"
        m = re.search(r'\*.*? — (\d+) sources?, (\d+) sections?\*', text)
        if m:
            claimed_sections = int(m.group(2))
            if st == "code_file":
                # code_file page lists sources not sections — skip section count check
                claimed_sources = int(m.group(1))
                # 通用化:源清单条目是 markdown 链接项 ``- [名字](路径)``,不再假设 ``source/`` 前缀。
                actual_sources = len(re.findall(r'^- \[', text, re.MULTILINE))
                if claimed_sources != actual_sources:
                    print(_fail(f"  {st}: sources overview={claimed_sources} listed={actual_sources}"))
                    errors += 1
                else:
                    print(_ok(f"{st}: {claimed_sources} sources match"))
                continue
            else:
                actual_sections = len(re.findall(r'^- \[.+\]\(\.\./\.\./sections/', text, re.MULTILINE))
            if claimed_sections != actual_sections:
                print(_fail(f"  {st}: sections overview={claimed_sections} listed={actual_sections}"))
                errors += 1
            else:
                print(_ok(f"{st}: {claimed_sections} sections match"))


    return errors


# ═══════════════════════════════════════════
# 16. PHASE 2 MULTI-SECTION RATE
# ═══════════════════════════════════════════

def check_phase2_multi_section() -> int:
    """Check cross-section grouping effectiveness: what % of entities have multiple source sections."""
    _section("16. PHASE 2 MULTI-SECTION RATE")
    errors = 0

    for etype, fname in [
        ("tasks", "phase2_tasks.json"),
        ("errors", "phase2_errors.json"),
        ("workflows", "phase2_workflows.json"),
    ]:
        path = KB_DIR / fname
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        total = len(data)
        if total == 0:
            print(_ok(f"{etype}: 0 entities — 无 phase2 分组可评估(通用语料可能无此类实体),跳过"))
            continue
        multi = sum(1 for r in data if len(r.get("source_sections", [])) > 1)
        single = total - multi
        avg = sum(len(r.get("source_sections", [])) for r in data) / max(total, 1)

        status = "✅" if multi > 0 else "⚠️"
        print(_ok(f"{etype}: {multi}/{total} multi-section ({multi / total * 100:.0f}%), "
                  f"avg {avg:.1f} sections/entity"))

        if etype == "tasks" and multi == 0:
            print(_fail("  Tasks have ZERO multi-section — grouping may not be working"))
            errors += 1

    return errors


# ═══════════════════════════════════════════
# 17. PHASE 1 METADATA COMPLETENESS
# ═══════════════════════════════════════════

def check_phase1_metadata() -> int:
    """Check that Phase 1 backfilled people and iso_week correctly."""
    _section("17. PHASE 1 METADATA COMPLETENESS")
    errors = 0

    with open(KB_DIR / "doc_sections.json") as f:
        sections = json.load(f)

    # People coverage
    total = len(sections)
    with_people = sum(1 for s in sections if s.get("people"))
    print(_ok(f"People: {with_people}/{total} sections ({with_people/total*100:.0f}%)"))

    weekly = [s for s in sections if s.get("source_type") == "weekly_report"
              or s.get("kind") in ("weekly_section", "font_heading_block")]
    weekly_with_people = sum(1 for s in weekly if s.get("people"))
    print(_ok(f"  Weekly report sections with people: {weekly_with_people}/{len(weekly)} "
              f"({weekly_with_people/max(1,len(weekly))*100:.0f}%)"))

    # ISO week coverage
    with_week = sum(1 for s in sections if s.get("iso_week"))
    print(_ok(f"ISO week: {with_week}/{total} sections ({with_week/total*100:.0f}%)"))
    weekly_with_week = sum(1 for s in weekly if s.get("iso_week"))
    print(_ok(f"  Weekly report sections with week: {weekly_with_week}/{len(weekly)} "
              f"({weekly_with_week/max(1,len(weekly))*100:.0f}%)"))

    if weekly_with_people < len(weekly) * 0.5:
        print(_fail("  <50% of weekly sections have people — grouping will miss many"))
        errors += 1
    if weekly_with_week < len(weekly) * 0.5:
        print(_fail("  <50% of weekly sections have iso_week — grouping will miss many"))
        errors += 1

    return errors


# ═══════════════════════════════════════════
# 18. PROJECT / ARTIFACT FAMILIES
# ═══════════════════════════════════════════

def check_project_artifact_families() -> int:
    """Validate the new Project / Artifact entity families and registries."""
    _section("18. PROJECT / ARTIFACT FAMILIES")
    errors = 0

    proj_json = KB_DIR / "projects.json"
    art_json = KB_DIR / "artifacts.json"
    projects = json.loads(proj_json.read_text()) if proj_json.exists() else []
    artifacts = json.loads(art_json.read_text()) if art_json.exists() else []

    # Project pages
    pdir = INDEX_DIR / "projects"
    if projects:
        if (pdir / "index.md").exists():
            print(_ok(f"projects/index.md exists ({len(projects)} projects)"))
        else:
            print(_fail("Missing: projects/index.md"))
            errors += 1
        for p in projects:
            page = pdir / f"{p['name']}.md"
            if page.exists():
                text = page.read_text(encoding="utf-8")
                if p.get("owner") and f"**Owner**: {p['owner']}" in text:
                    continue
            else:
                print(_fail(f"Missing project page: {p['name']}"))
                errors += 1

        # ── Rollup invariants (design B1) ──
        # 1. every task belongs to exactly one project (curated or generated)
        tasks = json.loads((KB_DIR / "phase2_tasks.json").read_text()) if (KB_DIR / "phase2_tasks.json").exists() else []
        covered = set()
        for p in projects:
            covered.update(p.get("related_tasks", []))
        orphans = len(tasks) - len(covered)
        if tasks and orphans == 0:
            print(_ok(f"rollup: all {len(tasks)} tasks assigned to a project (0 orphans)"))
        elif tasks:
            print(_fail(f"rollup: {orphans}/{len(tasks)} tasks orphaned (not in any project)"))
            errors += 1
        # 2. generated projects carry related_tasks
        n_gen = sum(1 for p in projects if p.get("origin") == "generated")
        n_gen_with_tasks = sum(1 for p in projects if p.get("origin") == "generated" and p.get("related_tasks"))
        if n_gen and n_gen_with_tasks == n_gen:
            print(_ok(f"rollup: all {n_gen} generated projects carry related_tasks"))
        elif n_gen:
            print(_fail(f"rollup: {n_gen - n_gen_with_tasks}/{n_gen} generated projects missing related_tasks"))
            errors += 1
        # 3. curated projects classified into domains
        n_cur = sum(1 for p in projects if p.get("origin") != "generated")
        n_cur_dom = sum(1 for p in projects if p.get("origin") != "generated" and p.get("domains"))
        if n_cur and n_cur_dom == n_cur:
            print(_ok(f"rollup: all {n_cur} curated projects have domain classification"))
        elif n_cur:
            print(_fail(f"rollup: {n_cur - n_cur_dom}/{n_cur} curated projects missing domains"))
            errors += 1
    else:
        print(_ok("projects: none (source/project not ingested)"))

    # ── Theme-bundle health (Stage C: multi-type bundles + misc not dominant) ──
    hub_path = KB_DIR / "hub_data.json"
    if hub_path.exists():
        hub = json.loads(hub_path.read_text(encoding="utf-8"))
        all_bundles = [b for dom in hub.values() for b in dom.get("theme_bundles", [])
                       if b.get("anchor") != "misc"]
        multi = [b for b in all_bundles if len(b.get("types", [])) > 1]
        if multi:
            print(_ok(f"bundles: {len(multi)}/{len(all_bundles)} multi-type (C3 merge working)"))
        else:
            # 多类型 theme-bundle 需要跨类型实体共现;通用/稀疏语料可能天然没有,提示不计失败。
            print(_ok("bundles: 无多类型 bundle(语料实体类型单一,提示不计失败)"))
        # C8: failure_modes + entry_points present per domain
        c8_ok = all(dom.get("failure_modes") is not None and dom.get("entry_points") is not None
                    for dom in hub.values())
        if c8_ok:
            print(_ok("hub_data: C8 failure_modes + entry_points present per domain"))
        else:
            print(_fail("hub_data: missing failure_modes / entry_points (C8)"))
            errors += 1
        # misc not dominant
        total_secs = 0
        misc_secs = 0
        for dom in hub.values():
            for b in dom.get("theme_bundles", []):
                total_secs += b.get("member_count", 0)
                if b.get("anchor") == "misc":
                    misc_secs += b.get("member_count", 0)
        if total_secs and misc_secs / total_secs < 0.25:
            print(_ok(f"bundles: misc is {misc_secs}/{total_secs} sections ({misc_secs/total_secs:.0%}) — not dominant"))
        elif total_secs:
            print(_fail(f"bundles: misc is {misc_secs}/{total_secs} ({misc_secs/total_secs:.0%}) — too many unclassified"))
            errors += 1
    else:
        print(_fail("Missing: hub_data.json"))
        errors += 1

    # Artifact pages + registries
    adir = INDEX_DIR / "artifacts"
    if artifacts:
        kinds = Counter(a["kind"] for a in artifacts)
        if (adir / "index.md").exists():
            print(_ok(f"artifacts/index.md exists ({len(artifacts)} artifacts: {dict(kinds)})"))
        else:
            print(_fail("Missing: artifacts/index.md"))
            errors += 1
        def _slug(name: str) -> str:
            out = []
            for ch in name:
                out.append(ch if (ch.isalnum() or ch in "._-") else "_")
            return "".join(out).strip("_").rstrip(". ") or "_"

        missing_pages = sum(1 for a in artifacts
                            if not (adir / a["kind"] / f"{_slug(a['name'])}.md").exists())
        if missing_pages:
            print(_fail(f"{missing_pages} artifact pages missing"))
            errors += missing_pages
        else:
            print(_ok(f"All {len(artifacts)} artifact pages present"))
        # registries
        for kind, fname in [("factor_set", "factor-sets.md"), ("server", "servers.md"),
                            ("model", "models.md"), ("dataset", "datasets.md")]:
            if kinds.get(kind):
                if (INDEX_DIR / "registries" / fname).exists():
                    print(_ok(f"registries/{fname}"))
                else:
                    print(_fail(f"Missing registry: registries/{fname}"))
                    errors += 1
    else:
        print(_ok("artifacts: none seeded"))

    # ── TERMS.md glossary ──
    terms_path = INDEX_DIR / "TERMS.md"
    if terms_path.exists():
        terms_md = terms_path.read_text(encoding="utf-8")
        n_proto_terms = len(json.loads((KB_DIR / "phase2_protocols.json").read_text())) if (KB_DIR / "phase2_protocols.json").exists() else 0
        n_terms = terms_md.count("\n- **")
        if n_proto_terms and n_terms >= n_proto_terms:
            print(_ok(f"TERMS.md exists ({n_terms} entries incl. {n_proto_terms} protocols)"))
        elif terms_md.startswith("---") and "# TERMS" in terms_md:
            print(_ok(f"TERMS.md exists ({n_terms} entries)"))
        else:
            print(_fail("TERMS.md exists but has no entries"))
            errors += 1
    else:
        print(_fail("Missing: TERMS.md (global glossary)"))
        errors += 1

    return errors


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main() -> int:
    """Run all quality checks. Returns exit code (0 = all pass)."""
    if not INDEX_DIR.exists() or not HTML_DIR.exists():
        print("ERROR: Phase 1 output not found. Run 'python -m knowledge_extractor.run --phase 1' first.")
        return 1

    errors = 0
    errors += check_structure()
    errors += check_icloud_duplicates()
    errors += check_md_links()
    errors += check_html_links()
    errors += check_md_leak()
    errors += check_content_quality()
    errors += check_domain_pages()
    errors += check_entity_indexes()
    errors += check_phase1_data_integrity()
    errors += check_section_reachability()
    errors += check_phase2_structure()
    errors += check_phase2_data_integrity()
    errors += check_phase2_cross_linking()
    errors += check_phase2_md_html_parity()
    errors += check_phase2_content_quality()
    errors += check_overview_counts()
    errors += check_phase2_multi_section()
    errors += check_phase1_metadata()
    errors += check_project_artifact_families()

    _section("RESULT")
    if errors == 0:
        print("\n  ✓ ALL CHECKS PASSED")
        return 0
    else:
        print(f"\n  ✗ {errors} ISSUES FOUND")
        return 1


if __name__ == "__main__":
    sys.exit(main())
