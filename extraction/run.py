"""Phase 2 pipeline orchestrator — builds cross-section groups, then runs all extractors.

Usage: python -m extraction.run

Flow:
  1. cross_group.build_extraction_groups() — embedding-based semantic grouping
  2. extract_entity("ProtocolRecord", groups["protocols"]) — module group → LLM → protocols
  3. extract_entity("ErrorRecord", groups["errors"])       — group → LLM → errors
  4. extract_entity("WorkflowRecord", groups["workflows"]) — group → LLM → workflows
  5. extract_entity("TaskRecord", groups["tasks"])         — group → LLM → tasks

Output: artifact/knowledge_bits/phase2_{tasks,errors,workflows,protocols}.json
"""

from __future__ import annotations

import json

from util.paths import KB_DIR, ROOT
from extraction.cross_group import build_extraction_groups
from extraction.entities.generic import extract_entity
from util.ontology import get as get_entity, get_optional, to_dict
from extraction.pipeline import PHASE2_DIR, stage, tick, write_json


def main() -> None:
    stage("Phase 2: Structured Knowledge Extraction")

    # ── 0. Build cross-section groups ──
    groups = build_extraction_groups()

    # ── 通用实体抽取:遍历 entities.json 中带 list_key 的实体(不硬编码实体名) ──
    from util.ontology import all_entities
    extracted = {}
    for entity_name, cls in all_entities().items():
        lk = getattr(cls, "_list_key", "")
        if not lk or lk not in groups:
            continue
        records = extract_entity(entity_name, groups[lk])
        write_json(PHASE2_DIR / f"phase2_{lk}.json", [to_dict(r) for r in records])
        tick(f"saved {len(records)} {entity_name}")
        extracted[lk] = [to_dict(r) for r in records]
    # extracted: {list_key -> [dict]},供后续确定性增强按需取用(不硬编码实体名)

    # ── 元函数增强(由 entities.json 的 meta 声明驱动:seed / group / index / synthesize) ──
    from util.ontology import DocSectionRecord, SourceRecord
    # 导入元函数实现包以触发其 @register(把各实体的 seed/group/index/synthesize 登记进注册表)
    import extraction.meta_ops  # noqa: F401
    from extraction.meta import run_meta
    sources = [SourceRecord(**s) for s in json.loads((KB_DIR / "sources.json").read_text())]
    doc_sections = [DocSectionRecord(**d) for d in json.loads((KB_DIR / "doc_sections.json").read_text())]
    _RC = get_optional("ResourceRecord")
    resources = [_RC(**r) for r in json.loads((KB_DIR / "resources.json").read_text())] if _RC else []
    code_units = json.loads((KB_DIR / "code_units.json").read_text()) if (KB_DIR / "code_units.json").exists() else []

    context = {
        "sources": sources, "doc_sections": doc_sections,
        "resources": resources, "code_units": code_units, "root": ROOT,
    }
    results = run_meta(extracted, context, PHASE2_DIR)

    # ── Hub data(theme bundles + sub-domain labels)—— 传通用 results({list_key: [dict]}) ──
    from render.hubgen import build_hub_data
    hub_data = build_hub_data(
        doc_sections=[to_dict(d) for d in doc_sections],
        entities=results,
    )
    write_json(PHASE2_DIR / "hub_data.json", hub_data)
    tick(f"saved hub_data.json ({len(hub_data)} domains, theme bundles)")

    # ── 9. Re-render so person page, entity index, and new families reflect all data ──
    from render.okf_markdown import render_index as _render_index
    from util.ontology import CodeUnitRecord, build_domains
    code_units = [CodeUnitRecord(**cu) for cu in json.loads((KB_DIR / "code_units.json").read_text())]
    _render_index(build_domains(), sources, doc_sections, code_units)
    from render.html import convert_index_to_html
    convert_index_to_html()
    tick("re-rendered index (md + html) with enriched people + new entities")

    # ── Summary ──
    print(f"\n══ Phase 2 Complete ══")
    for lk in sorted(results):
        print(f"  {lk}: {len(results[lk])}")
    print(f"  Output:    {PHASE2_DIR}")


if __name__ == "__main__":
    main()
