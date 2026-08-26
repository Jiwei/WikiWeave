"""通用知识抽取工具 · 顶层编排入口。

串联三段流水线,并在启动时做「model_data 完整性」自适应门禁:
  - 探测语料内容类型(是否含代码/非代码);
  - 若所需 model_data 已定义 → 直接跑流水线(跳过 exploration);
  - 否则 → 提示用户先用智能体执行 exploration 补齐缺失的 model_data。

三段流水线(合并为一个入口,便于阅读且不漏改):
  - Phase 1(_run_phase1):源清点 → 分段 → 抽取 → 代码扫描 → 领域分类 → 建域索引 → 渲染
  - Phase 2(extraction.run.main):结构化实体抽取 + 元函数增强 + 再渲染
  - check(validation.quality_check.main):质量校验门禁

用法:
    python run.py [输入目录] [--phase all|1|2|check] [--skip-check]
    KE_INPUT_PATH=/path/to/corpus python run.py --phase 1

输入目录优先级:CLI 位置参数 > 环境变量 KE_INPUT_PATH > config/settings.json 的 input_path。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from util import model_data
from util.ontology import build_domains, to_dict
from util.paths import INDEX_DIR, KB_DIR, OUT_DIR, input_path
from extraction.pipeline import stage, tick, write_json


def _check_model_data_ready() -> list[str]:
    """探测内容类型并返回缺失的 model_data 键(空列表=已就绪,可跳过 exploration)。"""
    from exploration.inventory import detect_content_types
    has_code, has_non_code = detect_content_types()
    return model_data.missing_model_data(has_code, has_non_code)


# ── Phase 1:领域分类(LLM 主 / 关键词兜底) ──

def _classify_doc_sections(doc_sections, source_by_id, llm_available: bool) -> None:
    from segmentation.classify import (
        classify_batch_with_llm,
        classify_keywords,
    )
    from util.llm import phase_settings

    p1 = phase_settings("classify")
    if llm_available and doc_sections:
        n_batches = len(doc_sections) // p1["batch_size"] + 1
        tick(f"classifying {len(doc_sections)} sections (LLM, {n_batches} batches)...")
        items = [(ds, f"{ds.name} {ds.kind} {ds.content}") for ds in doc_sections]
        classify_batch_with_llm("doc_section", items, batch_size=p1["batch_size"])
    else:
        tick(f"LLM unavailable — classifying {len(doc_sections)} sections (keyword)...")
        for ds in doc_sections:
            src = source_by_id.get(ds.source_id)
            text = f"{ds.name} {ds.kind} {src.path if src else ''} {ds.content}"
            ds.domains = classify_keywords(text)
    remaining = sum(1 for ds in doc_sections if ds.domains == ["others"])
    tick(f"  done: {len(doc_sections) - remaining} classified, {remaining} 'others'")


def _classify_code_units(code_units, source_by_id) -> None:
    from segmentation.classify import classify_keywords
    for cu in code_units:
        src = source_by_id.get(cu.source_id)
        cu.domains = classify_keywords(f"{cu.name} {cu.kind} {src.path if src else ''}")


def _run_phase1() -> None:
    """Phase 1:源清点 → 分段 → 抽取 → 代码扫描 → 领域分类 → 建域索引 → 渲染。"""
    from exploration.inventory import (
        build_source_inventory,
        collect_deferred_files,
    )
    from segmentation.doc.sections import build_section_inventory
    from segmentation.code.scan import link_code_to_docs, scan_codebase
    from util.llm import available
    from render.okf_markdown import render_index
    from render.html import convert_index_to_html

    domains = build_domains()

    stage("Phase 1: 通用知识抽取")
    base = input_path()
    print(f"  输入目录: {base}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    KB_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    stage("1. 源清点")
    sources = build_source_inventory()
    tick(f"{len(sources)} sources inventoried")

    stage("2. 分段(文档)")
    doc_sections = build_section_inventory(sources)
    doc_sections.sort(key=lambda ds: (ds.source_id, ds.id))
    tick(f"{len(doc_sections)} doc sections")

    stage("3. 代码扫描")
    code_units, code_modules, code_asts = scan_codebase(sources)
    tick(f"{len(code_units)} code units, {len(code_modules)} modules, {len(code_asts)} AST trees")

    source_by_id = {s.id: s for s in sources}
    link_code_to_docs(doc_sections, code_units, source_by_id)

    # 代码域(code_domain)分类:关联到 domains 的文件级分类
    from segmentation.code.code_map import classify_code_modules
    classify_code_modules(code_modules, code_units)
    tick(f"{len(code_modules)} code modules classified into code domains")

    stage("5. 领域分类")
    _classify_doc_sections(doc_sections, source_by_id, available())
    _classify_code_units(code_units, source_by_id)

    stage("6b. 代码分析(依赖图 / 调用图 / CFG)")
    code_analysis = {"call_graph": {"edges": [], "stats": {}},
                     "dependency_graph": {"edges": [], "stats": {}},
                     "cfg": [], "cfg_stats": {}}
    try:
        from segmentation.code.analysis import run_analysis
        code_analysis = run_analysis(sources, code_units, base)
        cg, dg = code_analysis["call_graph"]["stats"], code_analysis["dependency_graph"]["stats"]
        tick(f"调用图 {cg.get('edges', 0)} 边(内部 {cg.get('internal', 0)}), "
             f"依赖 {dg.get('edges', 0)} 边, CFG {code_analysis['cfg_stats'].get('functions', 0)} 函数 "
             f"(avg 圈复杂度 {code_analysis['cfg_stats'].get('avg_cyclomatic', 0)})")
    except Exception as e:  # 代码深度分析是增强项,失败不中断核心 KB
        tick(f"代码分析跳过: {type(e).__name__}: {e}")

    stage("7. 写知识单元(knowledge_bits)")
    write_json(KB_DIR / "sources.json", [to_dict(s) for s in sources])
    write_json(KB_DIR / "domains.json", [to_dict(d) for d in domains])
    write_json(KB_DIR / "doc_sections.json", [to_dict(ds) for ds in doc_sections])
    write_json(KB_DIR / "code_units.json", [to_dict(cu) for cu in code_units])
    write_json(KB_DIR / "code_modules.json", [to_dict(cm) for cm in code_modules])
    write_json(KB_DIR / "code_asts.json", code_asts)
    write_json(KB_DIR / "code_call_graph.json", code_analysis["call_graph"])
    write_json(KB_DIR / "code_dependency_graph.json", code_analysis["dependency_graph"])
    write_json(KB_DIR / "code_cfg.json", {"functions": code_analysis["cfg"], "stats": code_analysis["cfg_stats"]})

    deferred_files = collect_deferred_files()
    deferred_payload = [{
        "path": p.relative_to(base).as_posix() if p.is_absolute() else str(p),
        "file_type": p.suffix.lstrip(".").lower() or "unknown",
        "size_bytes": p.stat().st_size if p.exists() else 0,
        "reason": "deferred: unsupported file type (not code/doc/config)",
    } for p in deferred_files]
    write_json(KB_DIR / "deferred_sources.json", deferred_payload)
    tick("knowledge_bits written")

    stage("8. 工程聚类(代码堆)")
    try:
        from render.project_clusters import write_project_clusters
        pc = write_project_clusters()
        st = pc.get("stats", {})
        tick(f"{st.get('clusters', 0)} clusters, "
             f"{st.get('canonical_files', 0)} canonical / {st.get('duplicate_files', 0)} duplicate")
    except Exception as e:
        tick(f"工程聚类跳过: {type(e).__name__}: {e}")

    stage("9. 渲染索引")
    render_index(domains, sources, doc_sections, code_units)
    try:
        from render.code_analysis_page import render_code_analysis
        if render_code_analysis(code_analysis, code_units, sources):
            tick("code analysis page rendered")
    except Exception as e:
        tick(f"代码分析页跳过: {type(e).__name__}: {e}")
    convert_index_to_html()
    tick("index + html rendered")

    print(f"\nPhase 1 → {OUT_DIR}")
    print(f"  sources:      {len(sources)}")
    print(f"  doc_sections: {len(doc_sections)}")
    print(f"  code_units:   {len(code_units)}")
    print(f"  code_modules: {len(code_modules)}")
    print(f"  deferred:     {len(deferred_payload)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="通用知识抽取工具 · 顶层编排")
    parser.add_argument("input_dir", nargs="?", default=None,
                        help="待抽取的本地目录(可省略,改用 KE_INPUT_PATH / settings)")
    parser.add_argument("--phase", default="all", choices=["all", "1", "2", "check"],
                        help="运行哪一段(默认全跑)")
    parser.add_argument("--skip-check", action="store_true",
                        help="跳过 model_data 完整性门禁(强制运行)")
    parser.add_argument("--explore", action="store_true",
                        help="先调用 Claude Code 智能体执行 exploration(EDA → 写回 model_data)")
    parser.add_argument("--lang", default=None, choices=["en", "zh"],
                        help="artifact 主要语言(默认 en,可被 settings.language 覆盖)")
    args = parser.parse_args(argv)

    if args.input_dir:
        os.environ["KE_INPUT_PATH"] = str(Path(args.input_dir).expanduser().resolve())
    if args.lang:
        os.environ["KE_LANG"] = args.lang

    # 输入路径校验(未设置时给清晰报错)
    try:
        input_path()
    except ValueError as e:
        print(f"✋ {e}", flush=True)
        return 1

    # ── --explore: 调用 Claude Code 智能体执行 exploration(EDA → 写回 model_data) ──
    if args.explore:
        from exploration.agent import invoke_exploration
        rc = invoke_exploration()
        if rc != 0:
            return rc

    # ── model_data 完整性门禁(自适应):已定义 → 跳过 exploration;否则提示先 exploration ──
    if args.phase in ("all", "1") and not args.skip_check:
        missing = _check_model_data_ready()
        if missing:
            print("✋ model_data 不完整,需先执行 exploration 定义以下文件:", flush=True)
            for m in missing:
                print(f"  · {m}", flush=True)
            print("\n请运行 `python run.py --explore` 调用 Claude Code 智能体执行 exploration,"
                  "或手动完善 model_data/ 后重跑。\n"
                  "(如需强制跳过门禁,加 --skip-check)", flush=True)
            return 1

    if args.phase in ("all", "1"):
        _run_phase1()

    if args.phase in ("all", "2"):
        from extraction.run import main as phase2_main
        phase2_main()

    if args.phase in ("all", "check"):
        from validation.quality_check import main as check_main
        rc = check_main()
        return int(rc or 0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
