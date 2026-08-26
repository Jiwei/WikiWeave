"""渲染「代码分析」页(依赖图 / 调用图 / 控制流)到 index/code/analysis.md。

从 run_analysis 产出的归一化结果生成一页人类可读的知识库摘要:语言分布、调用/依赖/CFG
概览、圈复杂度热点、被调入度排行、依赖出度排行。convert_index_to_html 会自动转成 HTML。
"""
from __future__ import annotations

from collections import Counter

from knowledge_extractor.util.i18n import t
from knowledge_extractor.util.paths import INDEX_DIR


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def render_code_analysis(code_analysis: dict, code_units: list, sources: list | None = None) -> str | None:
    """写 index/code/analysis.md,返回路径字符串;无代码分析数据则跳过。"""
    cg = code_analysis.get("call_graph", {})
    dg = code_analysis.get("dependency_graph", {})
    cfgs = code_analysis.get("cfg", []) or []
    cfg_stats = code_analysis.get("cfg_stats", {})
    call_edges = cg.get("edges", []) or []
    dep_edges = dg.get("edges", []) or []
    if not (call_edges or dep_edges or cfgs):
        return None

    # 语言分布
    lang_count = Counter(getattr(cu, "language", "python") for cu in code_units if getattr(cu, "kind", "") != "module")
    unit_name = {cu.id: cu.name for cu in code_units}

    lines: list[str] = []
    lines.append(f"---\ntype: CodeAnalysis\ntitle: {t('code_analysis_title')}\n---\n")
    lines.append(f"# {t('code_analysis_title')}\n")
    lines.append("> 由各语言最优静态分析工具产出并归一化:Python 调用图=PyCG、CFG=Scalpel;"
                 "其余语言 AST/调用点=tree-sitter(名字级)。\n")

    # 概览
    lines.append(f"## {t('overview')}\n")
    lines.append(f"- 语言分布(定义单元):{', '.join(f'{k} {v}' for k, v in lang_count.most_common()) or '—'}")
    lines.append(f"- 调用图:{cg.get('stats', {}).get('edges', len(call_edges))} 边"
                 f"(内部 {cg.get('stats', {}).get('internal', 0)} / 外部 {cg.get('stats', {}).get('external', 0)})")
    lines.append(f"- 依赖图(文件级 import):{dg.get('stats', {}).get('edges', len(dep_edges))} 边,"
                 f"涉及 {dg.get('stats', {}).get('files', 0)} 文件")
    lines.append(f"- 控制流:{cfg_stats.get('functions', len(cfgs))} 函数,"
                 f"平均圈复杂度 {cfg_stats.get('avg_cyclomatic', 0)},最高 {cfg_stats.get('max_cyclomatic', 0)}\n")

    # 圈复杂度热点
    if cfgs:
        top = sorted(cfgs, key=lambda c: -(c.get("cyclomatic") or 0))[:15]
        _cell = lambda v: "—" if v is None else v
        rows = [[f"`{c.get('module','')}.{c.get('function','')}`", c.get("language", ""),
                 _cell(c.get("blocks")), _cell(c.get("edges")), c.get("cyclomatic", 0)] for c in top]
        lines.append(f"## {t('cyclomatic_hotspots')}\n")
        lines.append("> 圈复杂度 = 边 − 节点 + 2,越高越难测试/维护,是重构优先级信号。\n")
        lines.append(_table([t('function'), t('language'), "blocks", "edges", t('cyclomatic')], rows) + "\n")

    # 被调入度排行(内部符号)
    in_deg: Counter = Counter()
    for e in call_edges:
        if not e.get("external") and e.get("callee_unit"):
            in_deg[e["callee_unit"]] += 1
    if in_deg:
        rows = [[f"`{unit_name.get(uid, uid)}`", cnt] for uid, cnt in in_deg.most_common(15)]
        lines.append(f"## {t('most_called_symbols')}\n")
        lines.append(_table([t('symbol'), t('called')], rows) + "\n")

    # 依赖出度排行(文件)
    out_deg: Counter = Counter()
    src_name = {s.id: s.path for s in (sources or [])}
    for e in dep_edges:
        out_deg[e["from_source"]] += 1
    if out_deg:
        rows = [[f"`{src_name.get(sid, sid)}`", cnt] for sid, cnt in out_deg.most_common(15)]
        lines.append(f"## {t('heaviest_imports')}\n")
        lines.append(_table([t('file'), t('imports_in_repo')], rows) + "\n")

    out_dir = INDEX_DIR / "code"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "analysis.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)
