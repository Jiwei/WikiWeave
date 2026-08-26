"""归一化代码分析适配层 —— 依赖图 / 调用图 / 控制流图(CFG)。

「各语言最优工具拼装、不重复造轮子」:每门语言用其生态的成熟静态分析工具,输出统一
归一化到同一套边模型,供渲染与知识库消费。

已接入:
  - Python: 调用图 = PyCG,CFG = Scalpel(scalpel.cfg),依赖 = import 解析
待接入(运行时已具备/需装工具链):
  - Java  : Soot(JDK 已就绪)
  - JS/TS : Jelly(Node 已就绪)
  - Go    : golang.org/x/tools/go/callgraph(需装 Go)
  - C/C++ : clang/LLVM(需装 clang)

统一产物(写入 knowledge_bits/):
  code_call_graph.json       {edges:[{caller, callee, caller_unit, callee_unit, external, language}]}
  code_dependency_graph.json {edges:[{from_source, to_source, kind}]}  # 文件级
  code_cfg.json              [{unit, module, function, blocks, edges, branches, cyclomatic, language}]
"""
from __future__ import annotations

import json as _json
import os
import subprocess
import tempfile
from pathlib import Path
from shutil import which
from typing import Any

from util.ontology import CodeUnitRecord, SourceRecord
from util.paths import ROOT


def _tool(name: str, extra: tuple[str, ...] = ()) -> str | None:
    """定位外部 CLI(PATH 优先,再试常见安装位)。"""
    p = which(name)
    if p:
        return p
    for cand in extra:
        cp = Path(os.path.expanduser(cand))
        if cp.exists():
            return str(cp)
    return None


# ── 工具可用性(缺失则该语言分析跳过,不中断流水线) ──

def _pycg_available() -> bool:
    try:
        import PyCG.pycg  # noqa: F401
        return True
    except Exception:
        try:
            import pycg.pycg  # noqa: F401
            return True
        except Exception:
            return False


def _scalpel_cfg_available() -> bool:
    try:
        from scalpel.cfg import CFGBuilder  # noqa: F401
        return True
    except Exception:
        return False


# ── Python 适配器 ──

def _module_dotted(relpath: str) -> str:
    """extraction/run.py → extraction.run"""
    p = relpath
    if p.endswith(".py"):
        p = p[:-3]
    if p.endswith("/__init__"):
        p = p[: -len("/__init__")]
    return p.replace("/", ".")


def analyze_python(
    py_sources: list[SourceRecord],
    code_units: list[CodeUnitRecord],
    root: Path,
) -> dict[str, Any]:
    """PyCG 调用图 + Scalpel CFG + import 依赖。缺工具则对应部分为空。"""
    call_edges: list[dict] = []
    cfgs: list[dict] = []
    dep_edges: list[dict] = []

    # 模块点路径 → source_id(用于把 PyCG 节点/依赖解析回文件)
    # 同时登记「带顶层包名前缀」的变体:输入根名(如 auto_archivist)+ 相对模块,
    # 这样 import "auto_archivist.util.paths" 与相对键 "util.paths" 都能解析。
    root_pkg = Path(root).name
    mod_to_src: dict[str, str] = {}
    for s in py_sources:
        mod = _module_dotted(s.path)
        mod_to_src[mod] = s.id
        if root_pkg:
            mod_to_src[f"{root_pkg}.{mod}"] = s.id
    # 函数短名 → 该文件内 code_unit id(qualified 名末段匹配)
    unit_by_key: dict[tuple[str, str], str] = {}
    for cu in code_units:
        if cu.language != "python":
            continue
        short = cu.name.split(".")[-1]
        unit_by_key[(cu.source_id, short)] = cu.id

    def _resolve_unit(node_name: str) -> str | None:
        """PyCG 节点名(如 extraction.run.main)→ code_unit id(若为库内符号)。"""
        parts = node_name.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            mod = ".".join(parts[:cut])
            if mod in mod_to_src:
                short = parts[-1]
                return unit_by_key.get((mod_to_src[mod], short))
        return None

    def _is_internal(node_name: str) -> bool:
        parts = node_name.split(".")
        for cut in range(len(parts), 0, -1):
            if ".".join(parts[:cut]) in mod_to_src:
                return True
        return False

    # 1) PyCG 调用图
    if _pycg_available() and py_sources:
        try:
            try:
                from PyCG.pycg import CallGraphGenerator
                from PyCG import formats
            except Exception:
                from pycg.pycg import CallGraphGenerator  # type: ignore
                from pycg import formats  # type: ignore
            files = [str(root / s.path) for s in py_sources]
            cg = CallGraphGenerator(files, str(root), -1, "call-graph")
            cg.analyze()
            simple = formats.Simple(cg).generate()
            for caller, callees in simple.items():
                for callee in callees:
                    call_edges.append({
                        "caller": caller,
                        "callee": callee,
                        "caller_unit": _resolve_unit(caller),
                        "callee_unit": _resolve_unit(callee),
                        "external": not _is_internal(callee),
                        "language": "python",
                    })
        except Exception as e:
            print(f"  ⚠ analyze_python: PyCG 失败 ({type(e).__name__}: {e})", flush=True)

    # 2) Scalpel CFG(每函数块级控制流 + 圈复杂度)
    if _scalpel_cfg_available():
        from scalpel.cfg import CFGBuilder
        for s in py_sources:
            try:
                src = (root / s.path).read_text(encoding="utf-8", errors="ignore")
                top = CFGBuilder().build_from_src(Path(s.path).stem, src)
            except Exception:
                continue
            for fcfg in _iter_function_cfgs(top):
                fname = fcfg["name"]
                blocks = fcfg["blocks"]
                n = len(blocks)
                edges = [(b.id, e.target.id) for b in blocks for e in getattr(b, "exits", [])]
                m = len(edges)
                cfgs.append({
                    "unit": unit_by_key.get((s.id, fname)),
                    "source_id": s.id,
                    "module": _module_dotted(s.path),
                    "function": fname,
                    "blocks": n,
                    "edges": m,
                    "branches": sum(1 for b in blocks if len(getattr(b, "exits", [])) > 1),
                    "cyclomatic": (m - n + 2) if n else 0,
                    "language": "python",
                })

    # 3) 文件级依赖(import 解析回库内文件)
    for cu in code_units:
        if cu.kind != "module" or cu.language != "python":
            continue
        for imp in cu.imports:
            base = imp.split(":")[0].strip()
            tid = _resolve_module(base, mod_to_src)
            if tid and tid != cu.source_id:
                dep_edges.append({"from_source": cu.source_id, "to_source": tid, "kind": "import"})

    return {"call_edges": call_edges, "cfgs": cfgs, "dep_edges": dep_edges}


def _iter_function_cfgs(top_cfg) -> list[dict]:
    """遍历 Scalpel CFG 的所有函数(含类方法)。key 形如 (id, name)。"""
    out: list[dict] = []

    def _collect(cfg):
        fmap = getattr(cfg, "functioncfgs", {}) or {}
        for key, fc in fmap.items():
            name = key[1] if isinstance(key, tuple) and len(key) > 1 else str(key)
            out.append({"name": name, "blocks": list(fc)})
            _collect(fc)
        cmap = getattr(cfg, "class_cfgs", {}) or {}
        for _key, cc in cmap.items():
            _collect(cc)

    _collect(top_cfg)
    return out


def _resolve_module(dotted: str, mod_to_src: dict[str, str]) -> str | None:
    """把一个 import 的点路径解析到库内 source_id(尽量长匹配)。"""
    if not dotted:
        return None
    if dotted in mod_to_src:
        return mod_to_src[dotted]
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        cand = ".".join(parts[:cut])
        if cand in mod_to_src:
            return mod_to_src[cand]
    return None


# ── 顶层入口 ──

# ── JS/TS 适配器:madge(文件级依赖图) ──

def analyze_javascript(js_sources: list[SourceRecord], root: Path) -> dict[str, Any]:
    """madge → JS/TS 文件级 import 依赖图。缺 madge 则返回空。"""
    dep_edges: list[dict] = []
    madge = _tool("madge")
    if not madge or not js_sources:
        return {"dep_edges": dep_edges}
    path_to_src = {s.path: s.id for s in js_sources}
    rels = [s.path for s in js_sources]
    try:
        proc = subprocess.run([madge, "--json", *rels], cwd=str(root),
                              capture_output=True, text=True, timeout=180)
        data = _json.loads(proc.stdout or "{}")
    except Exception as e:
        print(f"  ⚠ analyze_javascript: madge 失败 ({type(e).__name__}: {e})", flush=True)
        return {"dep_edges": dep_edges}
    for frm, tos in data.items():
        fs = path_to_src.get(frm)
        for to in tos or []:
            ts = path_to_src.get(to)
            if fs and ts and fs != ts:
                dep_edges.append({"from_source": fs, "to_source": ts, "kind": "import"})
    return {"dep_edges": dep_edges}


# ── PHP 适配器:PDepend(每方法圈复杂度 = CFG 派生指标) ──

def analyze_php(php_sources: list[SourceRecord], code_units: list[CodeUnitRecord],
                root: Path) -> dict[str, Any]:
    """PDepend summary-xml → PHP 函数/方法圈复杂度(ccn)。缺 pdepend 则返回空。"""
    cfgs: list[dict] = []
    pdepend = _tool("pdepend", (
        "~/.config/composer/vendor/bin/pdepend",
        "~/.composer/vendor/bin/pdepend",
    ))
    if not pdepend or not php_sources:
        return {"cfgs": cfgs}
    import xml.etree.ElementTree as ET

    abs_to_src: dict[str, str] = {str((root / s.path).resolve()): s.id for s in php_sources}
    # 短名 → php 可调用单元(供 unit 映射)
    unit_by_short: dict[str, list[CodeUnitRecord]] = {}
    for cu in code_units:
        if cu.language == "php" and cu.kind in _CALLABLE_KINDS:
            unit_by_short.setdefault(cu.name.split(".")[-1], []).append(cu)

    xmlf = tempfile.mktemp(suffix=".xml")
    try:
        subprocess.run([pdepend, f"--summary-xml={xmlf}", str(root)],
                       capture_output=True, text=True, timeout=300)
        tree = ET.parse(xmlf)
    except Exception as e:
        print(f"  ⚠ analyze_php: pdepend 失败 ({type(e).__name__}: {e})", flush=True)
        return {"cfgs": cfgs}

    def _emit(fpath: str, fname: str, ccn: str | None):
        if ccn is None:
            return
        sid = abs_to_src.get(str(Path(fpath).resolve())) if fpath else None
        cands = unit_by_short.get(fname.split(".")[-1], [])
        same = [c for c in cands if c.source_id == sid] if sid else []
        unit = (same or cands or [None])[0]
        cfgs.append({
            "unit": unit.id if unit else None, "source_id": sid,
            "module": Path(fpath).stem if fpath else "",
            "function": fname, "blocks": None, "edges": None, "branches": None,
            "cyclomatic": int(ccn), "language": "php",
        })

    rootx = tree.getroot()
    for pkg in rootx.iter("package"):
        for cls in pkg.findall("class"):
            fnode = cls.find("file")
            fpath = fnode.get("name") if fnode is not None else ""
            for m in cls.findall("method"):
                _emit(fpath, f"{cls.get('name')}.{m.get('name')}", m.get("ccn"))
        for fn in pkg.findall("function"):
            fnode = fn.find("file")
            fpath = fnode.get("name") if fnode is not None else ""
            _emit(fpath, fn.get("name"), fn.get("ccn"))
    return {"cfgs": cfgs}


LANG_SUPPORTED = {"python", "javascript", "typescript", "php"}  # 深度工具已接;其余走名字级调用图
_CALLABLE_KINDS = {"function", "method"}


def _name_based_call_edges(code_units: list[CodeUnitRecord]) -> list[dict]:
    """非 Python 语言:用 tree-sitter 抽出的 calls(被调名)做名字级调用图。

    解析优先级:同文件同短名 > 全局唯一短名 > 判为外部。这是「语法级」调用图,
    不做类型推断,足够知识库用;Python 走 PyCG(更准),不在此处重复。
    """
    by_short: dict[str, list[CodeUnitRecord]] = {}
    for cu in code_units:
        if cu.kind in _CALLABLE_KINDS:
            by_short.setdefault(cu.name.split(".")[-1], []).append(cu)

    edges: list[dict] = []
    for cu in code_units:
        if cu.language == "python" or not cu.calls:
            continue
        for name in cu.calls:
            short = name.split(".")[-1].split("\\")[-1]
            cands = by_short.get(short, [])
            same_file = [c for c in cands if c.source_id == cu.source_id]
            target = None
            if same_file:
                target = same_file[0]
            elif len(cands) == 1:
                target = cands[0]
            edges.append({
                "caller": cu.name, "callee": short,
                "caller_unit": cu.id, "callee_unit": target.id if target else None,
                "external": target is None, "language": cu.language,
            })
    return edges


def run_analysis(sources: list[SourceRecord], code_units: list[CodeUnitRecord],
                 root: Path | None = None) -> dict[str, Any]:
    """按语言分派各自最优工具,合并归一化产物。"""
    root = root or ROOT
    py_sources = [s for s in sources if s.source_type == "code_file" and s.path.endswith(".py")]

    call_edges: list[dict] = []
    cfgs: list[dict] = []
    dep_edges: list[dict] = []

    if py_sources:
        r = analyze_python(py_sources, code_units, root)
        call_edges += r["call_edges"]
        cfgs += r["cfgs"]
        dep_edges += r["dep_edges"]

    # JS/TS:madge 文件级依赖图
    js_sources = [s for s in sources if s.source_type == "code_file"
                  and Path(s.path).suffix.lower() in {".js", ".jsx", ".mjs", ".cjs",
                                                       ".ts", ".mts", ".cts", ".tsx"}]
    if js_sources:
        dep_edges += analyze_javascript(js_sources, root)["dep_edges"]

    # PHP:PDepend 圈复杂度(CFG 派生)
    php_sources = [s for s in sources if s.source_type == "code_file"
                   and Path(s.path).suffix.lower() in {".php", ".php5", ".phtml"}]
    if php_sources:
        cfgs += analyze_php(php_sources, code_units, root)["cfgs"]

    # 非 Python 语言:tree-sitter 名字级调用图(JS/TS/Java/Go/C/C++/PHP)
    call_edges += _name_based_call_edges(code_units)

    call_graph = {
        "edges": call_edges,
        "stats": {
            "edges": len(call_edges),
            "internal": sum(1 for e in call_edges if not e["external"]),
            "external": sum(1 for e in call_edges if e["external"]),
        },
    }
    dependency_graph = {
        "edges": dep_edges,
        "stats": {"edges": len(dep_edges),
                  "files": len({e["from_source"] for e in dep_edges} | {e["to_source"] for e in dep_edges})},
    }
    cfg_stats = {
        "functions": len(cfgs),
        "avg_cyclomatic": round(sum(c["cyclomatic"] for c in cfgs) / len(cfgs), 2) if cfgs else 0,
        "max_cyclomatic": max((c["cyclomatic"] for c in cfgs), default=0),
    }
    return {
        "call_graph": call_graph,
        "dependency_graph": dependency_graph,
        "cfg": cfgs,
        "cfg_stats": cfg_stats,
    }
