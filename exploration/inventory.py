"""源清点(exploration):通用递归扫描任意输入目录,按扩展名/路径规则分类。

与原 task_assistant/indexing/inventory.py 的差异:
  - 输入不再依赖 ``source/`` 目录白名单(chat/weekly_report/...),而是递归扫 ``input_path()``。
  - 语料特有的「路径 → 来源类型」路由单一事实源为 model_data/sources/source_types.json
    (改主题时增删其中的 path_contains 规则即可)。
  - 通用「扩展名 → 代码/文档/配置」映射为跨语料一致的常量;非支持扩展名进入 deferred。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from knowledge_extractor.util import model_data
from knowledge_extractor.util.ontology import SourceRecord
from knowledge_extractor.util.paths import input_path

IGNORED_NAMES = {".DS_Store"}
IGNORED_PREFIXES = (".~",)

# ── 通用扩展名 → 类型映射(跨语料一致,不随主题变化) ──
CODE_ALLOWED_SUFFIXES = {
    ".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".mts", ".cts", ".tsx",
    ".java", ".go", ".c", ".h", ".cc", ".cpp", ".cxx", ".c++",
    ".hpp", ".hh", ".hxx", ".php", ".php5", ".phtml",
}
DOC_ALLOWED_SUFFIXES = {".md", ".pdf", ".docx", ".txt"}
CONFIG_ALLOWED_SUFFIXES = {".csv", ".json"}
CODE_ALLOWED_NAMES = {"README", "README.md", "read_me.md", "readme.md"}

# ── 扫描时忽略的目录(避免把产物/缓存/依赖当输入,形成自扫描递归) ──
_IGN_DIRS = {
    "__pycache__", "node_modules", ".git", ".venv", ".idea",
    ".ipynb_checkpoints", "artifact", "staticfiles", "dist", "build",
    ".pytest_cache", "model_data",
}


def file_hash(path: Path) -> str:
    """只读前 64KB + size + mtime 的轻量指纹(大文件不全量读)。"""
    stat = path.stat()
    with open(path, "rb") as fh:
        head = fh.read(65536)
    return hashlib.sha256(head + str(stat.st_size).encode() + str(stat.st_mtime).encode()).hexdigest()


def source_id_for(root: Path, path: Path) -> str:
    rel = path.relative_to(root).as_posix()
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]
    return f"src-{digest}"


def infer_source_type(path: Path) -> str:
    """判定一个文件的 source_type。

    先按 model_data/sources/source_types.json 的 path_contains 规则(语料特有,
    如 ``/chat/`` → chat)匹配;不命中则回退到通用扩展名映射。
    """
    # 规则里的 ``/目录名/`` 带前导斜杠做目录边界匹配;对相对路径补一个前导 ``/``,
    # 使位于输入根一级的目录(如 ``chat/...``)也能命中。
    path_str = "/" + path.as_posix()
    for rule in model_data.source_type_rules().get("rules", []):
        pc = rule.get("path_contains")
        if pc and pc in path_str:
            st = rule.get("source_type", "doc")
            # project 目录内进一步细分:代码→code_file,配置→config,其余→project
            if st == "project":
                if path.suffix.lower() in CODE_ALLOWED_SUFFIXES:
                    return "code_file"
                if path.suffix.lower() in CONFIG_ALLOWED_SUFFIXES:
                    return "config"
                return "project"
            return st
    # 通用扩展名 → 类型映射
    if path.suffix.lower() in CODE_ALLOWED_SUFFIXES:
        return "code_file"
    if path.name.lower().startswith(("read_me", "readme")):
        return "code_readme"
    if path.suffix.lower() in CONFIG_ALLOWED_SUFFIXES:
        return "config"
    return "doc"


def _should_skip(rel: Path) -> bool:
    """路径任一层命中忽略目录/隐藏目录即跳过。"""
    parts = rel.parts
    return any(x.startswith(".") or x in _IGN_DIRS for x in parts)


def _is_supported(rel: Path) -> bool:
    """该文件是否属于受支持的代码/文档/配置(否则进入 deferred)。"""
    return (
        rel.suffix.lower() in (CODE_ALLOWED_SUFFIXES | DOC_ALLOWED_SUFFIXES | CONFIG_ALLOWED_SUFFIXES)
        or rel.name in CODE_ALLOWED_NAMES
    )


def _iter_candidates():
    """遍历输入目录下所有非忽略文件(按路径排序,保证可复现)。"""
    base = input_path()
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(base)
        if _should_skip(rel):
            continue
        if p.name.startswith(".") or p.name in IGNORED_NAMES or p.name.startswith(IGNORED_PREFIXES):
            continue
        yield base, p


def build_source_inventory() -> list[SourceRecord]:
    """清点输入目录,产出按内容去重后的 SourceRecord 列表(仅受支持扩展名)。"""
    sources: list[SourceRecord] = []
    seen: dict[str, str] = {}
    for base, p in _iter_candidates():
        rel = p.relative_to(base)
        if not _is_supported(rel):
            continue
        h = file_hash(p)
        if h and h in seen:
            continue  # 内容去重
        if h:
            seen[h] = rel.as_posix()
        sources.append(SourceRecord(
            id=source_id_for(base, p),
            path=rel.as_posix(),
            source_type=infer_source_type(rel),
            file_type=p.suffix.lower().lstrip("."),
            title=p.stem,
            hash=h or "",
        ))
    return sources


def collect_deferred_files() -> list[Path]:
    """返回输入目录下「非受支持扩展名」的文件(被略过,留待记录/后续处理)。"""
    deferred: list[Path] = []
    for base, p in _iter_candidates():
        rel = p.relative_to(base)
        if not _is_supported(rel):
            deferred.append(p)
    return deferred


def detect_content_types() -> tuple[bool, bool]:
    """轻量探测语料是否含代码/非代码内容(不做 hash,仅看扩展名与 source_type 路由)。

    返回 (has_code, has_non_code),供编排器判定 model_data 完整性契约。
    """
    has_code = False
    has_non_code = False
    for base, p in _iter_candidates():
        rel = p.relative_to(base)
        if not _is_supported(rel):
            continue
        if infer_source_type(rel) in ("code_file", "code_readme"):
            has_code = True
        else:
            has_non_code = True
        if has_code and has_non_code:
            break
    return has_code, has_non_code
