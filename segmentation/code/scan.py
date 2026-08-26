"""Code-aware structural scan — produces CodeUnitRecord, AST trees, links code to docs."""
from __future__ import annotations

import ast
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from util.ontology import CodeModuleRecord, CodeUnitRecord, DocSectionRecord, SourceRecord
from util.paths import ROOT


# Generic function/file names that do NOT reliably indicate "this code is mentioned".
# These are too common to serve as a code↔doc link signal (e.g. `main` appears in
# 166 code units and as a substring in nearly every doc section).
_GENERIC_NAMES = {
    "main", "run", "get", "set", "read", "write", "process", "update", "clean",
    "split", "merge", "check", "parse", "load", "save", "convert", "map",
    "filter", "init", "start", "stop", "open", "close", "help", "test",
    "compute", "calculate", "handle", "execute", "validate", "normalize", "sort",
    "group", "join", "create", "delete", "list", "find", "search", "gen",
    "build", "apply", "transform", "extract", "remove", "add", "push", "pop",
    "call", "dump", "print", "log", "data", "info", "config", "setup", "utils",
    "isdir", "is_exists", "get_mode", "get_dates", "write_bin", "safe_makedirs",
    "run_cmd", "parse_args", "get_files", "handle_func",
    "teardown", "read_file", "write_file", "open_file",
}


def _sid(prefix: str, key: str) -> str:
    return f"{prefix}-{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def _ast_to_dict(node: ast.AST) -> dict[str, Any]:
    """Serialize an AST node tree to a JSON-compatible dict."""
    out: dict[str, Any] = {"_type": node.__class__.__name__}
    for field_name, value in ast.iter_fields(node):
        if field_name == "ctx":
            continue  # Load/Store context — noise for indexing
        if isinstance(value, ast.AST):
            out[field_name] = _ast_to_dict(value)
        elif isinstance(value, list):
            items = []
            for item in value:
                if isinstance(item, ast.AST):
                    items.append(_ast_to_dict(item))
                elif isinstance(item, str):
                    items.append(item)
            out[field_name] = items
        elif isinstance(value, bytes):
            out[field_name] = value.decode("utf-8", errors="replace")
        elif isinstance(value, (str, int, float, bool)):
            out[field_name] = value
        elif value is not None:
            # Ellipsis(``...`` 字面量)/complex 等非 JSON 原生常量值 → 转字符串,避免序列化崩溃
            out[field_name] = str(value)
    # add line numbers if present
    if hasattr(node, "lineno"):
        out["lineno"] = node.lineno
    if hasattr(node, "end_lineno"):
        out["end_lineno"] = getattr(node, "end_lineno", None)
    return out


def scan_codebase(sources: list[SourceRecord]) -> tuple[list[CodeUnitRecord], list[CodeModuleRecord], list[dict[str, Any]]]:
    """结构化代码扫描,按语言分派:.py→stdlib ast;其余支持语言→tree-sitter。

    返回 (code_units, code_modules, asts)。CodeModule 是文件级建模对象(每个代码文件一份,
    承载文件级元数据/import),由 module 级 unit 提炼;CodeUnit 仍是单元级(含 kind=module 的
    模块级单元)——二者有重叠但用途不同(见 model_data/code/entities.json)。
    """
    from segmentation.code import treesitter_scan as ts

    units: list[CodeUnitRecord] = []
    modules: list[CodeModuleRecord] = []
    asts: list[dict[str, Any]] = []
    for src in sources:
        if src.source_type != "code_file":
            continue
        if src.path.endswith(".py"):
            u, a = segment_python(src)
            lang = "python"
        else:
            lang = ts.language_for(src.path)
            if lang is None:
                continue
            u, a = ts.segment_treesitter(src)
        units.extend(u)
        if a:
            asts.append(a)
        # 文件级 CodeModule:从 module 级 unit(若有)提炼文件级元数据
        mu = next((x for x in u if x.kind == "module"), None)
        modules.append(CodeModuleRecord(
            id=f"{src.id}-cm",
            source_id=src.id,
            path=src.path,
            language=lang,
            doc=mu.doc if mu else "",
            imports=mu.imports if mu else [],
        ))
    return units, modules, asts


def segment_python(source: SourceRecord) -> tuple[list[CodeUnitRecord], dict[str, Any] | None]:
    path = ROOT / source.path
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(text)
    except Exception:
        return [], None

    ast_dict = {"source_id": source.id, "path": source.path, "tree": _ast_to_dict(tree)}

    units: list[CodeUnitRecord] = [
        # Every file is a module: its own top-level code (docstring, imports,
        # constants, __main__) is a first-class unit, alongside its functions/classes.
        CodeUnitRecord(
            id=f"{source.id}-cu-module", source_id=source.id,
            name=path.stem, kind="module",
            locator={"start_line": 1, "end_line": len(text.splitlines())},
            doc=ast.get_docstring(tree) or "",
            imports=_extract_imports(tree),
        )
    ]
    for idx, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            units.append(CodeUnitRecord(
                id=f"{source.id}-cu-{idx}", source_id=source.id,
                name=f"{path.stem}.{node.name}", kind="function",
                locator={"start_line": node.lineno, "end_line": getattr(node, "end_lineno", node.lineno)},
                doc=ast.get_docstring(node) or "",
                imports=_extract_imports(tree),
                literals=_extract_literals(node),
            ))
        elif isinstance(node, ast.ClassDef):
            units.append(CodeUnitRecord(
                id=f"{source.id}-cu-{idx}", source_id=source.id,
                name=f"{path.stem}.{node.name}", kind="class",
                locator={"start_line": node.lineno, "end_line": getattr(node, "end_lineno", node.lineno)},
                doc=ast.get_docstring(node) or "",
                imports=_extract_imports(tree),
                literals=_extract_literals(node),
            ))
    return units, ast_dict


def _extract_imports(tree: ast.AST) -> list[str]:
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.append(f"{node.module or ''}:{','.join(a.name for a in node.names)}")
    return imports


def _extract_literals(node: ast.AST) -> list[str]:
    vals: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            s = child.value
            if s.startswith("/") or s.endswith((".csv", ".json", ".py", ".md")):
                vals.append(s)
    return vals


def _read_section_content(section: DocSectionRecord, source_by_id: dict[str, SourceRecord]) -> str:
    # Prefer the section's already-extracted content: it is scoped correctly for
    # chat `messages` locators (which carry no start_line/end_line — reading the file
    # would return the whole chat and link every section to every mentioned code unit).
    if section.content:
        return section.content
    src = source_by_id.get(section.source_id)
    if not src:
        return section.name
    try:
        text = (ROOT / src.path).read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        s = section.locator.get("start_line", 1)
        e = section.locator.get("end_line", len(lines))
        return "\n".join(lines[s - 1 : e])
    except Exception:
        return section.name


def link_code_to_docs(doc_sections: list[DocSectionRecord], code_units: list[CodeUnitRecord],
                      source_by_id: dict[str, SourceRecord]) -> None:
    """Two-signal code↔doc linking: path overlap + name mention in section content.

    Signal 1 (0.9): a code unit's literal path (HDFS/file ref from its source) appears
        verbatim in a doc section's content — strong evidence the section discusses
        that code's data/artifacts.
    Signal 2 (0.5): a code unit's DISTINCTIVE name (source file stem or function name)
        is mentioned in a doc section's content — word-boundary match, generic names
        (main/run/get/set/utils/…) excluded to avoid path-paste noise.

    Both signals write section.code_mentions and code_unit.doc_mentions symmetrically.
    """
    # Reset doc_mentions so this run recomputes links from scratch (idempotent,
    # avoids stale accumulation across checkpoint-restore runs).
    for cu in code_units:
        cu.doc_mentions = []

    # index code units by source for quick lookup
    cu_by_id: dict[str, CodeUnitRecord] = {cu.id: cu for cu in code_units}

    # Signal 1: literal path → code unit ids. Only deep full paths (≥4 path segments)
    # are distinctive; base-path prefixes and format-string fragments like
    # "/ext1/user/" or "/ext1/user/hpc_" are not a meaningful code↔doc signal.
    literal_to_cus: dict[str, list[str]] = defaultdict(list)
    for cu in code_units:
        for lit in cu.literals:
            if len(lit) >= 8 and lit.startswith("/") and lit.count("/") >= 4:
                literal_to_cus[lit].append(cu.id)

    # Signal 2: distinctive stem → code unit ids
    cu_by_stem: dict[str, list[str]] = defaultdict(list)
    for cu in code_units:
        src = source_by_id.get(cu.source_id)
        src_path = src.path if src else ""
        stems = set()
        if cu.kind == "module":
            # The source file stem (e.g. "process_v1", "stat_v1") is the module's
            # own name — link it once per file, not accumulated across every unit.
            fstem = Path(src_path).stem.lower()
            if len(fstem) >= 5 and fstem not in _GENERIC_NAMES:
                stems.add(fstem)
        else:
            # function/class short name — must look like a code identifier (snake_case,
            # camelCase, or contains a digit), not a bare English word (price/volume/main).
            short = cu.name.rsplit(".", 1)[-1].lower()
            if (len(short) >= 5 and short not in _GENERIC_NAMES
                    and ("_" in short or any(c.isdigit() for c in short)
                         or any(c.isupper() for c in short))):
                stems.add(short)
        for stem in stems:
            cu_by_stem[stem].append(cu.id)

    # Drop non-distinctive signals: a literal/stem shared by many code units is a
    # duplicate-code artifact (base/ copies, checkpoint copies), not a real mention.
    _MAX_SHARED = 5
    literal_to_cus = {lit: ids for lit, ids in literal_to_cus.items() if len(ids) <= _MAX_SHARED}
    cu_by_stem = {stem: ids for stem, ids in cu_by_stem.items() if len(ids) <= _MAX_SHARED}

    for section in doc_sections:
        content = _read_section_content(section, source_by_id)
        content_lower = content.lower()
        matched: dict[str, float] = {}

        # Signal 1: a code unit's literal path appears verbatim in the section content
        for lit, cu_ids in literal_to_cus.items():
            if lit in content:
                for cu_id in cu_ids:
                    matched[cu_id] = max(matched.get(cu_id, 0), 0.9)

        # Signal 2: distinctive stem mentioned (word-boundary match)
        for stem, cu_ids in cu_by_stem.items():
            if re.search(rf"(?<![a-z0-9_]){re.escape(stem)}(?![a-z0-9_])", content_lower):
                for cu_id in cu_ids:
                    if cu_id not in matched:
                        matched[cu_id] = 0.5

        # write links symmetrically
        section.code_mentions = [f"{cu_id}:{conf:.1f}" for cu_id, conf in matched.items()]
        for cu_id, conf in matched.items():
            cu = cu_by_id.get(cu_id)
            if cu is not None:
                mention = f"{section.id}:{conf:.1f}"
                if mention not in cu.doc_mentions:
                    cu.doc_mentions.append(mention)
