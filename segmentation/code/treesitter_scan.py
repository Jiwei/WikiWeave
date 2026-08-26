"""多语言代码结构扫描(tree-sitter)—— JS/TS/Java/Go/C/C++。

与 Python 路径(stdlib ast,见 scan.py::segment_python)并列,产出同一套 CodeUnitRecord:
module / class / interface / struct / enum / function / method,并抽取 imports 与调用点
(calls),供依赖图 / 调用图构建使用。名字统一为 ``{文件stem}.{符号}``(方法再带类名前缀)。

解析器来自 tree-sitter-language-pack;若未安装则本模块的 available() 返回 False,
上层回退到「仅 Python」而不中断流水线。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from knowledge_extractor.util.ontology import CodeUnitRecord, SourceRecord
from knowledge_extractor.util.paths import ROOT

# ── 扩展名 → 语言 ──
EXT_LANG: dict[str, str] = {
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".c++": "cpp",
    ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".php": "php", ".php5": "php", ".phtml": "php",
}

SUPPORTED_EXTS = set(EXT_LANG)


def language_for(path: str | Path) -> str | None:
    return EXT_LANG.get(Path(path).suffix.lower())


# ── tree-sitter 解析器(懒加载 + 缓存);缺库则整体不可用 ──
_PARSERS: dict[str, Any] = {}
_AVAILABLE: bool | None = None


def available() -> bool:
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import tree_sitter_language_pack  # noqa: F401
            _AVAILABLE = True
        except Exception:
            _AVAILABLE = False
    return _AVAILABLE


def _parser(lang: str):
    if lang not in _PARSERS:
        from tree_sitter_language_pack import get_parser
        _PARSERS[lang] = get_parser(lang)
    return _PARSERS[lang]


# ── 每语言节点类型规格 ──
# func/method: 直接产出函数/方法单元;class/iface/struct/enum: 产出类型单元并递归其体找方法。
_JS_LIKE = {
    "func": {"function_declaration", "generator_function_declaration"},
    "method": {"method_definition"},
    "class": {"class_declaration"},
    "iface": set(),
    "struct": set(),
    "enum": set(),
    "import": {"import_statement"},
    "call": {"call_expression", "new_expression"},
}
_TS_LIKE = {
    **_JS_LIKE,
    "iface": {"interface_declaration"},
    "enum": {"enum_declaration"},
    "class": {"class_declaration", "abstract_class_declaration"},
}
SPEC: dict[str, dict[str, set]] = {
    "javascript": _JS_LIKE,
    "typescript": _TS_LIKE,
    "tsx": _TS_LIKE,
    "java": {
        "func": set(),
        "method": {"method_declaration", "constructor_declaration"},
        "class": {"class_declaration"},
        "iface": {"interface_declaration"},
        "struct": set(),
        "enum": {"enum_declaration"},
        "import": {"import_declaration"},
        "call": {"method_invocation", "object_creation_expression"},
    },
    "go": {
        "func": {"function_declaration"},
        "method": {"method_declaration"},
        "class": set(),
        "iface": set(),
        "struct": set(),
        "enum": set(),
        "import": {"import_declaration"},
        "call": {"call_expression"},
    },
    "c": {
        "func": {"function_definition"},
        "method": set(),
        "class": set(),
        "iface": set(),
        "struct": {"struct_specifier", "union_specifier"},
        "enum": {"enum_specifier"},
        "import": {"preproc_include"},
        "call": {"call_expression"},
    },
    "cpp": {
        "func": {"function_definition"},
        "method": set(),
        "class": {"class_specifier"},
        "iface": set(),
        "struct": {"struct_specifier", "union_specifier"},
        "enum": {"enum_specifier"},
        "import": {"preproc_include"},
        "call": {"call_expression", "new_expression"},
    },
    "php": {
        "func": {"function_definition"},
        "method": {"method_declaration"},
        "class": {"class_declaration", "trait_declaration"},
        "iface": {"interface_declaration"},
        "struct": set(),
        "enum": {"enum_declaration"},
        "import": {"namespace_use_declaration"},
        "call": {"function_call_expression", "member_call_expression",
                 "scoped_call_expression", "object_creation_expression"},
    },
}

# 递归进入以寻找内嵌定义的容器节点(不同语言的“体”)
_CONTAINER_TYPES = {
    "class_body", "interface_body", "enum_body", "declaration_list",
    "field_declaration_list", "namespace_definition", "class_specifier",
    "struct_specifier", "program", "source_file", "translation_unit",
    "statement_block", "object", "export_statement",
}
_COMMENT_TYPES = {"comment", "line_comment", "block_comment", "//"}
_IDENT_TYPES = {"identifier", "type_identifier", "field_identifier", "property_identifier"}


def _txt(node) -> str:
    return node.text.decode("utf-8", errors="replace") if node is not None else ""


def _find_ident(node) -> str:
    """返回子树里第一个标识符文本(用于兜底取名)。"""
    if node is None:
        return ""
    if node.type in _IDENT_TYPES:
        return _txt(node)
    for ch in node.named_children:
        r = _find_ident(ch)
        if r:
            return r
    return ""


def _c_declarator_name(node) -> str:
    """C/C++ function_definition:沿 declarator 字段下钻取函数名。"""
    d = node.child_by_field_name("declarator")
    seen = 0
    while d is not None and seen < 8:
        seen += 1
        if d.type in {"identifier", "field_identifier", "operator_name"}:
            return _txt(d)
        if d.type == "qualified_identifier":
            return _txt(d).split("::")[-1]
        nxt = d.child_by_field_name("declarator")
        if nxt is None:
            return _find_ident(d)
        d = nxt
    return ""


def _def_name(node, lang: str) -> str:
    nm = node.child_by_field_name("name")
    if nm is not None:
        return _txt(nm).split("::")[-1]
    if lang in {"c", "cpp"} and node.type == "function_definition":
        return _c_declarator_name(node)
    # Go type_spec / 兜底
    return _find_ident(node)


def _callee_name(call_node, lang: str) -> str:
    """从调用节点抽被调符号名(取最右标识符)。"""
    if lang in {"java", "php"}:
        nm = call_node.child_by_field_name("name")
        if nm is not None:
            return _txt(nm).split("\\")[-1]
        # object_creation_expression → type/class
        ty = call_node.child_by_field_name("type") or call_node.child_by_field_name("function")
        if ty is not None:
            return _txt(ty).split("\\")[-1].split("::")[-1]
    fn = call_node.child_by_field_name("function") or call_node.child_by_field_name("constructor")
    if fn is None:
        # new_expression(js): constructor field
        fn = call_node.child_by_field_name("type")
    if fn is None:
        return ""
    t = fn.type
    if t in _IDENT_TYPES:
        return _txt(fn)
    if t in {"member_expression", "selector_expression", "field_expression"}:
        prop = fn.child_by_field_name("property") or fn.child_by_field_name("field")
        if prop is not None:
            return _txt(prop)
    if t == "qualified_identifier":
        return _txt(fn).split("::")[-1]
    # 兜底:取最右标识符
    ident = ""
    for ch in fn.named_children:
        r = _find_ident(ch)
        if r:
            ident = r
    return ident or _find_ident(fn)


def _collect_calls(node, lang: str, call_types: set) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def walk(n):
        if n.type in call_types:
            name = _callee_name(n, lang)
            if name and name not in seen:
                seen.add(name)
                out.append(name)
        for ch in n.children:
            walk(ch)

    walk(node)
    return out


def _collect_imports(root, lang: str, import_types: set) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def add(v: str):
        v = v.strip().strip('"').strip("<>").strip("'")
        if v and v not in seen:
            seen.add(v)
            out.append(v)

    def walk(n):
        if n.type in import_types:
            if lang in {"javascript", "typescript", "tsx"}:
                src = n.child_by_field_name("source")
                add(_txt(src) if src is not None else _txt(n))
            elif lang == "java":
                # import_declaration → scoped_identifier
                for ch in n.named_children:
                    if ch.type in {"scoped_identifier", "identifier"}:
                        add(_txt(ch))
            elif lang == "go":
                for spec in n.named_children:
                    add(_find_ident_string(spec) or _txt(spec))
            elif lang in {"c", "cpp"}:
                for ch in n.named_children:
                    if "string" in ch.type or ch.type == "system_lib_string":
                        add(_txt(ch))
            elif lang == "php":
                # namespace_use_declaration → namespace_use_clause → qualified_name
                add(_find_qualified(n))
        for ch in n.children:
            walk(ch)

    walk(root)
    return out


def _find_ident_string(node) -> str:
    """Go import_spec:取字符串字面量。"""
    for ch in node.named_children:
        if "string" in ch.type:
            return _txt(ch)
    return ""


def _find_qualified(node) -> str:
    """PHP:在子树里找 qualified_name / name 文本(用于 use 导入)。"""
    if node is None:
        return ""
    if node.type in {"qualified_name", "name"}:
        return _txt(node)
    for ch in node.named_children:
        r = _find_qualified(ch)
        if r:
            return r
    return ""


def _leading_doc(node) -> str:
    prev = node.prev_named_sibling
    if prev is not None and prev.type in _COMMENT_TYPES:
        # 相邻(间隔 ≤1 行)才算文档注释
        if node.start_point[0] - prev.end_point[0] <= 1:
            raw = _txt(prev)
            for mark in ("/**", "*/", "///", "//", "/*", "*"):
                raw = raw.replace(mark, " ")
            return " ".join(raw.split())[:500]
    return ""


def _loc(node) -> dict:
    return {"start_line": node.start_point[0] + 1, "end_line": node.end_point[0] + 1}


def segment_treesitter(source: SourceRecord, text: str | None = None) -> tuple[list[CodeUnitRecord], dict | None]:
    """解析单个源文件 → (code_units, ast_summary)。语言不支持或解析失败返回 ([], None)。"""
    lang = language_for(source.path)
    if lang is None or not available():
        return [], None
    if text is None:
        try:
            text = (ROOT / source.path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return [], None
    try:
        tree = _parser(lang).parse(text.encode("utf-8"))
    except Exception:
        return [], None

    spec = SPEC[lang]
    root = tree.root_node
    stem = Path(source.path).stem
    imports = _collect_imports(root, lang, spec["import"])
    call_types = spec["call"]

    units: list[CodeUnitRecord] = [
        CodeUnitRecord(
            id=f"{source.id}-cu-module", source_id=source.id,
            name=stem, kind="module", language=lang,
            locator={"start_line": 1, "end_line": len(text.splitlines()) or 1},
            doc="", imports=imports,
        )
    ]
    definitions: list[dict] = []
    counter = {"i": 0}

    def new_id() -> str:
        counter["i"] += 1
        return f"{source.id}-cu-{counter['i']}"

    def emit_type(node, kind: str, qualifier: str):
        name = _def_name(node, lang)
        if not name:
            return
        full = f"{stem}.{qualifier}{name}" if qualifier else f"{stem}.{name}"
        units.append(CodeUnitRecord(
            id=new_id(), source_id=source.id, name=full, kind=kind, language=lang,
            locator=_loc(node), doc=_leading_doc(node), imports=imports,
        ))
        definitions.append({"name": full, "kind": kind, "line": node.start_point[0] + 1})
        body = node.child_by_field_name("body")
        walk(body if body is not None else node, f"{qualifier}{name}.")

    def emit_callable(node, kind: str, qualifier: str):
        name = _def_name(node, lang)
        if not name:
            return
        full = f"{stem}.{qualifier}{name}" if qualifier else f"{stem}.{name}"
        calls = _collect_calls(node, lang, call_types)
        units.append(CodeUnitRecord(
            id=new_id(), source_id=source.id, name=full,
            kind="method" if qualifier else kind, language=lang,
            locator=_loc(node), doc=_leading_doc(node), imports=imports, calls=calls,
        ))
        definitions.append({"name": full, "kind": kind, "line": node.start_point[0] + 1})

    def emit_go_type(type_decl, qualifier: str):
        for spec_node in type_decl.named_children:
            if spec_node.type != "type_spec":
                continue
            name = _find_ident(spec_node.child_by_field_name("name") or spec_node)
            body = spec_node.child_by_field_name("type")
            kind = "struct"
            if body is not None and body.type == "interface_type":
                kind = "interface"
            if not name:
                continue
            full = f"{stem}.{name}"
            units.append(CodeUnitRecord(
                id=new_id(), source_id=source.id, name=full, kind=kind, language=lang,
                locator=_loc(spec_node), doc=_leading_doc(type_decl), imports=imports,
            ))
            definitions.append({"name": full, "kind": kind, "line": spec_node.start_point[0] + 1})

    def walk(node, qualifier: str):
        if node is None:
            return
        for child in node.named_children:
            t = child.type
            if t in spec["func"] or t in spec["method"]:
                emit_callable(child, "function", qualifier)
            elif t in spec["class"]:
                emit_type(child, "class", qualifier)
            elif t in spec["iface"]:
                emit_type(child, "interface", qualifier)
            elif t in spec["struct"]:
                emit_type(child, "struct", qualifier)
            elif t in spec["enum"]:
                emit_type(child, "enum", qualifier)
            elif t == "type_declaration" and lang == "go":
                emit_go_type(child, qualifier)
            elif t in _CONTAINER_TYPES:
                walk(child, qualifier)
            else:
                # 继续下钻,兼容被包裹的定义(如 export_statement、template_declaration 等)
                walk(child, qualifier)

    walk(root, "")
    ast_summary = {"source_id": source.id, "path": source.path, "language": lang,
                   "definitions": definitions}
    return units, ast_summary
