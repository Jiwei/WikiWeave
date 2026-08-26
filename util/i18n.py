"""artifact 输出主要语言(i18n):目前仅支持英文(en,默认)与中文(zh)。

渲染代码用 t("key") 取当前语言的静态 UI 文本(标题/导航/标签);抽取出的正文内容不翻译。
"""
from __future__ import annotations

from .paths import language as current_language

# 静态 UI 文本翻译表。缺失的 key 回退到英文。
_T: dict[str, dict[str, str]] = {
    "en": {
        # 根索引 index.md
        "overview": "Overview",
        "query_guidance": "Query guidance for AI agents",
        "what_looking_for": "What are you looking for?",
        "statistics": "Statistics",
        "section_types": "Section types",
        "domains": "Domains",
        "index_title": "Knowledge Index",
        "entity_indexes": "Entity Indexes",
        "by_entity_type": "By entity type",
        "registries": "Registries",
        "by_source_type": "By source type",
        # 代码地图 code/index.md
        "naming_keys": "Naming Keys",
        "by_code_domain": "By Code Domain",
        "known_duplicates": "Known Duplicates",
        "external_repos": "External Repos (referenced but not ingested)",
        "see_also": "See also",
        "project_landscape": "Project Landscape",
        "project_import_graph": "Project Import Graph",
        "cross_project_data_flow": "Cross-Project Data Flow",
        # 代码分析 code/analysis.md
        "code_analysis_title": "Code Analysis (dependency / call graph / CFG)",
        "cyclomatic_hotspots": "Cyclomatic complexity hotspots (top 15)",
        "most_called_symbols": "Most-called internal symbols (in-degree top 15)",
        "heaviest_imports": "Files with heaviest import dependencies (out-degree top 15)",
        # 计数术语(统计区)
        "sources": "sources",
        "doc_sections": "doc sections",
        "code_units": "code units",
        "resources": "resources",
        "people": "people",
        # 表格列头
        "function": "Function",
        "language": "Language",
        "symbol": "Symbol",
        "called": "Called",
        "file": "File",
        "imports_in_repo": "Imports within repo",
        "cyclomatic": "Cyclomatic",
    },
    "zh": {
        "overview": "概览",
        "query_guidance": "AI 智能体查询指引",
        "what_looking_for": "你要找什么?",
        "statistics": "统计",
        "section_types": "分段类型",
        "domains": "领域",
        "index_title": "知识索引",
        "entity_indexes": "实体索引",
        "by_entity_type": "按实体类型",
        "registries": "注册表",
        "by_source_type": "按来源类型",
        "naming_keys": "命名键",
        "by_code_domain": "按代码域",
        "known_duplicates": "已知重复",
        "external_repos": "外部仓库(被引用但未收录)",
        "see_also": "另见",
        "project_landscape": "项目全景",
        "project_import_graph": "项目导入图",
        "cross_project_data_flow": "跨项目数据流",
        "code_analysis_title": "代码分析(依赖图 / 调用图 / 控制流)",
        "cyclomatic_hotspots": "圈复杂度热点(top 15)",
        "most_called_symbols": "被调最多的内部符号(入度 top 15)",
        "heaviest_imports": "import 依赖最重的文件(出度 top 15)",
        "sources": "来源",
        "doc_sections": "文档分段",
        "code_units": "代码单元",
        "resources": "资源",
        "people": "人物",
        "function": "函数",
        "language": "语言",
        "symbol": "符号",
        "called": "被调次数",
        "file": "文件",
        "imports_in_repo": "import 库内文件数",
        "cyclomatic": "圈复杂度",
    },
}


def t(key: str, lang: str | None = None) -> str:
    """取当前语言(或指定语言)的静态 UI 文本;缺失回退英文。"""
    lang = lang or current_language()
    table = _T.get(lang) or _T["en"]
    return table.get(key, _T["en"].get(key, key))


def prompt_language() -> str:
    """注入到生成自然语言内容的 LLM prompt 中的「输出语言」指令。"""
    return ("请用中文输出所有生成的内容(标题、摘要、定义、描述、步骤、进展)。"
            if current_language() == "zh"
            else "Write all generated content (titles, summaries, definitions, descriptions, steps, progress) in English.")
