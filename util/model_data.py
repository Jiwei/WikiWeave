"""model_data/ 统一加载器(带缓存)。

代码与数据隔离的读侧入口:过程代码用 ``load("domains/keywords")`` 之类取数据,
不再在 .py 里内嵌关键词表/标签/别名等常量。所有键相对 model_data/,省略 .json。
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from .paths import MODEL_DATA_DIR, MODEL_SCHEMA_DIR


def _strip_comments(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip_comments(x) for x in obj]
    return obj


def _merge(schema_val: Any, data_val: Any) -> Any:
    """合并 model_schema 与 model_data:dict 取并集(data 覆盖同名键),list 取非空者,标量 data 优先。"""
    if isinstance(schema_val, dict) and isinstance(data_val, dict):
        merged = dict(schema_val)
        merged.update(data_val)
        return merged
    if isinstance(schema_val, list) and isinstance(data_val, list):
        return data_val if data_val else schema_val
    return data_val if data_val is not None else schema_val


@lru_cache(maxsize=None)
def load(key: str, keep_comments: bool = False) -> Any:
    """读取合并后的 model data:model_schema(预定义)+ model_data(动态定义)。"""
    schema_path = MODEL_SCHEMA_DIR / f"{key}.json"
    data_path = MODEL_DATA_DIR / f"{key}.json"
    if not schema_path.exists() and not data_path.exists():
        raise FileNotFoundError(f"model_data 缺失: {key}")
    schema_val = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.exists() else None
    data_val = json.loads(data_path.read_text(encoding="utf-8")) if data_path.exists() else None
    merged = _merge(schema_val, data_val)
    return merged if keep_comments else _strip_comments(merged)


def exists(key: str) -> bool:
    return (MODEL_SCHEMA_DIR / f"{key}.json").exists() or (MODEL_DATA_DIR / f"{key}.json").exists()


def save(key: str, data: Any) -> None:
    path = MODEL_DATA_DIR / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    load.cache_clear()


def load_or_empty(key: str, default: Any = None) -> Any:
    """加载 model_data 文件;缺失时返回 default(而非抛错)。

    通用工具在 model_data 被清空(仅留默认实体)时,过程代码应优雅降级为空值,
    而非在 import 期就 FileNotFoundError 崩掉。
    """
    if not exists(key):
        return default
    return load(key)


# ── 四个维度的便捷取值 ──
# 知识无论形态与内容如何,都沿这四个固定维度建模(单一事实源都在 model_data/):
#   code      —— 代码理解词汇(命名键)、代码对象(entities)与代码域(code_domain)
#   domains   —— 领域分类法(主题 taxonomy)
#   ontology  —— 实体建模(实体 schema + 人物/资产类别等受控词汇)
#   sources   —— 来源发现与路由(如何找到并归类源材料)


# --- code:代码理解词汇(命名键)与代码域(code_domain) ---

def code_domains() -> dict:
    return load_or_empty("code/code_domain", {})


def naming_keys() -> dict:
    return load_or_empty("code/naming_keys", {})


# --- domains:领域分类法 ---

def domain_definitions() -> dict:
    return load_or_empty("domains/definitions", {})


def domain_keywords() -> dict:
    return load_or_empty("domains/keywords", {})


def valid_domains() -> list:
    return load_or_empty("domains/valid_domains", [])


# --- ontology:实体建模(schema + 人物/资产类别受控词汇) ---

def alias_map() -> dict:
    return load_or_empty("ontology/alias_map", {})


def honorifics() -> list:
    return load_or_empty("ontology/honorifics", [])


def nonperson_stoplist() -> list:
    return load_or_empty("ontology/nonperson_stoplist", [])


def seed_patterns() -> dict:
    """seed/group/index 元函数的实体专属模式数据(正则 + 受控词汇)。

    结构: {"artifact": {...}, "person": {...}, "project": {...}}。缺失时返回 {}。
    """
    return load_or_empty("ontology/seed_patterns", {})


def meta_functions() -> dict:
    """固定元函数词汇表(meta-functions)的先验设计声明。

    结构: {"vocabulary": [...], "meta_field_schema": {...}, "ops": {...}}。
    描述每个 op 的功能/参数/依赖/使用场景;exploration 据此决定实体如何声明 meta。
    """
    return load_or_empty("ontology/meta_functions", {})


def entity_schema() -> dict:
    """实体定义 schema 的先验声明(一个实体对象有哪些字段、类型/必填/闭集取值)。

    结构: {"entity_object": {...}, "field_entry": {...}, "field_types": [...],
           "render_schema": {...}, "render_detail_entry": {...}, "render_index_schema": {...}}。
    """
    return load_or_empty("ontology/entity_schema", {})


# --- sources:来源发现与路由 ---

def source_type_rules() -> dict:
    return load_or_empty("sources/source_types", {})


# ── 派生取值(供 prompt 模板 / 分类器使用) ──

def domain_list_text(exclude_others: bool = True) -> str:
    """逗号分隔的领域名清单,供抽取 prompt 的「仅从以下取值」子句使用。"""
    ds = [d for d in valid_domains() if not (exclude_others and d == "others")]
    return ", ".join(ds)


# ── model_data 完整性契约(供编排器门禁判断是否需先 exploration) ──

def missing_model_data(has_code: bool, has_non_code: bool) -> list[str]:
    """触发 exploration 的条件(基于合并后的 model_data):
    1. sources/source_types 必须有 rules 内容;
    2. domains/definitions 必须有内容;
    3. 非代码内容:ontology/entities 还需有除 DocSectionRecord 外的至少一种实体。
    (代码内容满足 1+2 即算 exploration 完成 —— code 下的结构已经可用。)
    """
    missing: list[str] = []
    st = load_or_empty("sources/source_types", {})
    if not st.get("rules"):
        missing.append("sources/source_types (rules 为空)")
    if not load_or_empty("domains/definitions", {}):
        missing.append("domains/definitions (为空)")
    if has_non_code:
        ents = load_or_empty("ontology/entities", {})
        if set(ents) <= {"DocSectionRecord"}:
            missing.append("ontology/entities (仅默认 DocSectionRecord)")
    return missing
