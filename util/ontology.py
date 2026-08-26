"""本体加载器:按 model_data/ontology/entities.json 动态生成 dataclass。

这是"models.py 数据化外置"的运行时支撑。过程代码照旧写::

    from knowledge_extractor.util.ontology import DocSectionRecord, to_dict

生成的类与原 indexing/models.py / phase2/models.py 的 dataclass 字段完全对齐,
因此现有函数逻辑无需改动。字段定义唯一事实源是 entities.json。
"""
from __future__ import annotations

from dataclasses import asdict, field, make_dataclass
from typing import Any, Optional

from . import model_data

_TYPE_MAP = {
    "str": str, "int": int, "float": float, "bool": bool,
    "list": list, "dict": dict, "str?": Optional[str],
}


def _field_spec(f: dict):
    name = f["name"]
    typ = _TYPE_MAP.get(f.get("type", "str"), Any)
    if "default" not in f:
        return (name, typ)
    default = f["default"]
    if isinstance(default, list):
        return (name, typ, field(default_factory=list))
    if isinstance(default, dict):
        return (name, typ, field(default_factory=dict))
    return (name, typ, field(default=default))


def _build_all() -> dict[str, type]:
    # 实体 schema 按维度拆分:各维度拥有自己的对象定义,合并为统一 dataclass 命名空间。
    #   sources/entities  —— SourceRecord(基础实体,exploration 默认产出)
    #   domains/entities  —— DomainRecord(基础实体,领域维度的默认产出)
    #   code/entities     —— CodeModuleRecord / CodeUnitRecord(基础实体,code 扫描默认产出)
    #   ontology/entities —— DocSectionRecord(基础实体)+ 发现型实体(Task/Error/...)
    spec: dict = {}
    for key in ("sources/entities", "domains/entities", "code/entities", "ontology/entities"):
        spec.update(model_data.load(key))
    classes: dict[str, type] = {}
    for cls_name, meta in spec.items():
        fields = [_field_spec(f) for f in meta["fields"]]
        cls = make_dataclass(cls_name, fields)
        cls.__doc__ = meta.get("doc", "")
        cls._layer = meta.get("layer", "")
        cls._ontology_group = meta.get("ontology_group", "")
        cls._base = meta.get("base", False)  # base=True:基础实体(默认必填);否则为发现型实体
        cls._definition = meta.get("definition", "")  # LLM 抽取提示词(含 {items}/{lang}/__DOMAINS__ 占位)
        cls._list_key = meta.get("list_key", "")      # LLM 返回 JSON 中承载实体列表的键
        # 多提示词实体(如 Artifact 的 backfill / synthesis)按需自取
        cls._definition_backfill = meta.get("definition_backfill", "")
        cls._definition_synthesis = meta.get("definition_synthesis", "")
        # 元函数声明(固定词汇表:extract/seed/group/index/synthesize,由 extraction/meta.py 执行)
        cls._meta = meta.get("meta", [])
        # 渲染声明(如何把该实体渲染成 OKF 页面,由 render/okf_markdown.py 执行)
        cls._render = meta.get("render", {})
        classes[cls_name] = cls
    return classes


_CLASSES = _build_all()
globals().update(_CLASSES)

__all__ = [*_CLASSES.keys(), "to_dict", "get", "get_optional", "all_entities", "enums",
           "build_domains"]


def get(name: str) -> type:
    return _CLASSES[name]


def get_optional(name: str) -> type | None:
    """返回实体类;未定义(如被清空)时返回 None,供过程代码优雅降级,不抛 KeyError。"""
    return _CLASSES.get(name)


def all_entities() -> dict[str, type]:
    return dict(_CLASSES)


def to_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)


def enums() -> dict:
    return model_data.load_or_empty("ontology/enums", {})


def build_domains() -> list:
    """从 model_data 构建领域记录列表(DomainRecord,label 取自 definitions.json,缺省回退名字)。"""
    defs = model_data.domain_definitions()
    domains = []
    for name in model_data.valid_domains():  # 已含 "others"
        meta = defs.get(name) if isinstance(defs, dict) else None
        label = meta.get("label", name) if isinstance(meta, dict) else name
        domains.append(DomainRecord(id=name, name=name, label=label))
    return domains

