"""通用 LLM 实体抽取器 —— 由 model_data/ontology/entities.json 的 definition + fields 驱动。

替代原先 tasks/errors/workflows/protocols 各自硬编码的抽取器:实体特定数据(提示词、输出
格式、字段 schema)全部落在 entities.json 的 definition / list_key / fields,过程代码只剩
「分组 → 一次 LLM 调用一组 → 解析 → 落盘」这套通用逻辑。
"""
from __future__ import annotations

import json
from dataclasses import fields as _dc_fields

from knowledge_extractor.util import model_data
from knowledge_extractor.util.ontology import get as _get_entity, to_dict
from knowledge_extractor.extraction.pipeline import (
    PHASE2_DIR, batch_extract, checkpoint_valid, save_checkpoint_meta,
    score_confidence, sha_id, source_label, stage, tick, write_json,
)

# 从「组」派生、而非 LLM 返回的字段(LLM 未给时回填)。
_GROUP_DERIVED = {
    "source_sections": lambda group: [s.get("id") for s in group if s.get("id")],
    "week": lambda group: group[0].get("iso_week", "") if group else "",
    "source_type": lambda group: group[0].get("source_type", "") if group else "",
}


def _build_items(groups: list[list[dict]]) -> list[dict]:
    """把每组 section/code-unit 渲染成标准文本,供 LLM 抽取。"""
    items = []
    for group in groups:
        parts = []
        for s in group:
            header = f"Section: {source_label(s.get('source_type', ''))}"
            if s.get("people"):
                header += f" | Person: {', '.join(map(str, s['people'][:5]))}"
            if s.get("iso_week"):
                header += f" | Week: {s['iso_week']}"
            elif s.get("time_range"):
                header += f" | Time: {s['time_range']}"
            parts.append(f"\n--- {header} ---")
            parts.append(f"Name: {s.get('name', '')}")
            if s.get("doc"):
                parts.append(f"Docstring: {s['doc']}")
            parts.append(f"Content: {s.get('content', '')}")
        items.append({"text": "\n".join(parts), "group": group})
    return items


def extract_entity(entity_name: str, groups: list[list[dict]]) -> list:
    """通用实体抽取:每组一次 LLM 调用,按 definition 提示词 + 字段 schema 解析为 dataclass。"""
    cls = _get_entity(entity_name)
    definition = getattr(cls, "_definition", "")
    list_key = getattr(cls, "_list_key", "")
    if not definition:
        raise ValueError(f"{entity_name} 缺 definition(需在 entities.json 中定义 LLM 提示词)")

    prompt = definition.replace("__DOMAINS__", model_data.domain_list_text())

    # ── Checkpoint ──
    ckpt_path = PHASE2_DIR / f"_phase2_ckpt_{entity_name.lower()}.json"
    src_hash = json.dumps(sorted(
        sorted(s.get("id", str(i)) for i, s in enumerate(g)) for g in groups
    ))
    if checkpoint_valid(ckpt_path, src_hash):
        tick("loading checkpoint...")
        return [cls(**r) for r in json.loads(ckpt_path.read_text())]

    items = _build_items(groups)
    tick(f"extracting {entity_name} from {len(items)} groups...")
    results = batch_extract(items, prompt, label=entity_name)

    field_names = {f.name for f in _dc_fields(cls)}
    id_field = "title" if "title" in field_names else ("name" if "name" in field_names else None)
    prefix = entity_name.replace("Record", "").lower()
    key_fields = [f for f in ("title", "name", "progress", "definition") if f in field_names]

    records = []
    for i, result in enumerate(results):
        if not result:
            continue
        group = items[i]["group"]
        extracted = result.get(list_key) if list_key else result
        if isinstance(extracted, dict):
            extracted = [extracted]
        for idx, e in enumerate(extracted or []):
            if not isinstance(e, dict):
                continue
            kw = {k: v for k, v in e.items() if k in field_names}
            # 组派生字段回填
            for field, fn in _GROUP_DERIVED.items():
                if field in field_names and not kw.get(field):
                    kw[field] = fn(group)
            try:
                record = cls(**kw)
            except TypeError:
                continue  # 缺必填字段 → 跳过该条
            if not record.id:
                key = f"{group[0].get('id', '') if group else ''}:{getattr(record, id_field, '') if id_field else ''}:{idx}"
                record.id = sha_id(prefix, key)
            record._confidence = score_confidence(to_dict(record), key_fields)
            records.append(record)

    tick(f"{len(records)} {entity_name} extracted")
    write_json(ckpt_path, [to_dict(r) for r in records])
    save_checkpoint_meta(src_hash, ckpt_path)
    return records
