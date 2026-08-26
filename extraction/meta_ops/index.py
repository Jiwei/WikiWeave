"""index 元函数实现(固定词汇表里的「聚合跨源提及为实体实例」操作)。

当前唯一实现:人物索引(chat 说话人 / 文件名署名 / 章节元数据 / task people 合并去重 +
置信度分桶)。识别人名的正则/敬称来自 model_data/ontology/seed_patterns.json 的 person 段;
非人名停用词来自 model_data/ontology/nonperson_stoplist.json。
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from knowledge_extractor.extraction.normalize import (
    _has_surname, _is_valid_name, _strip_honorific,
    build_name_pool_from_sections, normalize_person_field,
)
from knowledge_extractor.extraction.alias_map import load_alias_map
from knowledge_extractor.util import model_data
from knowledge_extractor.util.ontology import get as get_entity, to_dict
from knowledge_extractor.extraction.meta import register

_PERSON = model_data.seed_patterns().get("person", {})
_MSG_RE = re.compile(_PERSON["chat_message_pattern"]) if _PERSON.get("chat_message_pattern") else None
_FILENAME_NAME_RE = re.compile(_PERSON["filename_name_pattern"]) if _PERSON.get("filename_name_pattern") else None
_HONORIFIC_ONLY = set(_PERSON.get("honorific_only", []))
_NON_NAME = set(model_data.nonperson_stoplist())

_alias_cache: dict[str, str] | None = None


def _get_alias_map() -> dict[str, str]:
    global _alias_cache
    if _alias_cache is None:
        _alias_cache = load_alias_map()
    return _alias_cache


def _collect_chat_speakers(sources: list[dict], root: Path) -> dict[str, int]:
    """Extract speakers from chat message headers. Highly reliable."""
    speakers = Counter()
    for src in sources:
        if src.get("source_type") != "chat":
            continue
        path = root / src["path"]
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if _MSG_RE:
                m = _MSG_RE.match(line)
                if m:
                    speakers[m.group(1).strip()] += 1
    return dict(speakers)


def _collect_filename_people(sources: list[dict]) -> dict[str, int]:
    """Extract people from source filenames. Reliable for dedicated reports."""
    people = Counter()
    for src in sources:
        fname = Path(src["path"]).stem
        if _FILENAME_NAME_RE:
            m = _FILENAME_NAME_RE.match(fname)
            if m:
                name = m.group(1)
                if name not in _NON_NAME:
                    people[name] += 1
    return dict(people)


def _collect_section_people(sections: list[dict], name_pool: set[str] | None = None) -> dict[str, int]:
    """Extract people from section metadata, normalizing against a name pool."""
    if name_pool is None:
        name_pool = set()

    people = Counter()
    for sec in sections:
        for p in sec.get("people", []):
            if p in _NON_NAME or len(p) < 2:
                continue
            normalized = normalize_person_field(p, name_pool if name_pool else None)
            for n in normalized:
                if n not in _NON_NAME and _is_valid_name(n):
                    people[n] += 1
    return dict(people)


def _collect_task_people(tasks: list[dict]) -> dict[str, int]:
    """Extract people from task records."""
    people = Counter()
    for t in tasks:
        for p in t.get("people", []):
            if p not in _NON_NAME and len(p) >= 2:
                people[p] += 1
    return dict(people)


def build_people_index(
    sections: list[dict],
    sources: list[dict],
    tasks: list[dict] | None = None,
    root: Path | None = None,
) -> list[dict]:
    """Build a comprehensive, deduped, confidence-bucketed person index.

    Returns list of person dicts: name / aliases / confidence(HIGH|MEDIUM|LOW) /
    sources / count_chat / count_files / count_sections / count_tasks.
    """
    if root is None:
        root = Path(".")

    chat_names = _collect_chat_speakers(sources, root)
    file_names = _collect_filename_people(sources)
    name_pool = build_name_pool_from_sections(sections, sources)
    section_names = _collect_section_people(sections, name_pool)
    task_names = _collect_task_people(tasks) if tasks else {}

    merged: dict[str, dict] = {}

    def _canonical(name: str) -> str:
        amap = _get_alias_map()
        if name in amap:
            return amap[name]
        return _strip_honorific(name)

    def _add_occurrence(name: str, source: str, count: int):
        cname = _canonical(name)
        if cname in _NON_NAME or len(cname) < 2:
            return
        if cname in _HONORIFIC_ONLY:
            return
        if cname not in merged:
            merged[cname] = {
                "name": cname, "aliases": set(), "sources": set(),
                "count_sections": 0, "count_tasks": 0, "count_chat": 0, "count_files": 0,
            }
        if name != cname:
            merged[cname]["aliases"].add(name)
        merged[cname]["sources"].add(source)
        if source == "chat":
            merged[cname]["count_chat"] += count
        elif source == "filename":
            merged[cname]["count_files"] += count
        elif source == "section":
            merged[cname]["count_sections"] += count
        elif source == "task":
            merged[cname]["count_tasks"] += count

    for name, count in chat_names.items():
        _add_occurrence(name, "chat", count)
    for name, count in file_names.items():
        _add_occurrence(name, "filename", count)
    for name, count in section_names.items():
        _add_occurrence(name, "section", count)
    for name, count in task_names.items():
        _add_occurrence(name, "task", count)

    for cname, info in merged.items():
        chat_count = info["count_chat"]
        task_count = info["count_tasks"]
        has_filename = "filename" in info["sources"]
        has_surname_flag = _has_surname(cname)
        if chat_count > 0 or has_filename or task_count > 20:
            info["confidence"] = "HIGH"
        elif has_surname_flag and (task_count >= 5 or info["count_sections"] >= 5):
            info["confidence"] = "MEDIUM"
        elif task_count >= 10 and has_surname_flag:
            info["confidence"] = "MEDIUM"
        else:
            info["confidence"] = "LOW"

    result = sorted(merged.values(), key=lambda x: (
        {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[x["confidence"]],
        -(x["count_tasks"] + x["count_sections"] + x["count_chat"]),
    ))
    for r in result:
        r["aliases"] = sorted(r["aliases"])
        r["sources"] = sorted(r["sources"])
    return result


@register("index")
def index_people(cls, params, context, entities):
    """聚合跨源提及(chat/文件名/章节/task)为 Person 实例,按来源置信分桶。"""
    people_dicts = build_people_index(
        [to_dict(d) for d in context["doc_sections"]],
        [to_dict(s) for s in context["sources"]],
        entities.get("tasks"),
        root=context["root"],
    )
    PersonRecord = get_entity("PersonRecord")
    people = [PersonRecord(
        name=p["name"], aliases=p["aliases"], confidence=p["confidence"],
        sources=p["sources"], count_chat=p["count_chat"],
        count_files=p["count_files"], count_sections=p["count_sections"],
        count_tasks=p["count_tasks"],
    ) for p in people_dicts]
    for pr in people:
        pdomains = Counter()
        for ds in context["doc_sections"]:
            if pr.name in (ds.people or []):
                for d in (ds.domains or []):
                    if d != "others":
                        pdomains[d] += 1
        pr.domains = [d for d, _ in pdomains.most_common(5)]
    return [to_dict(p) for p in people]
