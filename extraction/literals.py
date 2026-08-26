"""Literal extraction: resources (paths/files) from text materials.

人物发现由 extraction/people_index.py 负责(从内容 + alias_map 运行时派生),
不再在此做「已知人物种子」的字面量匹配。
"""
from __future__ import annotations

import hashlib
import re

from knowledge_extractor.util.ontology import DocSectionRecord, SourceRecord, get_optional
from knowledge_extractor.util.paths import ROOT

PATH_RE = re.compile(r"(/[-A-Za-z0-9_./~]{4,})")
FILE_REF_RE = re.compile(r"([A-Za-z0-9_./-]+\.(?:py|md|csv|parquet|zst|json|pdf|docx))")
MIN_PATH_SEGMENTS = 2


def _sid(prefix: str, key: str) -> str:
    return f"{prefix}-{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def _read_section_text(source: SourceRecord, section: DocSectionRecord, cache: dict[str, str]) -> str:
    key = source.path
    if key not in cache:
        try:
            from knowledge_extractor.segmentation.doc.sections import _read_text
            cache[key] = _read_text(ROOT / source.path)
        except Exception:
            cache[key] = ""
    lines = cache[key].splitlines()
    s = section.locator.get("start_line", 1)
    e = section.locator.get("end_line", len(lines))
    return "\n".join(lines[s - 1 : e])


def extract_resources(
    sources: list[SourceRecord], doc_sections: list[DocSectionRecord],
) -> list[ResourceRecord]:
    """Extract resource references (HDFS/local paths, file references) from section content."""
    ResourceRecord = get_optional("ResourceRecord")
    if ResourceRecord is None:
        return []  # 未定义(被清空)→ 跳过资源抽取
    source_by_id = {s.id: s for s in sources}
    text_cache: dict[str, str] = {}
    resources: dict[str, ResourceRecord] = {}

    for section in doc_sections:
        src = source_by_id.get(section.source_id)
        if not src:
            continue
        text = _read_section_text(src, section, text_cache)

        for match in PATH_RE.finditer(text):
            val = match.group(1)
            if val.count("/") < MIN_PATH_SEGMENTS:
                continue
            rid = _sid("res", val)
            kind = "hdfs_path" if any(val.startswith(p) for p in ("/user/", "/ext", "/mnt", "/cache")) else "local_path"
            if rid not in resources:
                resources[rid] = ResourceRecord(id=rid, name=val, kind=kind)
            if src.id not in resources[rid].source_ids:
                resources[rid].source_ids.append(src.id)

        for match in FILE_REF_RE.finditer(text):
            val = match.group(1)
            rid = _sid("res", val)
            if rid not in resources:
                resources[rid] = ResourceRecord(id=rid, name=val, kind="file_ref")
            if src.id not in resources[rid].source_ids:
                resources[rid].source_ids.append(src.id)

    return list(resources.values())
