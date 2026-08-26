"""实体抽取:通用 LLM 抽取器(由 entities.json 的 definition 驱动)。

原 tasks/errors/workflows/protocols 四个硬编码抽取器已合并为 generic.extract_entity;
确定性增强(projects/artifacts)已移到 extraction/ 顶层。
"""
from extraction.entities.generic import extract_entity

__all__ = ["extract_entity"]
