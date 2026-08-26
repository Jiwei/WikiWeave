"""synthesize 元函数实现(固定词汇表里的「LLM 精炼实体实例」操作)。

含两种 synthesize:
  - synthesize:title  (Project)  : 为 curated projects 派生语义标题(definition 提示词)
  - synthesize        (Artifact) : LLM 后补缺失交付物 + 合成详情字段(definition_backfill /
                                   definition_synthesis 提示词)

提示词文本与占位符在 entities.json(definition 系列字段),这里只保留「组装 prompt → 调 LLM →
解析 JSON → 回填字段」的通用执行逻辑。
"""
from __future__ import annotations

import re

from knowledge_extractor.util import model_data
from knowledge_extractor.util.ontology import get as get_entity, to_dict
from knowledge_extractor.util.i18n import prompt_language
from knowledge_extractor.extraction.meta import register
from knowledge_extractor.extraction.meta_ops.seed import _add, ARTIFACT_KINDS

_GENERIC_NAMES = set(model_data.seed_patterns().get("artifact", {}).get("generic_deliverable_names", []))


def _derive_curated_titles(curated: list) -> None:
    """LLM-derive a semantic title for each curated project from its content.

    Falls back to the codename if the LLM is unavailable or the parse fails.
    `name` (codename) stays the stable identity; the LLM only fills `title`.
    """
    from knowledge_extractor.util.llm import chat, available

    if not available():
        return

    lines = []
    for p in curated:
        lines.append(
            f'- "{p.name}": task_type="{p.task_type}"; output="{p.output_positioning}"; '
            f'summary="{p.summary[:150]}"'
        )
    prompt = get_entity("ProjectRecord")._definition.format(items="\n".join(lines), lang=prompt_language())
    try:
        raw = chat(prompt)
    except Exception as e:
        print(f"  title LLM failed ({e}) — keeping codenames", flush=True)
        return

    import json as _json
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return
    try:
        titles = _json.loads(m.group())
    except Exception:
        return
    if not isinstance(titles, dict):
        return
    for p in curated:
        t = titles.get(p.name)
        if isinstance(t, str) and t.strip() and t.strip() != p.name:
            p.title = t.strip()[:80]


def _bridge_cu_ids(cu_ids: list[str], sections_by_id: dict[str, dict]) -> list[str]:
    """Map protocol cu- ids to their source file's doc sections."""
    def _src_of(ds):
        return ds["source_id"] if isinstance(ds, dict) else getattr(ds, "source_id", "")

    out = set()
    for cid in cu_ids:
        if "-sec-" in cid and cid in sections_by_id:
            out.add(cid)
            continue
        if "-cu-" in cid:
            src_prefix = cid.rsplit("-cu-", 1)[0]
            for dsid, ds in sections_by_id.items():
                if _src_of(ds) == src_prefix:
                    out.add(dsid)
    return sorted(out)


def _llm_backfill_artifacts(
    records: dict, tasks: list[dict], protocols: list[dict], sections_by_id: dict[str, dict],
) -> None:
    """LLM scan of task titles + protocol names for missed named deliverables."""
    from knowledge_extractor.util.llm import chat, available
    import json as _json

    if not available():
        return

    items = []
    for t in (tasks or [])[:120]:
        title = t.get("title", "")
        if title:
            items.append(f"- task: {title}")
    for p in (protocols or [])[:60]:
        name = p.get("name", "")
        if name:
            items.append(f"- protocol: {name}")
    if not items:
        return
    known = sorted({f"{a.kind}:{a.name}" for a in records.values()})
    prompt = get_entity("ArtifactRecord")._definition_backfill.format(
        known=", ".join(known) or "none", items="\n".join(items), lang=prompt_language())
    try:
        raw = chat(prompt)
    except Exception as e:
        print(f"  artifact LLM backfill failed ({e})", flush=True)
        return
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return
    try:
        found = _json.loads(m.group())
    except Exception:
        return
    known_names = {a.name.lower() for a in records.values()}
    for f in (found or []):
        if not isinstance(f, dict):
            continue
        name = str(f.get("name", "")).strip()
        kind = str(f.get("kind", "")).strip()
        desc = str(f.get("description", "")).strip()
        if ":" in name and name.split(":", 1)[0].strip() in ARTIFACT_KINDS:
            name = name.split(":", 1)[1].strip()
        if not name or kind not in ARTIFACT_KINDS:
            continue
        if name.lower() in _GENERIC_NAMES or name.lower() in known_names:
            continue
        secs = []
        tl = name.lower()
        for t in (tasks or []):
            if tl in (t.get("title", "") + " " + (t.get("progress") or "")).lower():
                secs.extend(t.get("source_sections", []))
        for p in (protocols or []):
            if tl in (p.get("name", "") + " " + (p.get("definition") or "")).lower():
                secs.extend(_bridge_cu_ids(p.get("source_sections", []), sections_by_id))
        if not secs:
            continue
        _add(records, kind, name, source_sections=sorted(set(secs))[:30],
             summary=desc or f"Named {kind} (LLM-detected)")
        known_names.add(name.lower())
        print(f"  artifact LLM backfill +{kind}: {name}", flush=True)


def _synthesize_details(records: list, doc_sections: list) -> None:
    """LLM-synthesize rich detail fields for each artifact from its source sections."""
    from knowledge_extractor.util.llm import chat, available
    import json as _json

    if not available():
        return
    sections_by_id = {ds.id: ds for ds in doc_sections}

    target = [r for r in records if r.source_sections]
    if not target:
        return

    for r in target:
        evidence = []
        for sid in r.source_sections[:8]:
            ds = sections_by_id.get(sid)
            if ds:
                evidence.append(f"[{ds.name}] {ds.content[:300]}")
        if not evidence:
            continue
        prompt = get_entity("ArtifactRecord")._definition_synthesis.format(
            kind=r.kind, name=r.name, evidence="\n".join(evidence), lang=prompt_language())
        try:
            raw = chat(prompt)
        except Exception:
            continue
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            continue
        try:
            parsed = _json.loads(m.group())
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        if isinstance(parsed.get("description"), str) and parsed["description"].strip():
            r.description = parsed["description"].strip()[:400]
        if isinstance(parsed.get("inputs"), list):
            r.inputs = [str(x).strip() for x in parsed["inputs"] if str(x).strip()][:8]
        if isinstance(parsed.get("outputs"), list):
            r.outputs = [str(x).strip() for x in parsed["outputs"] if str(x).strip()][:8]
        if isinstance(parsed.get("paths"), list):
            r.paths = [str(x).strip() for x in parsed["paths"] if str(x).strip()][:8]
        if isinstance(parsed.get("specs"), list):
            r.specs = [str(x).strip() for x in parsed["specs"] if str(x).strip()][:10]
        if isinstance(parsed.get("owner"), str) and parsed["owner"].strip():
            r.owner = parsed["owner"].strip()

    n = sum(1 for r in target if r.description)
    print(f"  synth: LLM-synthesized details for {n}/{len(target)} artifacts", flush=True)


def _finalize_artifacts(
    records: dict, tasks: list[dict] | None, protocols: list[dict] | None,
    doc_sections: list,
) -> list:
    """LLM 后补 + 合成详情 + 置信度评分,返回最终 artifact 列表。"""
    sections_by_id = {ds.id: ds for ds in doc_sections}
    _llm_backfill_artifacts(records, tasks or [], protocols or [], sections_by_id)
    out = list(records.values())
    _synthesize_details(out, doc_sections)

    for r in out:
        conf = 0.4
        if r.source_sections:
            conf += 0.2
        if r.related_projects:
            conf += 0.15
        if r.owner:
            conf += 0.1
        if r.description:
            conf += 0.1
        r._confidence = round(min(1.0, conf), 2)
    return out


@register("synthesize", "title")
def synthesize_project_titles(cls, params, context, entities):
    """LLM 精炼:为 curated projects 派生语义标题(definition 提示词)。"""
    projects = [get_entity("ProjectRecord")(**p) for p in entities.get("projects", [])]
    curated = [p for p in projects if p.origin == "curated"]
    _derive_curated_titles(curated)
    return [to_dict(p) for p in projects]


@register("synthesize")
def synthesize_artifacts(cls, params, context, entities):
    """LLM 后补 + 合成详情:读取 seed 产物,做 backfill + synthesis + 置信度。"""
    records: dict[tuple[str, str], "ArtifactRecord"] = {}
    for a in entities.get("artifacts", []):
        rec = get_entity("ArtifactRecord")(**a)
        records[(rec.kind, rec.name)] = rec
    out = _finalize_artifacts(records, entities.get("tasks"), entities.get("protocols"),
                              context["doc_sections"])
    return [to_dict(a) for a in out]
