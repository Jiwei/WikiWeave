"""group 元函数实现(固定词汇表里的「用子实体归并/聚合父实体」操作)。

当前唯一实现:task → project 归并(curated 匹配 + 未匹配任务族提升为 generated)。
评分权重 / min_score / margin 来自 entities.json 的 group.params(可调,无需改码);
噪音短语/停用词来自 model_data/ontology/seed_patterns.json 的 project 段。
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from util import model_data
from util.ontology import get as get_entity, to_dict
from extraction.pipeline import sha_id
from extraction.meta import register

_PROJ = model_data.seed_patterns().get("project", {})
_NON_PERSON = set(_PROJ.get("non_person_owner", []))
_FAMILY_STOPWORDS = set(_PROJ.get("family_title_stopwords", []))
_ASSET_KW = _PROJ.get("asset_kw", {})


def _asset_from_text(text: str) -> str:
    """资产类别签名(comm/fut/cb),关键词表来自 model_data/ontology/seed_patterns.json。"""
    tl = (text or "").lower()
    for ac, kws in _ASSET_KW.items():
        if any(k.lower() in tl for k in kws):
            return ac
    return "none"


def _task_project_score(task: dict, project, weights: dict) -> float:
    """Weighted multi-signal score: how strongly this task belongs to the project.

    各信号权重来自 entities.json 的 group.params.weights(可调,无需改码)。
    """
    score = 0.0
    t_person = task.get("person", "")
    t_title = (task.get("title", "") + " " + (task.get("progress") or "")).lower()
    t_domains = set(task.get("domains", []) or [])

    if t_person and t_person == project.owner:
        score += weights.get("key_match", 0)
    overlap = t_domains & set(project.domains)
    score += weights.get("domain_overlap", 0) * len(overlap)
    if project.name.lower() in t_title:
        score += weights.get("title_token", 0)
    for code in project.code_refs:
        stem = code.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        if stem and (stem in t_title or project.name.lower() in t_title):
            score += weights.get("code_stem", 0)
            break
    return score


def _default_group_weights() -> dict:
    """group 元函数的评分权重默认值(单一事实源:entities.json 的 ProjectRecord.meta)。"""
    for m in get_entity("ProjectRecord")._meta:
        if m.get("op") == "group":
            return m.get("params", {}).get("weights", {})
    return {}


def _rollup_tasks_to_projects(
    tasks: list[dict], curated: list,
    weights: dict | None = None, min_score: float = 3.5, margin: float = 1.0,
) -> tuple[list, set[str]]:
    """Match tasks to curated projects; cluster unmatched into task families.

    Assignment requires a **clear winner**: best score ≥ min_score AND best beats
    second-best by ≥ margin. Ambiguous tasks fall through to generated promotion.
    """
    weights = weights or _default_group_weights()
    assigned: set[str] = set()
    for t in tasks:
        scored = sorted(((_task_project_score(t, p, weights), p) for p in curated), key=lambda x: -x[0])
        if not scored:
            continue
        best, best_score = scored[0][1], scored[0][0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        if best_score >= min_score and (best_score - second) >= margin:
            best.related_tasks.append(t["id"])
            assigned.add(t["id"])
    return curated, assigned


def _family_title(tasks: list[dict], dom: str) -> str:
    """Deterministic title for a generated task family (domain+top keyword)."""
    words = Counter()
    for t in tasks:
        for tok in re.findall(r"[a-zA-Z0-9_.]+|[一-鿿]{2,6}", (t.get("title", "")).lower()):
            if tok not in _FAMILY_STOPWORDS:
                words[tok] += 1
    base = " ".join(w for w, _ in words.most_common(3)) or "任务族"
    return f"{base} ({dom})"


def _promote_generated_projects(
    tasks: list[dict], assigned: set[str], sections_by_id: dict[str, dict],
) -> list:
    """Cluster unmatched tasks into task families → generated ProjectRecords.

    Clusters by (primary_domain, asset-signature); person is an attribute, not the
    cluster key (the design's「工作线」).
    """
    unmapped = [t for t in tasks if t["id"] not in assigned]
    clusters: dict[tuple, list[dict]] = defaultdict(list)
    for t in unmapped:
        dom = (t.get("domains") or ["others"])[0]
        asset = _asset_from_text(t.get("title", "") + " " + (t.get("progress") or ""))
        clusters[(dom, asset)].append(t)

    generated: list = []
    for (dom, asset), cl in sorted(clusters.items()):
        member_secs = sorted({sid for t in cl for sid in t.get("source_sections", [])})
        person_counts = Counter(t.get("person", "unknown") for t in cl
                                if t.get("person") not in ("", "unknown", *_NON_PERSON))
        owner = person_counts.most_common(1)[0][0] if person_counts else ""
        title = _family_title(cl, dom)
        name = re.sub(r"[^A-Za-z0-9_]+", "_", title)[:60].strip("_") or "task_family"
        rec = get_entity("ProjectRecord")(
            id=sha_id("project", f"generated:{dom}:{asset}:{name}"),
            name=name,
            title=title,
            owner=owner,
            origin="generated",
            task_type=f"任务族（{dom}）",
            asset_classes=[asset] if asset != "none" else [],
            domains=[dom],
            related_tasks=[t["id"] for t in cl],
            source_sections=member_secs,
            summary=f"{len(cl)} 个任务聚合的任务族（{dom}）",
        )
        generated.append(rec)
    return generated


@register("group")
def group_tasks_into_projects(cls, params, context, entities):
    """子实体归并:把 tasks 归入 projects(评分权重/阈值来自 params)。"""
    curated = [get_entity("ProjectRecord")(**p) for p in entities.get("projects", [])]
    tasks = entities.get("tasks", [])
    weights = params.get("weights", {})
    min_score = params.get("min_score", 3.5)
    margin = params.get("margin", 1.0)
    sections_by_id = {ds.id: ds for ds in context["doc_sections"]}
    curated, assigned = _rollup_tasks_to_projects(tasks, curated, weights, min_score, margin)
    records = curated + _promote_generated_projects(tasks, assigned, sections_by_id)
    return [to_dict(r) for r in records]
