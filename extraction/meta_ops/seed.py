"""seed 元函数实现(固定词汇表里的确定性「从语料发现实体实例」操作)。

含两种 seed:
  - seed:spec  (Project)  : 解析 source/project/ 规格文档 → curated projects
  - seed:regex (Artifact) : 正则 + 内容驱动 + 资源复用 → artifacts

语料专属的正则/词典已外置到 model_data/ontology/seed_patterns.json(经
model_data.seed_patterns() 读取),这里只保留「扫描 + 应用模式 + 毕业规则」的通用执行逻辑。
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from knowledge_extractor.util import model_data
from knowledge_extractor.util.ontology import DocSectionRecord, SourceRecord
from knowledge_extractor.util.ontology import get as get_entity, to_dict
from knowledge_extractor.util.paths import ROOT
from knowledge_extractor.extraction.pipeline import sha_id
from knowledge_extractor.extraction.meta import register

# ── 模式数据(单一事实源:model_data/ontology/seed_patterns.json) ──
_P = model_data.seed_patterns()
_ART = _P.get("artifact", {})

_FACTOR_SET_RE = re.compile(_ART["factor_set_pattern"]) if _ART.get("factor_set_pattern") else None
_MODEL_REG_RE = re.compile(_ART["model_registry_pattern"], re.IGNORECASE) if _ART.get("model_registry_pattern") else None
_SERVER_RE = re.compile(_ART["server_pattern"]) if _ART.get("server_pattern") else None
_HOST_PATH_RE = re.compile(_ART["host_path_pattern"]) if _ART.get("host_path_pattern") else None
_MODEL_SEEDS = set(_ART.get("model_seeds", []))
_NON_ARTIFACT_TOKEN = set(_ART.get("non_artifact_tokens", []))
_MODEL_KEYWORDS = tuple(_ART.get("model_keywords", []))
_GENERIC_TECHNIQUE = tuple(_ART.get("generic_technique_words", []))
_HARD_BLOCK_MODELS = set(_ART.get("hard_block_model_names", []))

# 闭集枚举(原 phase2/models.ARTIFACT_KINDS)现从 model_data 读取
ARTIFACT_KINDS = tuple(model_data.load_or_empty("ontology/enums", {}).get("artifact_kinds", []))

# ── 项目规格解析的通用正则(非语料专属,可留在代码) ──
_PATH_RE = re.compile(r"(/[A-Za-z0-9_./~{}-]{4,})")
_CODE_REF_RE = re.compile(r"([A-Za-z0-9_]+\.py)")
_KEYVAL_RE = re.compile(r"^[-*]\s*([^：:]+)[：:]\s*(.+?)\s*$")

def _read(path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _find_spec_file(project_dir) -> "object | None":
    """Return the spec markdown for a project dir: 任务说明文档.md or {name}.md."""
    for cand in (project_dir / "任务说明文档.md", project_dir / f"{project_dir.name}.md"):
        if cand.exists():
            return cand
    mds = sorted(project_dir.glob("*.md"))
    return mds[0] if mds else None


def _parse_info_block(text: str) -> dict:
    m = re.search(r"##\s*任务信息\s*\n(.*?)(?=\n##\s|\Z)", text, re.DOTALL)
    if not m:
        return {}
    info = {}
    for line in m.group(1).splitlines():
        kv = _KEYVAL_RE.match(line.strip())
        if kv:
            info[kv.group(1).strip()] = kv.group(2).strip()
    return info


def _section(text: str, heading: str) -> str:
    m = re.search(rf"##\s*{re.escape(heading)}\s*\n(.*?)(?=\n##\s|\Z)", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_code_refs(info: dict, text: str) -> list[str]:
    refs = set()
    if "对应代码" in info:
        refs.update(_CODE_REF_RE.findall(info["对应代码"]))
    refs.update(_CODE_REF_RE.findall(_section(text, "关键代码位置")))
    return sorted(refs)


def _extract_resource_paths(text: str) -> list[str]:
    paths = set()
    for section_name in ("输入", "输出", "任务背景"):
        for m in _PATH_RE.finditer(_section(text, section_name)):
            p = m.group(1)
            if p.count("/") >= 2:
                paths.add(p)
    return sorted(paths)


def parse_project(project_dir) -> "ProjectRecord":
    """Deterministically parse one source/project/<name>/ spec into a ProjectRecord."""
    name = project_dir.name
    spec = _find_spec_file(project_dir)
    text = _read(spec) if spec else ""
    info = _parse_info_block(text)

    owner = info.get("负责人", "")
    start = info.get("任务开始时间", "")
    end = info.get("任务结束时间", "")
    status = "completed" if re.fullmatch(r"\d{8}", end or "") else "active"
    task_type = info.get("任务类型", "")
    output_positioning = info.get("输出定位", "")

    goal = _section(text, "核心处理目标") or _section(text, "任务背景")
    summary = re.sub(r"\s+", " ", goal).strip()[:300]

    return get_entity("ProjectRecord")(
        id=sha_id("project", name),
        name=name,
        title=name,
        owner=owner,
        start_date=start,
        status=status,
        origin="curated",
        task_type=task_type,
        output_positioning=output_positioning,
        domains=[],
        code_refs=_extract_code_refs(info, text),
        resource_paths=_extract_resource_paths(text),
        related_tasks=[],
        source_sections=[],
        summary=summary,
    )


def _classify_project_domains(project, sections_by_id: dict[str, "object"]) -> None:
    """Derive a curated project's domains from its spec sections' domain labels."""
    dom_count: Counter = Counter()
    for sid in project.source_sections:
        sec = sections_by_id.get(sid)
        if sec is None:
            continue
        domains = sec.domains if hasattr(sec, "domains") else (sec.get("domains") if isinstance(sec, dict) else [])
        for d in (domains or []):
            if d != "others":
                dom_count[d] += 1
    project.domains = [d for d, _ in dom_count.most_common(3)]


# ── Artifact seed:通用「发现具名交付物」执行逻辑 ──

def _add(records: dict, kind: str, name: str, **kw):
    assert kind in ARTIFACT_KINDS, f"unknown artifact kind: {kind}"
    key = (kind, name)
    if key not in records:
        records[key] = get_entity("ArtifactRecord")(id=sha_id("artifact", f"{kind}:{name}"),
                                                    kind=kind, name=name, **kw)
    rec = records[key]
    for f in ("related_projects", "related_tasks",
              "related_code", "related_resources", "source_sections"):
        for v in kw.get(f, []):
            if v not in getattr(rec, f):
                getattr(rec, f).append(v)
    if kw.get("owner") and not rec.owner:
        rec.owner = kw["owner"]
    return rec


def _discover_candidates(doc_sections: list, resources: list, source_filenames: list) -> list[tuple[str, str, str]]:
    """Discover artifact candidates from the corpus (sections + resources + filenames).

    Returns [(kind, name, regex_pattern)]. Tokenization/counting 属通用算法,留在代码;
    受控词汇(模型种子词 / 非制品 token / 关键词)来自 model_data。
    """
    text = "\n".join(f"{ds.name} {ds.content}" for ds in doc_sections)
    candidates: dict[tuple[str, str], str] = {}

    fs_pat = re.compile(r"\b(factor_[A-Za-z0-9_]+(?:\.csv)?)\b")
    for m in fs_pat.finditer(text):
        tok = m.group(1).rstrip(".csv")
        candidates.setdefault(("factor_set", tok), rf"\b{re.escape(tok)}\b")
    for fn in source_filenames:
        for m in fs_pat.finditer(fn):
            tok = m.group(1).rstrip(".csv")
            candidates.setdefault(("factor_set", tok), rf"\b{re.escape(tok)}\b")
    for r in resources:
        rname = r.get("name", "") if isinstance(r, dict) else getattr(r, "name", "")
        for m in fs_pat.finditer(rname):
            tok = m.group(1).rstrip(".csv")
            candidates.setdefault(("factor_set", tok), rf"\b{re.escape(tok)}\b")

    srv_pat = re.compile(r"\b(server\d{2}|node\d{2})\b")
    for m in srv_pat.finditer(text):
        candidates.setdefault(("server", m.group(1)), rf"\b{re.escape(m.group(1))}\b")
    for r in resources:
        rname = r.get("name", "") if isinstance(r, dict) else getattr(r, "name", "")
        for m in srv_pat.finditer(rname):
            candidates.setdefault(("server", m.group(1)), rf"\b{re.escape(m.group(1))}\b")

    ds_pat = re.compile(r"\b\w*(?:data|dataset|shard)\b")
    for m in ds_pat.finditer(text):
        candidates.setdefault(("dataset", m.group(0)), rf"\b{re.escape(m.group(0))}\b")

    for name in _MODEL_SEEDS:
        if re.search(rf"\b{re.escape(name)}\b", text):
            candidates.setdefault(("model", name), rf"\b{re.escape(name)}\b")

    tok_counter = Counter()
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9_]{2,20})\b", text):
        w = m.group(1)
        if w not in _NON_ARTIFACT_TOKEN:
            tok_counter[w] += 1
    for w, c in tok_counter.most_common():
        if c < 3:
            break
        wl = w.lower()
        has_model_kw = any(k in wl for k in _MODEL_KEYWORDS)
        is_generic = any(g in wl for g in _GENERIC_TECHNIQUE)
        if has_model_kw and not is_generic:
            candidates.setdefault(("model", w), rf"\b{re.escape(w)}\b")

    return [(k, n, p) for (k, n), p in candidates.items()]


def _content_seed_artifacts(records: dict, doc_sections: list, resources: list,
                            source_filenames: list) -> None:
    """Scan all section content + resources + filenames for named deliverables and
    promote those that pass the graduation rule (≥3 sections OR ≥2 domains)."""
    name_secs: dict[tuple[str, str], list[str]] = {}
    name_domains: dict[tuple[str, str], set] = {}
    name_people: dict[tuple[str, str], Counter] = {}

    candidates = _discover_candidates(doc_sections, resources, source_filenames)
    for kind, name, pat in candidates:
        rx = re.compile(pat, re.IGNORECASE)
        key = (kind, name)
        for ds in doc_sections:
            text = f"{ds.name} {ds.content}"
            if not rx.search(text):
                continue
            name_secs.setdefault(key, []).append(ds.id)
            for d in (ds.domains or []):
                if d != "others":
                    name_domains.setdefault(key, set()).add(d)
            for p in (ds.people or []):
                if p and p != "unknown":
                    name_people.setdefault(key, Counter())[p] += 1

    for (kind, name) in list(name_secs.keys()):
        if kind == "model" and name.lower() in _HARD_BLOCK_MODELS:
            continue
        secs = sorted(set(name_secs[(kind, name)]))
        domains = name_domains.get((kind, name), set())
        if len(secs) >= 3 or len(domains) >= 2:
            owner = name_people.get((kind, name), Counter()).most_common(1)
            owner = owner[0][0] if owner else ""
            _add(records, kind, name, source_sections=secs[:40],
                 owner=owner, summary="")


def _link_related_entities(records: dict, doc_sections: list,
                           code_units: list, resources: list) -> None:
    """Link each artifact to its related code units and resources (name mention)."""
    for a in records.values():
        al = a.name.lower()
        for cu in code_units or []:
            cu_text = (f"{cu.get('name', '')} {' '.join(cu.get('literals', []))} "
                       f"{cu.get('doc', '')}").lower()
            if al and al in cu_text:
                if cu.get("id") not in a.related_code:
                    a.related_code.append(cu["id"])
        for r in resources or []:
            rname = (r.get("name", "") if isinstance(r, dict) else getattr(r, "name", "")).lower()
            if al and al in rname:
                rid = r.get("id") if isinstance(r, dict) else getattr(r, "id", "")
                if rid and rid not in a.related_resources:
                    a.related_resources.append(rid)


def build_seed_records(
    sources: list[SourceRecord],
    doc_sections: list[DocSectionRecord],
    projects: list | None = None,
    resources: list | None = None,
    code_units: list | None = None,
) -> dict[tuple[str, str], "ArtifactRecord"]:
    """确定性 seed:项目产出/注册表/服务器/内容驱动/资源复用 → artifacts 记录表(无 LLM)。"""
    records: dict[tuple[str, str], "ArtifactRecord"] = {}
    sections_by_source: dict[str, list[str]] = {}
    for ds in doc_sections:
        sections_by_source.setdefault(ds.source_id, []).append(ds.id)

    # Seed 1: project outputs (scripts + datasets)
    for p in projects or []:
        pname = p.get("name") if isinstance(p, dict) else getattr(p, "name", "")
        owner = p.get("owner", "") if isinstance(p, dict) else getattr(p, "owner", "")
        pid = p.get("id") if isinstance(p, dict) else getattr(p, "id", "")
        psecs = p.get("source_sections", []) if isinstance(p, dict) else getattr(p, "source_sections", [])
        codes = p.get("code_refs", []) if isinstance(p, dict) else getattr(p, "code_refs", [])
        outpos = p.get("output_positioning", "") if isinstance(p, dict) else getattr(p, "output_positioning", "")
        for c in codes:
            _add(records, "script", c, owner=owner, related_projects=[pid],
                 source_sections=list(psecs), summary=outpos)
        if outpos:
            _add(records, "dataset", pname, owner=owner, related_projects=[pid],
                 source_sections=list(psecs), summary=outpos)

    # Seed 2: registries / factor sets / label files from source CSVs
    for s in sources:
        path = s.path
        base = Path(path).name
        secs = sections_by_source.get(s.id, [])
        if _FACTOR_SET_RE and _FACTOR_SET_RE.search(base):
            _add(records, "factor_set", base[:-4], source_sections=secs,
                 summary=f"Factor column definition file `{base}`")
        elif _MODEL_REG_RE and _MODEL_REG_RE.search(base):
            _add(records, "model", base[:-4], source_sections=secs,
                 summary=f"Model registry `{base}`")
        elif base == "labels.csv":
            _add(records, "dataset", f"labels ({Path(path).parent.name})",
                 summary=f"Label (y) column list `{base}`")

    # Seed 3: servers referenced in project specs / docs
    server_text_sources = [s for s in sources if s.source_type in ("project", "doc", "code_readme")]
    for s in server_text_sources:
        try:
            text = (ROOT / s.path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if _SERVER_RE:
            for m in _SERVER_RE.finditer(text):
                _add(records, "server", m.group(1), source_sections=sections_by_source.get(s.id, []),
                     summary=f"Compute/storage host `{m.group(1)}`")

    # Seed 0: content-driven (graduation rule)
    _content_seed_artifacts(records, doc_sections, resources or [], [s.path for s in sources])

    # Seed 3b: reuse the resources pipeline (host paths → server/path provenance)
    for r in resources or []:
        rname = r.get("name", "") if isinstance(r, dict) else getattr(r, "name", "")
        rsrcs = r.get("source_ids", []) if isinstance(r, dict) else getattr(r, "source_ids", [])
        rdomains = r.get("domains", []) if isinstance(r, dict) else getattr(r, "domains", [])
        if _HOST_PATH_RE:
            for m in _HOST_PATH_RE.finditer(rname):
                host = m.group(1)
                secs = []
                for sid in rsrcs:
                    secs.extend(sections_by_source.get(sid, []))
                _add(records, "server", host, source_sections=sorted(set(secs))[:40],
                     summary="")

    _link_related_entities(records, doc_sections, code_units or [], resources or [])
    return records


# ── 元函数注册(由 entities.json 的 meta 声明派发) ──

@register("seed", "spec")
def seed_projects(cls, params, context, entities):
    """确定性 seed:解析 source/project/ 规格文档 → curated projects(不含 rollup)。"""
    sources = context["sources"]
    doc_sections = context["doc_sections"]
    sections_by_id = {ds.id: ds for ds in doc_sections}
    sections_by_source: dict[str, list[str]] = {}
    for ds in doc_sections:
        sections_by_source.setdefault(ds.source_id, []).append(ds.id)
    source_by_path = {s.path: s for s in sources}
    project_root = ROOT / "source" / "project"
    curated: list = []
    project_dirs = (sorted(d for d in project_root.iterdir() if d.is_dir() and not d.name.startswith("."))
                    if project_root.exists() else [])
    for pdir in project_dirs:
        rec = parse_project(pdir)
        spec = _find_spec_file(pdir)
        if spec is not None:
            rel = spec.relative_to(ROOT).as_posix()
            src = source_by_path.get(rel)
            if src:
                rec.source_sections = sections_by_source.get(src.id, [])
        _classify_project_domains(rec, sections_by_id)
        curated.append(rec)
    return [to_dict(p) for p in curated]


@register("seed", "resource")
def seed_resources(cls, params, context, entities):
    """确定性 seed:扫描 section 文本 → 资源引用(ResourceRecord)。"""
    from knowledge_extractor.extraction.literals import extract_resources
    return [to_dict(r) for r in extract_resources(context["sources"], context["doc_sections"])]


@register("seed", "regex")
def seed_artifacts(cls, params, context, entities):
    """确定性 seed:regex 模式 + 内容驱动 + 资源复用 → artifacts 记录表(无 LLM)。"""
    projects = [get_entity("ProjectRecord")(**p) for p in entities.get("projects", [])]
    records = build_seed_records(
        context["sources"], context["doc_sections"], projects,
        [to_dict(r) for r in context.get("resources", [])],
        context.get("code_units", []),
    )
    return [to_dict(a) for a in records.values()]
