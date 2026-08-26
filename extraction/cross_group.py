"""Cross-section grouping for Phase 2 extraction.

Groups Phase 1 sections by rule-based seed groups before extraction,
so the LLM sees all related content for one entity in a single call.

Uses BAAI/bge-m3 for embedding (8192-token context window).
Embedding-based clustering is disabled — seed groups are used directly.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from knowledge_extractor.util.paths import KB_DIR

# ── Embedding model (lazy load, shared) ──

_embed_model = None

EMBED_MODEL_NAME = "BAAI/bge-m3"  # 8192-token context, 1024-dim, multi-lingual


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="mps")
    return _embed_model


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into numpy array."""
    if not texts:
        return np.array([])
    model = _get_embed_model()
    return np.array(model.encode(texts, show_progress_bar=False))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _section_text(section: dict) -> str:
    """Get representative text for a section: title + first 8000 chars of content.

    bge-m3 has an 8192-token context window. Chinese text averages ~1.2 chars/token
    with BGE's tokenizer, so 8000 chars ≈ 6600 tokens — safe margin.
    """
    name = section.get("name", "")[:120]
    content = section.get("content", "")[:8000]
    return f"{name} {content}"


def _mean_embedding(embeddings: list) -> np.ndarray:
    """Mean of multiple embedding vectors."""
    if len(embeddings) == 0:
        return np.array([])
    return np.mean(np.array(embeddings), axis=0)


# ── Greedy clustering ──

def _greedy_cluster(
    members: list[dict],
    embeddings: np.ndarray,
    threshold: float,
    require_domain_overlap: bool = False,
) -> list[list[int]]:
    """Cluster member indices by embedding similarity.

    Returns list of clusters, each cluster is a list of indices into members.
    """
    n = len(members)
    if n <= 1:
        return [[0]] if n == 1 else []

    # Compute pairwise similarities
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim

    # Find highest-similarity pair to seed first cluster
    max_sim = 0
    best_pair = (0, 1)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i][j] > max_sim:
                max_sim = sim_matrix[i][j]
                best_pair = (i, j)

    if max_sim < threshold:
        # No pair exceeds threshold — everything is solo
        return [[i] for i in range(n)]

    clusters = [[best_pair[0], best_pair[1]]]
    assigned = set(clusters[0])

    for i in range(n):
        if i in assigned:
            continue
        best_cluster = -1
        best_sim = 0
        for ci, cluster in enumerate(clusters):
            # Mean similarity to cluster
            sims = [sim_matrix[i][j] for j in cluster]
            mean_sim = sum(sims) / len(sims)
            if require_domain_overlap:
                member_domains = set(members[i].get("domains", []))
                cluster_domains = set()
                for j in cluster:
                    cluster_domains.update(members[j].get("domains", []))
                if not (member_domains & cluster_domains):
                    continue
            if mean_sim > best_sim:
                best_sim = mean_sim
                best_cluster = ci

        if best_sim >= threshold:
            clusters[best_cluster].append(i)
            assigned.add(i)
        else:
            clusters.append([i])
            assigned.add(i)

    return clusters


# ── TASK grouping ──

def group_for_tasks(sections: list[dict], sources: dict) -> list[list[dict]]:
    """Group sections for task extraction.

    Seed: person + ISO week from source_type=weekly_report.
    Chat sections joined deterministically by matching (person, iso_week).
    Embedding-based expansion and clustering DISABLED.
    """
    import re as _re

    weekly = [s for s in sections if s.get("source_type") == "weekly_report"]
    chat = [s for s in sections if s.get("source_type") == "chat"]

    # ── Person fallback helpers ──

    def _extract_person_from_name(name: str) -> str | None:
        """Try to extract a person name from a section name."""
        # "by何" or "by 何" pattern at end of line
        m = _re.search(r'by\s*([一-鿿]{2,4})\s*$', name)
        if m:
            return m.group(1)
        # "Name12.13-12.17 周总结" — name at start before digits or date
        m = _re.match(r'^([一-鿿]{2,3})(?:\d|[\.\-\s]*\d)', name)
        if m:
            return m.group(1)
        # "Name: " prefix (Chinese colon)
        m = _re.match(r'^([^:：]+)[：:]', name)
        if m:
            candidate = m.group(1).strip()
            if len(candidate) <= 10 and not any(
                kw in candidate for kw in ("本周", "下周", "上周", "周报", "工作", "计划", "总结")
            ):
                return candidate
        return None

    # Source-level person cache: parse person from source filename once
    _source_person_cache: dict[str, str | None] = {}

    def _person_from_source(src_id: str) -> str | None:
        if src_id in _source_person_cache:
            return _source_person_cache[src_id]
        src = sources.get(src_id, {})
        path = src.get("path", "")
        # "张三-工作周报0117.pdf" or "李四 5.28 周报.pdf"
        m = _re.match(r'^source/[^/]*/([一-鿿]{2,4})[\s\-_]', path)
        result = m.group(1) if m else None
        _source_person_cache[src_id] = result
        return result

    # ── Step 1: Rule-based seed groups ──
    seed_groups: dict[tuple, list[dict]] = defaultdict(list)
    dropped_person = 0
    dropped_week = 0

    for s in weekly:
        person = (s.get("people") or [None])[0]
        week = s.get("iso_week") or "unknown"

        # Fallback: try section name
        if not person:
            person = _extract_person_from_name(s.get("name", ""))
        # Fallback: try source filename
        if not person:
            person = _person_from_source(s.get("source_id", ""))

        if not person:
            dropped_person += 1
        if week == "unknown":
            dropped_week += 1

        if not person or week == "unknown":
            continue
        seed_groups[(person, week)].append(s)

    print(f"  Task seeds: {len(seed_groups)} groups from {sum(len(v) for v in seed_groups.values())} sections"
          f" (dropped: {dropped_person} no-person, {dropped_week} no-week)")

    # ── Step 2: Deterministic chat join by (person, iso_week) ──
    chat_joined = 0
    chat_solo = 0
    seed_ids = {s["id"] for g in seed_groups.values() for s in g}

    for s in chat:
        if s["id"] in seed_ids:
            continue  # already in a seed group (should not happen, but safe)
        person = (s.get("people") or [None])[0]
        week = s.get("iso_week") or ""

        if not person or not week:
            chat_solo += 1
            continue

        key = (person, week)
        if key in seed_groups:
            seed_groups[key].append(s)
            chat_joined += 1
        else:
            chat_solo += 1  # chat in a week with no matching weekly report

    print(f"  Chat: {chat_joined} joined to seed groups, {chat_solo} solo")

    # ── Step 3: Build groups list ──
    groups = list(seed_groups.values())

    # Solo chat sections not joined to any seed group
    joined_ids = {s["id"] for g in seed_groups.values() for s in g}
    for s in chat:
        if s["id"] not in joined_ids:
            groups.append([s])

    print(f"  → {len(groups)} task extraction groups "
          f"(avg {sum(len(g) for g in groups)/max(1,len(groups)):.1f} sections/group)")
    return groups


# ── ERROR grouping ──

def group_for_errors(sections: list[dict], sources: dict) -> list[list[dict]]:
    """Group sections for error extraction.

    Seeds from two sources:
      - Chat sections in error-prone domains, grouped by (source_file, day)
      - Weekly report sections with explicit error signals, solo groups
    Embedding-based expansion and clustering DISABLED — seed groups used directly.
    """
    error_keywords = ["FAILED", "排查debug", "排查 bug", "报错", "OOM", "异常",
                      "bug", "受阻", "失败", "超时", "权限", "oom", "oom killed"]

    # Step 1a: Chat seeds — group by source file + same day(关键词信号,与领域无关)
    chat = [s for s in sections
            if s.get("source_type") == "chat"
            and any(kw in s.get("content", "") for kw in error_keywords)]

    seed_groups: dict[tuple, list[dict]] = defaultdict(list)
    for s in chat:
        src_id = s.get("source_id", "unknown")
        time_range = s.get("time_range", "")
        day = time_range[:10] if len(time_range) >= 10 else "unknown"
        if day == "unknown":
            continue
        seed_groups[(src_id, day)].append(s)

    chat_count = len(chat)
    print(f"  Error seeds: {chat_count} from chat, ", end="")

    # Step 1b: Weekly report seeds — sections with explicit error signals
    weekly_err = [s for s in sections
                  if s.get("source_type") == "weekly_report"
                  and any(kw in s.get("content", "") for kw in error_keywords)]
    print(f"{len(weekly_err)} from weekly reports")

    # Weekly error sections — group by (person, iso_week)
    for s in weekly_err:
        person = (s.get("people") or ["unknown"])[0]
        week = s.get("iso_week") or "unknown"
        if person != "unknown" and week != "unknown":
            seed_groups[("weekly_err", person, week)].append(s)
        else:
            seed_groups[("weekly_err", s["id"])].append(s)  # fallback solo

    # Steps 2-3 (embedding expansion + clustering): DISABLED
    # See git history for full implementation.

    groups = list(seed_groups.values())

    print(f"  → {len(groups)} error extraction groups "
          f"(avg {sum(len(g) for g in groups)/max(1,len(groups)):.1f} sections/group)")
    return groups


# ── WORKFLOW grouping ──

def group_for_workflows(sections: list[dict], sources: dict) -> list[list[dict]]:
    """Group sections for workflow extraction.

    Sources (weekly_report excluded — too noisy for workflow extraction):
      - code_readme: sections with procedural keywords, grouped by source file
      - chat: sections with instructional keywords, solo groups
    Embedding-based clustering DISABLED.
    """
    procedural_keywords = ["主要脚本", "使用说明", "运行流程", "运行提醒",
                           "步骤", "处理流程", "使用方法"]
    instructional_keywords = ["先", "然后", "再", "步骤", "流程", "检查", "确认",
                              "格式", "命名", "保存到", "上传"]

    # Collect seeds
    code_readme_seeds = []
    chat_seeds = []
    for s in sections:
        src_type = s.get("source_type", "")
        content = s.get("content", "")
        name = s.get("name", "")

        if src_type == "code_readme":
            if any(kw in name or kw in content[:200] for kw in procedural_keywords):
                code_readme_seeds.append(s)
        elif src_type == "chat":
            if any(kw in content[:300] for kw in instructional_keywords):
                chat_seeds.append(s)

    # Group code_readme by source file (sections from same README → one group)
    from collections import defaultdict
    by_source = defaultdict(list)
    for s in code_readme_seeds:
        by_source[s["source_id"]].append(s)

    groups = list(by_source.values())  # multi-section groups from same README
    for s in chat_seeds:
        groups.append([s])             # solo groups for chat

    print(f"  Workflow: {len(code_readme_seeds)} code_readme seeds → {len(by_source)} groups, "
          f"{len(chat_seeds)} chat seeds → solo")
    print(f"  → {len(groups)} workflow extraction groups "
          f"(avg {sum(len(g) for g in groups)/max(1,len(groups)):.1f} sections/group)")
    return groups


# ── PROTOCOL grouping ──

def group_for_protocols(code_units: list[dict]) -> list[list[dict]]:
    """Group code units for protocol extraction.

    Group by module name. Large modules → per-module. Small → batched.
    """
    # Filter candidates: docstring or path literals
    candidates = [cu for cu in code_units
                  if cu.get("doc", "").strip() or cu.get("literals", [])
                  if ".ipynb_checkpoints" not in cu.get("source_id", "")]

    # Group by source file (keeps file-local conventions together)
    by_module = defaultdict(list)
    for cu in candidates:
        source_id = cu.get("source_id", "")
        by_module[source_id].append(cu)

    groups = []
    small_batch = []

    for module, cus in by_module.items():
        if len(cus) >= 4:
            groups.append(cus)  # large module → own call
        else:
            small_batch.extend(cus)  # collect for batching

    # Batch small ones
    batch_size = 5
    for i in range(0, len(small_batch), batch_size):
        batch = small_batch[i:i + batch_size]
        if batch:
            groups.append(batch)

    print(f"  Protocol groups: {len(by_module)} modules → "
          f"{sum(1 for g in groups if len(g) >= 4)} large + "
          f"{sum(1 for g in groups if len(g) < 4)} batched = {len(groups)} groups")
    return groups


# ── Main entry point ──

def build_extraction_groups() -> dict:
    """Build extraction groups for all four entity types.

    Returns: {"tasks": [[section...], ...], "errors": [...], ...}

    Results are checkpointed — re-running with unchanged input data returns
    cached groups instantly.
    """
    import hashlib

    ckpt_path = KB_DIR / "_phase2_ckpt_groups.json"
    meta_path = KB_DIR / "_phase2_ckpt_meta_groups.json"

    with open(KB_DIR / "doc_sections.json", encoding="utf-8") as f:
        sections = json.load(f)
    with open(KB_DIR / "sources.json", encoding="utf-8") as f:
        sources_list = json.load(f)
    sources = {s["id"]: s for s in sources_list}

    # ── Input hash for checkpointing ──
    section_ids = json.dumps(sorted(s["id"] for s in sections))
    source_ids = json.dumps(sorted(s["id"] for s in sources_list))
    code_hash = ""
    code_path = KB_DIR / "code_units.json"
    if code_path.exists():
        with open(code_path, encoding="utf-8") as f:
            code_units_raw = json.load(f)
        code_hash = json.dumps(sorted(
            cu.get("id", str(i)) for i, cu in enumerate(code_units_raw)
        ))
    src_hash = hashlib.sha1(
        (section_ids + source_ids + code_hash).encode()
    ).hexdigest()[:16]

    # ── Checkpoint hit ──
    if meta_path.exists() and ckpt_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("hash") == src_hash:
                print("══ Loading cached extraction groups ══\n")
                cached = json.loads(ckpt_path.read_text())
                print(f"  Tasks:     {len(cached['tasks'])} groups")
                print(f"  Errors:    {len(cached['errors'])} groups")
                print(f"  Workflows: {len(cached['workflows'])} groups")
                print(f"  Protocols: {len(cached['protocols'])} groups")
                return cached
        except (json.JSONDecodeError, KeyError):
            pass  # corrupt cache → rebuild

    # ── Build groups ──

    # Enrich sections with source_type from source
    source_types = {s["id"]: s["source_type"] for s in sources_list}
    for s in sections:
        s["source_type"] = source_types.get(s.get("source_id", ""), "")

    # ── Filter: skip template files ──
    import re as _re
    _template_re = _re.compile(r'模版|模板|template', _re.IGNORECASE)
    before = len(sections)
    sections = [s for s in sections
                if not _template_re.search(sources.get(s["source_id"], {}).get("path", ""))]
    if len(sections) < before:
        print(f"  Filtered out {before - len(sections)} template file sections")

    print("\n══ Building Extraction Groups ══\n")

    print("── Task groups ──")
    task_groups = group_for_tasks(sections, sources)

    print("\n── Error groups ──")
    error_groups = group_for_errors(sections, sources)

    print("\n── Workflow groups ──")
    workflow_groups = group_for_workflows(sections, sources)

    print("\n── Protocol groups ──")
    with open(KB_DIR / "code_units.json", encoding="utf-8") as f:
        code_units = json.load(f)
    protocol_groups = group_for_protocols(code_units)

    print(f"\n══ Summary ══")
    print(f"  Tasks:     {len(task_groups)} groups")
    print(f"  Errors:    {len(error_groups)} groups")
    print(f"  Workflows: {len(workflow_groups)} groups")
    print(f"  Protocols: {len(protocol_groups)} groups")

    result = {
        "tasks": task_groups,
        "errors": error_groups,
        "workflows": workflow_groups,
        "protocols": protocol_groups,
    }

    # ── Save checkpoint (full groups with section/code-unit dicts) ──
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    meta_path.write_text(json.dumps({"hash": src_hash}, ensure_ascii=False), encoding="utf-8")

    return result


if __name__ == "__main__":
    build_extraction_groups()
