"""Domain classification — keyword fallback + batched LLM (domain + people)."""
from __future__ import annotations

import json
import re

from knowledge_extractor.util import model_data
from knowledge_extractor.util.llm import LlmConfig

# ── 关键词表:单一事实源是 model_data/domains/keywords.json ──
DOMAIN_KEYWORDS: dict[str, list[str]] = model_data.domain_keywords()


def classify_keywords(text: str, min_hits: int = 1) -> list[str]:
    """Rule-based keyword classification. min_hits=1 for coverage, 2 for precision."""
    text_lower = text.lower()
    matches = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if sum(1 for kw in keywords if kw.lower() in text_lower) >= min_hits:
            matches.append(domain)
    return matches or ["others"]


BATCH_PROMPT = """Classify each of the following {count} items. For each, return a JSON object with:
- "domains": array of 1-4 domain names. Assign ALL that fit — many items span multiple domains. Available domains:

{domain_definitions}

Use "others" only if no domain fits at all.
- "people": array of person names found in the text. Use [] if none.

Return ONLY a JSON array of {count} objects, in order. No extra text.

{items}"""


# ── Definition loading (single source: domain_definitions.json) ──


def _flatten_defs(raw: dict) -> dict[str, str]:
    """Convert structured domain definitions to flat {name: text} for LLM prompt."""
    result = {}
    field_labels = {
        "concept": "Concept",
        "typical_activities": "Typical Activities",
        "not": "What it is NOT",
        "discriminating_signals": "Discriminating Signals",
    }
    for k, v in raw.items():
        if isinstance(v, dict):
            label = v.get("label", "")
            header = f"{k} — {label}" if label else k
            parts = [header]
            for field, flabel in field_labels.items():
                if v.get(field):
                    parts.append(f"{flabel}: {v[field]}")
            result[k] = "\n".join(parts)
        else:
            result[k] = v
    return result


def _load_domain_definitions() -> dict[str, str]:
    """Load domain definitions as {name: definition_text} for the LLM prompt.

    Single source of truth: model_data/domains/definitions.json (structured
    {label, concept, typical_activities, not, discriminating_signals} per domain).
    """
    raw = model_data.domain_definitions()
    if not raw:
        raise FileNotFoundError(
            "model_data/domains/definitions.json is missing — required for LLM classification."
        )
    return _flatten_defs(raw)


def classify_batch_with_llm(content_type: str, items: list, batch_size: int = 20) -> None:
    """Batch-classify: for each item sets .domains, .people in-place."""
    from knowledge_extractor.util.llm import _call_llm

    cfg = LlmConfig.from_env()
    if not cfg.available:
        for obj, text in items:
            obj.domains = classify_keywords(text)
            obj.people = []  # people come from filename backfill + LLM classification
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed

    PREDEFINED_NAMES = model_data.valid_domains()
    VALID_DOMAINS = set(PREDEFINED_NAMES)
    domain_defs = _load_domain_definitions()
    defs_text = "\n".join(f"- {name}: {defn}" for name, defn in domain_defs.items() if name in PREDEFINED_NAMES)
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    def _process(bi_batch):
        bi, batch = bi_batch
        item_descs = [f"[{idx}] {text}" for idx, (_, text) in enumerate(batch)]
        prompt = BATCH_PROMPT.format(
            count=len(batch), domain_definitions=defs_text, items="\n\n".join(item_descs),
        )
        results_list = []
        try:
            result = _call_llm(prompt, cfg)
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                for idx, (obj, _) in enumerate(batch):
                    if idx < len(parsed) and isinstance(parsed[idx], dict):
                        doms = parsed[idx].get("domains", ["others"]) or ["others"]
                        doms = [d for d in doms if d in VALID_DOMAINS] or ["others"]
                        results_list.append((idx, doms, parsed[idx].get("people", []) or []))
                    else:
                        results_list.append((idx, classify_keywords(batch[idx][1]), []))
            else:
                for idx, (_, text) in enumerate(batch):
                    results_list.append((idx, classify_keywords(text), []))
        except Exception:
            for idx, (_, text) in enumerate(batch):
                results_list.append((idx, classify_keywords(text), []))
        return bi, results_list

    from knowledge_extractor.util.paths import get_llm_settings
    settings = get_llm_settings("phase1")

    with ThreadPoolExecutor(max_workers=settings["max_workers"]) as ex:
        futures = {ex.submit(_process, (bi, batch)): bi for bi, batch in enumerate(batches)}
        for future in as_completed(futures):
            bi, results_list = future.result()
            for idx, doms, pp in results_list:
                obj = batches[bi][idx][0]
                obj.domains = doms
                if pp:
                    obj.people = pp  # only overwrite if LLM actually found people
            print(f"    batch {bi + 1}/{len(batches)}", flush=True)
