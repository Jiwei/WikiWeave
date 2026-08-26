"""Phase 2 shared pipeline — LLM calling, batching, checkpoint, confidence scoring.

Reuses util.llm for API configuration.
"""

from __future__ import annotations

import hashlib
import json
import re
import time as _t
from pathlib import Path
from typing import Any

from util.i18n import prompt_language
from util.llm import chat, phase_settings, available
from util.paths import KB_DIR

# ── Output paths ──
PHASE2_DIR = KB_DIR
PHASE2_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA_VERSION = "0.2.0"

# ── Human-readable source type labels ──

_SOURCE_LABELS = {
    "weekly_report": "weekly report",
    "chat": "chat message",
    "code_readme": "code documentation",
    "code_file": "code",
    "doc": "document",
    "config": "config file",
}


def source_label(source_type: str) -> str:
    """Human-readable label for a source type."""
    return _SOURCE_LABELS.get(source_type, source_type)


_last_tick = _t.time()
_stage_start = _t.time()


def tick(label: str) -> None:
    now = _t.time()
    print(f"  [{now - _stage_start:5.1f}s] {label}", flush=True)


def stage(label: str) -> None:
    global _stage_start
    _stage_start = _t.time()
    print(f"\n── {label} ──", flush=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha_id(prefix: str, key: str) -> str:
    return f"{prefix}-{hashlib.sha1(key.encode()).hexdigest()[:12]}"


# ── Batch LLM extraction ──

def batch_extract(
    items: list[dict],
    prompt_template: str,
    max_workers: int = 0,
    label: str = "extract",
) -> list[dict]:
    """Batch process items through LLM extraction. Returns list of parsed JSON results.

    Each item dict should have a 'text' key with the content to process.
    The prompt_template should be a Python format string with {items} placeholder.
    Results are parsed as JSON array matching input order.
    Items with parse failures fall back to None (caller handles filtering).
    """
    if not available():
        print(f"  LLM unavailable — skipping {label}", flush=True)
        return [None] * len(items)

    settings = phase_settings("extract")
    bs = 1  # Phase 2: one group per LLM call (deeper task, per-group JSON format)
    workers = settings["max_workers"] if max_workers == 0 else max_workers

    batches = [items[i:i + bs] for i in range(0, len(items), bs)]
    results: list[Any] = [None] * len(items)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _process(bi_batch):
        bi, batch = bi_batch
        item_descs = [f"=== GROUP {idx} ===\n{item.get('text', '')}" for idx, item in enumerate(batch)]
        prompt = prompt_template.format(
            count=len(batch),
            items="\n\n---\n\n".join(item_descs),
            lang=prompt_language(),
        )
        batch_results = []
        try:
            result = chat(prompt)

            # Strip markdown code fences
            cleaned = result
            m = re.search(r"```(?:json)?\s*\n?(.*?)```", result, re.DOTALL)
            if m:
                cleaned = m.group(1).strip()

            parsed = None
            parse_ok = False

            # Try 1: full response as JSON (handles both {...} and [...])
            try:
                parsed = json.loads(cleaned)
                parse_ok = True
            except json.JSONDecodeError:
                pass

            # Try 2: regex-extract JSON array from response
            if not parse_ok:
                match = re.search(r"\[.*\]", cleaned, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group())
                        parse_ok = True
                    except json.JSONDecodeError:
                        pass

            # Try 3: regex-extract JSON object from response
            if not parse_ok:
                match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group())
                        parse_ok = True
                    except json.JSONDecodeError:
                        pass

            if parse_ok and parsed is not None:
                if isinstance(parsed, list):
                    # Array format: one element per group
                    for idx in range(len(batch)):
                        if idx < len(parsed) and isinstance(parsed[idx], dict):
                            batch_results.append((bi * bs + idx, parsed[idx]))
                        else:
                            batch_results.append((bi * bs + idx, None))
                elif isinstance(parsed, dict):
                    # Single-object format: assign to first group (batch_size=1)
                    batch_results.append((bi * bs, parsed))
                    for idx in range(1, len(batch)):
                        batch_results.append((bi * bs + idx, None))
                else:
                    for idx in range(len(batch)):
                        batch_results.append((bi * bs + idx, None))
            else:
                for idx in range(len(batch)):
                    batch_results.append((bi * bs + idx, None))
        except Exception as e:
            print(f"    batch {bi + 1} error: {e}", flush=True)
            for idx in range(len(batch)):
                batch_results.append((bi * bs + idx, None))
        return bi, batch_results

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_process, (bi, batch)): bi for bi, batch in enumerate(batches)}
        for future in as_completed(futures):
            bi, batch_results = future.result()
            for global_idx, result in batch_results:
                results[global_idx] = result
            print(f"    {label} batch {bi + 1}/{len(batches)}", flush=True)

    return results


# ── Confidence scoring ──

def score_confidence(record: dict, required_fields: list[str]) -> float:
    """Score extraction confidence 0.0–1.0 based on field presence and quality."""
    score = 0.0
    field_count = len(required_fields)
    if field_count == 0:
        return 1.0

    for field in required_fields:
        val = record.get(field)
        if val is not None and val != "" and val != []:
            score += 1.0 / field_count

    # Bonus: title is concise (suggests quality extraction)
    title = record.get("title", "") or record.get("name", "")
    if title and len(title) < 100:
        score = min(1.0, score + 0.1)

    # Penalty: very short journal/definition (suggests surface-level extraction)
    journal = record.get("progress", "") or record.get("definition", "")
    if journal and len(journal) < 20:
        score = max(0.1, score - 0.3)

    return round(score, 2)


# ── Checkpoint ──

def checkpoint_valid(path: Path, src_hash: str) -> bool:
    # Per-extractor meta: _phase2_ckpt_meta_{type}.json
    meta_name = path.stem.replace("_phase2_ckpt_", "_phase2_ckpt_meta_")
    meta = PHASE2_DIR / f"{meta_name}.json"
    if not meta.exists():
        return False
    if not path.exists():
        return False
    data = json.loads(meta.read_text())
    return data.get("hash") == src_hash and data.get("schema_version") == SCHEMA_VERSION


def save_checkpoint_meta(src_hash: str, ckpt_path: Path) -> None:
    meta_name = ckpt_path.stem.replace("_phase2_ckpt_", "_phase2_ckpt_meta_")
    meta = PHASE2_DIR / f"{meta_name}.json"
    write_json(meta, {
        "hash": src_hash,
        "schema_version": SCHEMA_VERSION,
    })
