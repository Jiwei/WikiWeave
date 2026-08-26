"""Build and cache the person-alias table (pinyin/initials/typo/username → Chinese).

The alias table is data, not code. ``build_alias_map`` derives it as a pipeline step:
it merges a curated seed (``config/alias_map.json``) with chat-derived usernames
(``username(中文名)`` speakers) and caches the result to
``artifact/knowledge_bits/_alias_map.json``. Consumers call ``load_alias_map``, which
prefers the cache and falls back to the seed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from knowledge_extractor.util.paths import KB_DIR, ROOT
from knowledge_extractor.util import model_data

CACHE_PATH = KB_DIR / "_alias_map.json"

_MSG_RE = re.compile(r"^(.+?)\s+(\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2})$")


def _derive_chat_aliases(sources: list[dict], root: Path) -> dict[str, str]:
    """Derive username→Chinese aliases from chat ``username(中文名)`` speakers."""
    aliases: dict[str, str] = {}
    for src in sources:
        if src.get("source_type") != "chat":
            continue
        path = root / src["path"]
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = _MSG_RE.match(line)
            if not m:
                continue
            speaker = m.group(1).strip()
            # "user1(张三)" → explicit username↔name pair
            pm = re.match(r"^(.+?)\(([一-鿿]{2,4})\)$", speaker)
            if pm:
                aliases[pm.group(1)] = pm.group(2)
                aliases[speaker] = pm.group(2)
            else:
                # "赵六01" → numeric suffix is a dedup artifact, strip it
                cm = re.match(r"^([一-鿿]{2,4})\d+$", speaker)
                if cm:
                    aliases[speaker] = cm.group(1)
    return aliases


def build_alias_map(sources: list[dict], root: Path = ROOT) -> dict[str, str]:
    """Build the alias table from seed + chat usernames and cache it to disk."""
    seed = model_data.alias_map()  # 策展别名种子（原 config/alias_map.json）
    merged = {**_derive_chat_aliases(sources, root), **seed}  # curated seed wins
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


def load_alias_map() -> dict[str, str]:
    """Return the alias table, preferring the pipeline-built cache over the seed."""
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return model_data.alias_map()  # 策展别名种子（原 config/alias_map.json）
