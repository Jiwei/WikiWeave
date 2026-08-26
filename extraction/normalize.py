"""Person name normalization for Phase 2 entities.

Post-processes extracted records to:
  1. Split comma/、-separated multi-person strings into lists
  2. Normalize nicknames (strip honorifics like 哥/姐/老师/总)
  3. Match given-name-only references against the pool of known full names
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from knowledge_extractor.util import model_data


# ── Honorific stripping ──

_HONORIFICS = model_data.honorifics()

# ── 中文姓氏:标准《百家姓》完整单姓 + 复姓表(替代自维护的 100 姓氏短表,覆盖 >99.9% 人口,
#    并支持复姓如 欧阳/司马/上官)。来源:北宋《百家姓》,canonical 标准姓氏表。 ──
_SINGLE_SURNAMES = frozenset(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章"
    "云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常"
    "乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞"
    "熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍"
    "虞万支柯昝管卢莫经房裘缪干解应宗丁宣贲邓郁单杭洪包诸左石崔吉钮龚程嵇邢滑裴陆荣翁"
    "荀羊於惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫"
    "宁仇栾暴甘钭厉戎祖武符刘景詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲邰从鄂索咸籍赖卓蔺屠蒙"
    "池乔阴郁胥能苍双闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍却璩桑桂濮牛寿通边扈燕冀郏浦尚农"
    "温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡国文寇广禄阙东"
    "欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养鞠须丰巢关蒯相查后荆红"
    "游竺权逯盖益桓公"
)
_COMPOUND_SURNAMES = frozenset(
    "万俟 司马 上官 欧阳 夏侯 诸葛 闻人 东方 赫连 皇甫 尉迟 公羊 澹台 公冶 宗政 濮阳 "
    "淳于 单于 太叔 申屠 公孙 仲孙 轩辕 令狐 钟离 宇文 长孙 慕容 鲜于 闾丘 司徒 司空 "
    "亓官 司寇 仉督 子车 颛孙 端木 巫马 公西 漆雕 乐正 壤驷 公良 拓跋 夹谷 宰父 谷梁 "
    "晋楚 闫法 汝鄢 涂钦 段干 百里 东郭 南门 呼延 归海 羊舌 微生 岳帅 缑亢 况后 有琴 "
    "梁丘 左丘 东门 西门 商牟 佘佴 伯赏 南宫 墨哈 谯笪 年爱 阳佟 第五 言福".split()
)

# Characters unlikely to appear in real Chinese names
_NON_NAME_RE = re.compile(r"[a-zA-Z0-9\s\.\,\;\:\!\?\@\#\$\%\^\&\*\(\)\-\+\=\[\]\{\}\|\/\~\`\'\"]")


_alias_cache: dict[str, str] | None = None


def _alias_map() -> dict[str, str]:
    """Lazy-load the alias table (built & cached by the pipeline, seed fallback)."""
    global _alias_cache
    if _alias_cache is None:
        from knowledge_extractor.extraction.alias_map import load_alias_map
        _alias_cache = load_alias_map()
    return _alias_cache


def _has_surname(name: str) -> bool:
    """Check if a Chinese name starts with a known surname (single or compound)."""
    if len(name) < 2:
        return False
    return name[0] in _SINGLE_SURNAMES or name[:2] in _COMPOUND_SURNAMES


def _is_valid_name(name: str) -> bool:
    """Check if a string looks like a valid Chinese person name."""
    if not name or len(name) < 2 or len(name) > 4:
        return False
    if _NON_NAME_RE.search(name):
        return False
    return all('一' <= c <= '鿿' for c in name)


def _strip_honorific(name: str) -> str:
    """Remove common Chinese honorifics from the end of a name."""
    for h in _HONORIFICS:
        if name.endswith(h) and len(name) > len(h):
            return name[:-len(h)]
    return name


def normalize_person_field(person: str, name_pool: set[str] | None = None) -> list[str]:
    """Normalize a person field into a list of clean person names.

    Handles:
      - Comma/Chinese-comma/semicolon-separated lists → splits into individuals
      - Honorific stripping: 张三哥 → 张三
      - Given-name-only matching: 三 → 张三 (if name_pool has 张三)
      - Non-name filtering: rejects obvious non-names like 接口, 架构

    Args:
        person: Raw person string from extraction
        name_pool: Set of known full names for given-name matching.
                   If None, no given-name-to-full-name matching is done.

    Returns:
        List of normalized person names (may be empty if all parts are invalid)
    """
    if not person or not person.strip():
        return []

    # Split on common separators
    parts = re.split(r'[,，、;\s]+', person.strip())
    parts = [p.strip() for p in parts if p.strip()]

    results = []
    for part in parts:
        # Skip obvious non-names and section header phrases（非人名黑名单 → model_data）
        _NON_NAME_PHRASES = set(model_data.nonperson_stoplist())
        if part in _NON_NAME_PHRASES:
            continue

        # Apply alias normalization (pinyin/initial/typo → canonical) BEFORE the
        # ASCII/name checks so "zhangsan" → "张三", "ls" → "李四" survive.
        part = _alias_map().get(part, part)

        # Strip honorifics
        cleaned = _strip_honorific(part)

        # Skip any remaining ASCII/pinyin that has no known alias (path components,
        # usernames like "zli", model tokens like "WF" that aren't mapped, etc.)
        if all(c.isascii() for c in cleaned):
            continue

        if not _is_valid_name(cleaned):
            continue

        # Full name with surname → keep directly
        if _has_surname(cleaned):
            results.append(cleaned)
            continue

        # Given-name-only (no surname) → try to match to name_pool
        if name_pool:
            matches = [n for n in name_pool
                       if n.endswith(cleaned) and len(n) > len(cleaned)]
            if len(matches) == 1:
                results.append(matches[0])
            elif len(matches) > 1:
                results.append(matches[0])  # ambiguous, pick first
            else:
                results.append(cleaned)  # no match, keep as-is
        else:
            results.append(cleaned)

    return results


def build_name_pool_from_sections(sections: list[dict], sources: list[dict] | None = None) -> set[str]:
    """Build a clean name pool from section metadata and source filenames.

    Filters to names that have a known Chinese surname — this excludes
    section sub-headers and other non-name strings that happen to be 2-4
    Chinese characters.
    """
    names = set()
    for sec in sections:
        for p in sec.get("people", []):
            if _is_valid_name(p) and _has_surname(p):
                names.add(p)

    if sources:
        for src in sources:
            fname = Path(src.get("path", "")).stem
            m = re.match(r'^([一-鿿]{2,4})[\s\-_]', fname)
            if m:
                name = m.group(1)
                if _has_surname(name):
                    names.add(name)

    return names
