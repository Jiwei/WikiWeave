"""固定元函数词汇表(meta-functions)。

实体可被动态发现(entities.json),但作用于实体的变换是预定义的通用操作(静态词汇表):
  - extract   : LLM 从分组抽取实体实例(见 entities/generic.py)
  - seed      : 确定性从内容发现实体实例(规格解析 / 正则模式)
  - group     : 子实体归并到父实体(rollup + cluster,评分权重在 model_data)
  - index     : 聚合跨源提及为实体实例
  - synthesize: LLM 精炼实体实例(definition 提示词)

元函数签名: fn(cls, params, context, entities) -> list[dict](返回该实体的实例 dict 列表)。
entities.json 里每个实体用 "meta": [{"op": ..., "depends_on": [...], "params": {...}}] 声明。
"""
from __future__ import annotations

from knowledge_extractor.util.ontology import all_entities
from knowledge_extractor.extraction.pipeline import stage, tick, write_json

_META_OPS: dict[str, callable] = {}


def _key(op: str, params: dict) -> str:
    """注册表键 = op:kind(有 kind 时),否则 op。

    同一 op 可被不同实体以不同实现复用(如 seed:spec vs seed:regex),kind 用于区分。
    """
    kind = (params or {}).get("kind")
    return f"{op}:{kind}" if kind else op


def register(op: str, kind: str | None = None):
    """装饰器:把一个 (cls, params, context, entities) -> list[dict] 函数注册为某类元函数。"""
    def deco(fn):
        _META_OPS[f"{op}:{kind}" if kind else op] = fn
        return fn
    return deco


def run_meta(entities: dict, context: dict, out_dir) -> dict:
    """按 entities.json 的 meta 声明(含 depends_on 依赖)运行元函数,返回合并后的 {list_key -> [dict]}。

    同一实体的多个 meta(如 seed→group→synthesize)顺序执行,每一步的中间结果写回 entities,
    供下一步读取(从而 group 能拿到 seed 的 curated, synthesize 能拿到 group 的合并结果)。
    """
    stage("Meta-functions (declared in entities.json)")

    pending: dict[str, tuple] = {}
    for name, cls in all_entities().items():
        metas = getattr(cls, "_meta", [])
        if metas:
            deps = set()
            for m in metas:
                deps.update(m.get("depends_on", []))
            pending[name] = (cls, metas, deps)

    remaining = dict(pending)
    while remaining:
        progressed = False
        for name in list(remaining):
            cls, metas, deps = remaining[name]
            if all(d in entities for d in deps):
                lk = getattr(cls, "_list_key", "") or name.lower()
                out: list[dict] = []
                for m in metas:
                    fn = _META_OPS.get(_key(m["op"], m.get("params", {})))
                    if fn:
                        out = fn(cls, m.get("params", {}), context, entities)
                        entities[lk] = out  # 让后续 meta 读取上一步中间结果
                write_json(out_dir / f"{lk}.json", out)
                tick(f"saved {len(out)} {lk}")
                del remaining[name]
                progressed = True
        if not progressed:
            raise RuntimeError(f"meta 依赖无法满足: {list(remaining)}")
    return entities
