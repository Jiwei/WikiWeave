"""实体抽取阶段（原参考项目 indexing/phase2）。

从 segmentation 产出的分类 section / code_unit 中抽取结构化实体：
  Task / Error / Workflow / Protocol / Project / Artifact，以及 people 归一化与索引。

子模块：
  cross_group   — 跨 section 分组（抽取前的规则 seed 分组）
  pipeline      — LLM 批量调用 / 批处理 / checkpoint / 置信度打分
  entities/*    — 六类实体抽取器
  normalize     — 人名归一化
  people_index  — 人物索引构建
  alias_map     — 别名表构建与缓存
  literals      — 资源/人名字面量抽取
  run           — 编排入口（原 run_phase2）
"""
