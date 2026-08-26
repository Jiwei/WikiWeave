"""元函数实现包(固定词汇表 seed/group/index/synthesize 的代码侧)。

导入本包会触发所有 @register,把各实体的元函数登记进 extraction/meta.py 的注册表。
按「操作」组织(而非按实体):新增需要 group/index 的实体直接复用,无需新文件。
"""
from . import seed, group, index, synthesize  # noqa: F401
