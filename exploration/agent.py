"""调用外部智能体执行 exploration(EDA → 写回 model_data)。

当前仅支持 Claude Code(claude CLI,headless print 模式)。其它 agent harness 后续再接入。
"""
from __future__ import annotations

import subprocess

from util.paths import PKG_ROOT, input_path


def skill_path() -> str:
    """explore_skill 的 SKILL.md 绝对路径(交给智能体阅读)。"""
    return str(PKG_ROOT / "exploration" / "explore_skill" / "SKILL.md")


def build_prompt() -> str:
    """构造交给 Claude Code 智能体的 exploration 任务提示。"""
    return (
        f"请阅读并严格遵循知识抽取探索技能 {skill_path()}。\n"
        f"对输入语料目录 {input_path()} 做 EDA,发现并写回完整 model_data/ 的四个维度:"
        "sources/entities.json + sources/source_types.json、"
        "domains/definitions.json + sub_domains.json + valid_domains.json + keywords.json、"
        "code/entities.json + code_domain.json + naming_keys.json(若语料含代码)、"
        "ontology/entities.json + 人物/资产词汇(若语料含非代码)。"
        "必须满足 model_data 完整性契约(见技能)。"
        "清点语料请复用 exploration.inventory"
        "(detect_content_types / build_source_inventory / infer_source_type),勿手写扫描逻辑。"
        "写回后运行 `python run.py --skip-check --phase 1` 验证门禁通过。"
    )


def invoke_exploration() -> int:
    """调用 Claude Code 智能体执行 exploration,返回其退出码。

    注:headless 非交互写回 model_data 需要跳过权限提示,故加 --dangerously-skip-permissions;
    若你的 Claude Code 已配置相应权限规则,可去掉该 flag。
    """
    prompt = build_prompt()
    print("→ 调用 Claude Code 智能体执行 exploration(claude -p)…", flush=True)
    cmd = ["claude", "-p", "--dangerously-skip-permissions", prompt]
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("✋ 未找到 claude CLI,无法自动执行 exploration。"
              "请安装 Claude Code,或手动按 exploration/explore_skill/SKILL.md 完善 model_data/。", flush=True)
        return 1
