"""路径中枢:包根、model_data、config、输入/输出目录。

与原 indexing/paths.py 的差异:输入路径不再写死为 ``source/``,而是由
config/settings.json 的 ``input_path`` 或 CLI 覆盖 —— 这是"通用工具"的前提。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PKG_ROOT / "config"
# model_schema = 预定义的知识数据结构(固定:schema + 基础实体 + 通用受控词汇)
# model_data   = 用户动态挖掘/定义的数据(exploration 输出;运行时与 model_schema 合并)
MODEL_SCHEMA_DIR = PKG_ROOT / "model_schema"
MODEL_DATA_DIR = PKG_ROOT / "model_data"


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


_SETTINGS = _load_json(CONFIG_DIR / "settings.json")


def settings() -> dict:
    return _SETTINGS


def input_path() -> Path:
    """待抽取的本地目录。优先级:环境变量 KE_INPUT_PATH > settings.input_path。"""
    p = os.environ.get("KE_INPUT_PATH") or _SETTINGS.get("input_path", "")
    if not p:
        raise ValueError("未设置输入路径:请用 CLI 传入、设 KE_INPUT_PATH 或写 config/settings.json 的 input_path")
    return Path(p).expanduser().resolve()


def language() -> str:
    """artifact 主要语言(默认 en)。优先级:环境变量 KE_LANG > settings.language。"""
    lang = (os.environ.get("KE_LANG") or _SETTINGS.get("language", "en")).lower()
    return "zh" if lang.startswith("zh") else "en"


def output_dir() -> Path:
    od = _SETTINGS.get("output_dir", "artifact")
    p = Path(od)
    return p if p.is_absolute() else PKG_ROOT / p


def _out_sub(key: str, default: str) -> Path:
    sub = _SETTINGS.get("output", {}).get(key, default)
    return output_dir() / sub


def kb_dir() -> Path:
    return _out_sub("knowledge_bits_subdir", "knowledge_bits")


def index_dir() -> Path:
    return _out_sub("index_subdir", "index")


def index_html_dir() -> Path:
    return _out_sub("index_html_subdir", "index_html")


def report_dir() -> Path:
    return _out_sub("reports_subdir", "reports")


def source_hash_path() -> Path:
    return kb_dir() / "_source_hashes.json"


def portable_path(path: str) -> str:
    """把绝对输入路径转成以输入根为基准的相对路径,让产物可移植。"""
    try:
        return str(Path(path).resolve().relative_to(input_path()))
    except Exception:
        return path


# ── 兼容层:承接参考项目 indexing/paths.py 的旧名(供搬移代码零改动使用) ──

def root() -> Path:
    """旧 ROOT 语义:源材料根。未设输入路径时兜底为包根,避免 import 期崩溃。"""
    try:
        return input_path()
    except Exception:
        return PKG_ROOT


def portable_code_path(path: str) -> str:
    return portable_path(path)


# ROOT = 源材料根(输入目录),消费方以 ``ROOT / source.path`` 读取源文件。
# 注意:不是包目录 —— 包内资源用 PKG_ROOT/MODEL_DATA_DIR/CONFIG_DIR。
# 若未设输入路径则 root() 兜底为 PKG_ROOT,避免 import 期崩溃。
ROOT = root()

KB_DIR = kb_dir()
OUT_DIR = output_dir()
INDEX_DIR = index_dir()
INDEX_HTML_DIR = index_html_dir()
REPORT_DIR = report_dir()

def get_llm_settings(phase):
    from util import llm
    return llm.phase_settings(phase)
