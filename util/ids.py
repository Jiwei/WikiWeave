"""稳定 ID 工具:路径/键 → 确定性短 ID(checkpoint 复用的基石)。"""
from __future__ import annotations

import hashlib


def sha_id(prefix: str, key: str, length: int = 12) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}-{digest}"


def file_hash(path, head_bytes: int = 65536) -> str:
    """只读前 head_bytes + size + mtime 的 sha256,大文件不全量读(增量指纹)。"""
    import os
    h = hashlib.sha256()
    try:
        st = os.stat(path)
        with open(path, "rb") as f:
            h.update(f.read(head_bytes))
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
    except Exception:
        return ""
    return h.hexdigest()
