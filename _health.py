"""整包体检:列出所有 .py,逐个 py_compile 与 import,报告写入 _health.txt。

绕开 shell 输出污染:结果落文件,主 agent 用 Read 读取。
"""
import os
import py_compile
import importlib

ROOT = os.path.dirname(os.path.abspath(__file__))          # knowledge_extractor/
PKG_PARENT = os.path.dirname(ROOT)
os.environ.setdefault("KE_INPUT_PATH", ROOT)

SKIP = {"_health", "_smoke_base"}

pyfiles = []
for dp, dn, fn in os.walk(ROOT):
    if "__pycache__" in dp:
        continue
    for f in fn:
        if f.endswith(".py"):
            pyfiles.append(os.path.join(dp, f))
pyfiles.sort()

lines = [f"TOTAL_PY={len(pyfiles)}", "", "== COMPILE =="]
compile_fail = 0
for p in pyfiles:
    rel = os.path.relpath(p, PKG_PARENT)
    try:
        py_compile.compile(p, doraise=True)
    except Exception as e:
        compile_fail += 1
        lines.append(f"FAIL {rel} :: {type(e).__name__}: {str(e)[:140]}")
lines.append(f"compile_fail={compile_fail}")

lines += ["", "== IMPORT =="]
import_fail = 0
for p in pyfiles:
    rel = os.path.relpath(p, PKG_PARENT)
    stem = os.path.basename(p)[:-3]
    if stem in SKIP:
        continue
    if rel.endswith("__init__.py"):
        mod = os.path.dirname(rel).replace(os.sep, ".")
    else:
        mod = rel[:-3].replace(os.sep, ".")
    if not mod:
        continue
    try:
        importlib.import_module(mod)
    except Exception as e:
        import_fail += 1
        lines.append(f"FAIL {mod} :: {type(e).__name__}: {str(e)[:180]}")
lines.append(f"import_fail={import_fail}")

with open(os.path.join(ROOT, "_health.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("HEALTH_DONE")
