"""Render readable code structure pages from Phase 1 AST + code unit data.

Usage: python -m indexing.render_code_index
Input:  artifact/knowledge_bits/code_units.json + code_asts.json + sources.json + doc_sections.json
Output: artifact/index/code/ (OKF markdown pages)

Each page shows a Python file's structure: imports, functions/classes with docstrings,
string literals (paths), and cross-references to document sections that mention this code.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from knowledge_extractor.util.paths import index_dir, kb_dir, root, portable_code_path

INDEX_DIR = index_dir()
KB_DIR = kb_dir()
# TODO(搬移): 新 paths 未在改写规则中列出 root();若无此函数需按实际接口调整
ROOT = root()

_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

CODE_OUT = INDEX_DIR / "code"


def _fm(typ: str, title: str = "", description: str = "", tags: list[str] | None = None) -> str:
    """OKF frontmatter."""
    lines = ["---", f"type: {typ}"]
    if title:
        lines.append(f"title: {title}")
    if description:
        lines.append(f"description: {description}")
    if tags:
        lines.append(f"tags: [{', '.join(tags)}]")
    lines.append(f"timestamp: {_TIMESTAMP}")
    lines.append("---\n")
    return "\n".join(lines)


def _rel_link(target: str, depth: int) -> str:
    """Generate a relative markdown link."""
    import urllib.parse
    encoded = target
    if any(c in target for c in "() "):
        parts = target.split("/")
        encoded = "/".join(
            urllib.parse.quote(p, safe="") if any(c in p for c in "() ") else p
            for p in parts
        )
    prefix = "/".join([".."] * depth)
    return f"[{target}]({prefix}/{encoded})"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _parse_doc_mentions(mentions: list[str]) -> dict[str, float]:
    """Parse doc_mention strings like 'src-xxx-sec-5:0.5' into {section_id: confidence}."""
    result = {}
    for m in mentions:
        if ":" in m:
            sid, conf = m.rsplit(":", 1)
            try:
                result[sid] = float(conf)
            except ValueError:
                result[sid] = 0.0
    return result


def _read_source_lines(path: str) -> list[str]:
    """Read a source file's lines from disk (empty list on any failure)."""
    try:
        return (ROOT / path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []


def _slice_source(lines: list[str], locator: dict) -> str:
    """Slice ``lines`` to the ``locator`` line range, or "" if the range is unusable."""
    if not lines:
        return ""
    start = locator.get("start_line")
    end = locator.get("end_line")
    if not isinstance(start, int) or not isinstance(end, int):
        return ""
    s = max(1, start)
    e = min(end, len(lines))
    if s > e:
        return ""
    return "\n".join(lines[s - 1 : e])


def _module_source(lines: list[str], culist: list[dict]) -> str:
    """Top-level source not inside any function/class body (the module's own code).

    Module-level code is non-contiguous (interleaved with ``def``/``class``), so it is
    computed as the complement of the function/class line ranges: the module docstring,
    imports, constants, and any ``__main__`` guard.
    """
    if not lines:
        return ""
    ranges = []
    for cu in culist:
        if cu.get("kind") not in ("function", "class"):
            continue
        loc = cu.get("locator", {})
        s, e = loc.get("start_line"), loc.get("end_line")
        if isinstance(s, int) and isinstance(e, int):
            ranges.append((s, e))
    ranges.sort()
    segments: list[str] = []
    cursor = 1
    for s, e in ranges:
        if s > cursor:
            segments.append("\n".join(lines[cursor - 1 : s - 1]))
        cursor = max(cursor, e + 1)
    if cursor <= len(lines):
        segments.append("\n".join(lines[cursor - 1 :]))
    return "\n".join(seg for seg in segments if seg.strip())


def render_code_index() -> None:
    """Main renderer: generates artifact/index/code/ pages."""
    # ── Load data ──
    with open(KB_DIR / "code_units.json", encoding="utf-8") as f:
        units = json.load(f)
    with open(KB_DIR / "sources.json", encoding="utf-8") as f:
        sources = {s["id"]: s for s in json.load(f)}
    with open(KB_DIR / "doc_sections.json", encoding="utf-8") as f:
        sections = {s["id"]: s for s in json.load(f)}

    # ── Group code units by source file ──
    units_by_source: dict[str, list[dict]] = defaultdict(list)
    for cu in units:
        units_by_source[cu["source_id"]].append(cu)

    # ── Filter: exclude .ipynb_checkpoints ──
    file_entries = []
    for sid, culist in units_by_source.items():
        src = sources.get(sid)
        if not src:
            continue
        path = src["path"]
        if ".ipynb_checkpoints" in path:
            continue
        file_entries.append((sid, src, culist))

    file_entries.sort(key=lambda x: x[1]["path"])

    # ── Directory grouping for index ──
    dir_groups: dict[str, list] = defaultdict(list)
    for sid, src, culist in file_entries:
        dir_name = str(Path(src["path"]).parent)
        dir_groups[dir_name].append((sid, src, culist))

    # ── Render per-file pages ──
    CODE_OUT.mkdir(parents=True, exist_ok=True)
    rendered_count = 0

    for sid, src, culist in file_entries:
        path = src["path"]
        rel = Path(path)

        # Full source lines, read once per file and sliced per unit below.
        source_lines = _read_source_lines(path)

        # Module-level code (docstring, imports, constants, __main__) — rendered once
        # as a first-class "module" entity above its function/class members.
        module_src = _module_source(source_lines, culist)

        # Collect imports (from first unit that has imports, usually the module unit)
        all_imports = []
        for cu in culist:
            all_imports.extend(cu.get("imports", []))
        all_imports = sorted(set(all_imports))

        # Collect all string literals across units
        all_literals = []
        for cu in culist:
            all_literals.extend(cu.get("literals", []))
        all_literals = sorted(set(all_literals))

        # Collect all doc mentions across units
        all_mentions: dict[str, float] = {}
        for cu in culist:
            mentions = _parse_doc_mentions(cu.get("doc_mentions", []))
            for sec_id, conf in mentions.items():
                all_mentions[sec_id] = max(all_mentions.get(sec_id, 0), conf)

        # Sort mentions by confidence
        sorted_mentions = sorted(all_mentions.items(), key=lambda x: -x[1])

        # ── Build page ──
        func_count = sum(1 for cu in culist if cu["kind"] == "function")
        class_count = sum(1 for cu in culist if cu["kind"] == "class")
        module_count = 1 if module_src else 0

        parts = [f"# {rel.name}\n"]
        parts.append(f"**Path**: `{portable_code_path(path)}`  ")
        parts.append(f"**Source type**: `{src['source_type']}`  ")
        parts.append(
            f"**Structure**: {func_count} functions, {class_count} classes, "
            f"{module_count} modules  "
        )

        # ── Imports section ──
        if all_imports:
            parts.append("\n## Imports\n")
            # Group imports: stdlib vs third-party vs internal
            stdlib = {"os", "sys", "re", "json", "hashlib", "subprocess", "datetime",
                      "logging", "ast", "csv", "io", "math", "pathlib", "collections",
                      "typing", "time", "gc", "argparse", "shutil", "glob", "tempfile",
                      "copy", "itertools", "functools", "multiprocessing", "threading"}
            third_party = {"pandas", "numpy", "pyarrow", "tqdm", "pandarallel",
                          "scipy", "sklearn", "torch", "matplotlib", "seaborn",
                          "polars", "hdbscan", "sentence_transformers", "pymupdf",
                          "docx", "markdown", "yaml", "PIL", "cv2"}

            internal = []
            external = []
            for imp in all_imports:
                base = imp.split(".")[0].split(":")[0]
                if base in stdlib:
                    continue  # skip stdlib for clarity
                elif base in third_party:
                    external.append(imp)
                else:
                    internal.append(imp)

            if internal:
                parts.append("### Internal / Project\n")
                for imp in sorted(set(internal)):
                    parts.append(f"- `{imp}`")
                parts.append("")
            if external:
                parts.append("### Third-party\n")
                for imp in sorted(set(external)):
                    parts.append(f"- `{imp}`")
                parts.append("")

        # ── Module-level code (first-class "module" entity) ──
        if module_src:
            module_cu = next((cu for cu in culist if cu.get("kind") == "module"), None)
            module_anchor = module_cu["id"] if module_cu else f"module-{sid}"
            parts.append(f'\n<a id="{module_anchor}"></a>\n')
            parts.append(f"### `{rel.stem}` (module)\n")
            parts.append("Top-level code — module docstring, imports, constants, `__main__`.\n")
            fence = "`" * (4 if "```" in module_src else 3)
            parts.append(f"\n{fence}python\n{module_src}\n{fence}\n")

        # ── Code Units (functions, classes, modules) ──
        for cu in sorted(culist, key=lambda c: c.get("locator", {}).get("start_line", 0)):
            kind = cu["kind"]
            name = cu["name"]
            loc = cu.get("locator", {})
            start, end = loc.get("start_line", "?"), loc.get("end_line", "?")

            if kind == "module":
                continue  # module-level code rendered separately above

            # Stable anchor so code-units.md / code map can deep-link to this unit.
            anchor = cu.get("id", f"cu-{name}")
            parts.append(f'<a id="{anchor}"></a>\n')
            parts.append(f"### `{name}` ({kind})\n")
            parts.append(f"Lines {start}–{end}  ")

            if cu.get("doc"):
                doc = cu["doc"].strip()
                parts.append(f"\n```text\n{doc[:2000]}\n```\n")

            # Full function/class body — closes the test-report gap where the code
            # index told *where* a function is but not *what* its source is.
            src_body = _slice_source(source_lines, loc)
            if src_body:
                lang = cu.get("language", "python") or "python"
                fence = "`" * (4 if "```" in src_body else 3)
                parts.append(f"\n{fence}{lang}\n{src_body}\n{fence}\n")

            # Show literals for this unit
            cu_literals = cu.get("literals", [])
            if cu_literals:
                parts.append("\n**Paths & file references:**\n")
                for lit in cu_literals[:20]:
                    parts.append(f"- `{lit}`")
                parts.append("")

            # Show doc mentions for this unit
            cu_mentions = _parse_doc_mentions(cu.get("doc_mentions", []))
            if cu_mentions:
                top_mentions = sorted(cu_mentions.items(), key=lambda x: -x[1])[:5]
                parts.append("\n**Mentioned in:**\n")
                for sec_id, conf in top_mentions:
                    sec = sections.get(sec_id, {})
                    sec_name = sec.get("name", sec_id)[:80] if sec else sec_id
                    parts.append(f"- [{sec_name}](../sections/{sec_id}.md) (confidence: {conf:.1f})")
                parts.append("")

        # ── All doc mentions (cross-file aggregate) ──
        if sorted_mentions:
            parts.append("## All Document Mentions\n")
            for sec_id, conf in sorted_mentions[:30]:
                sec = sections.get(sec_id, {})
                sec_name = sec.get("name", sec_id)[:80] if sec else sec_id
                parts.append(f"- [{sec_name}](../sections/{sec_id}.md) (confidence: {conf:.1f})")

        # ── Navigation ──
        parts.append(f"\n## See also\n")
        parts.append(f"- [Code index](../index.md) — all code files")

        # ── Write ──
        # Use the full relative path for the output file to preserve directory structure
        page_name = rel.as_posix().replace("/", "_") + ".md"
        _write(CODE_OUT / page_name,
               _fm("CodeModule", rel.name,
                   f"{func_count} functions, {class_count} classes") +
               "\n".join(parts))
        rendered_count += 1

    # ── Render code index page ──
    idx_parts = [
        "# Code Structure Index\n",
        f"**{len(file_entries)}** Python source files with **{len(units)}** code units "
        f"(functions, classes, modules).\n",
        "\nEach page shows a file's imports, functions/classes with docstrings, "
        "string literals (paths and file references), and cross-references to "
        "document sections that mention this code.\n",
        "\n## Files by Directory\n",
    ]

    for dir_name in sorted(dir_groups.keys()):
        entries = dir_groups[dir_name]
        display = dir_name.replace("source/", "", 1) if dir_name.startswith("source/") else dir_name
        func_total = sum(
            sum(1 for cu in culist if cu["kind"] == "function")
            for _, _, culist in entries
        )
        idx_parts.append(f"\n### {display} ({len(entries)} files, {func_total} functions)\n")
        for sid, src, culist in entries:
            fname = Path(src["path"]).name
            page_name = src["path"].replace("/", "_") + ".md"
            funcs = sum(1 for cu in culist if cu["kind"] == "function")
            classes = sum(1 for cu in culist if cu["kind"] == "class")
            desc_parts = [f"{funcs} funcs"] if funcs else []
            if classes:
                desc_parts.append(f"{classes} classes")
            desc = ", ".join(desc_parts) if desc_parts else "module"
            idx_parts.append(f"- [{fname}]({page_name}) — {desc}")

    # Navigation
    idx_parts.append(f"\n## See also\n")
    idx_parts.append(f"- [Knowledge index](../index.md) — main wiki index")
    idx_parts.append(f"- [All entity types](../index.md#by-entity-type) — entity indexes")

    _write(CODE_OUT / "index.md",
           _fm("CodeIndex", "Code Structure Index",
               f"{len(file_entries)} Python source files") +
           "\n".join(idx_parts))

    print(f"Code index: {rendered_count} file pages + index page → {CODE_OUT}")


if __name__ == "__main__":
    render_code_index()
