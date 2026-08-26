"""Convert the Phase 1 OKF markdown index into a self-contained navigable HTML site.

Usage: python -m indexing.render_html
Input:  artifact/index/
Output: artifact/index_html/
"""

from __future__ import annotations

import re
from pathlib import Path

import markdown
import yaml

from util.paths import index_dir, output_dir
from util import model_data

INDEX_DIR = index_dir()
OUT_DIR = output_dir()

HTML_OUT = OUT_DIR / "index_html"

# ── CSS (inline for self-contained pages) ──

CSS = """\
:root {
    --bg: #f8f9fa; --surface: #ffffff; --text: #212529; --muted: #6c757d;
    --border: #dee2e6; --accent: #2563eb; --accent-hover: #1d4ed8;
    --code-bg: #f1f5f9; --sidebar-w: 260px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans SC", sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; display: flex; min-height: 100vh; }
/* sidebar */
nav { width: var(--sidebar-w); min-width: var(--sidebar-w); background: #1e293b; color: #e2e8f0; padding: 24px 0; position: sticky; top: 0; max-height: 100vh; overflow-y: auto; font-size: 14px; }
nav h2 { font-size: 16px; padding: 0 20px 16px; color: #fff; border-bottom: 1px solid #334155; margin-bottom: 12px; }
nav h2 a { color: #fff; text-decoration: none; }
nav h2 a:hover { text-decoration: underline; }
nav .nav-section { margin-bottom: 16px; }
nav .nav-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #94a3b8; padding: 8px 20px 4px; font-weight: 600; }
nav a { display: block; padding: 4px 20px; color: #cbd5e1; text-decoration: none; border-left: 2px solid transparent; transition: all 0.15s; }
nav a:hover { color: #fff; background: #334155; border-left-color: var(--accent); }
nav a.active { color: #fff; background: #1e3a5f; border-left-color: #60a5fa; font-weight: 500; }
nav .count { font-size: 11px; color: #64748b; }
/* main */
main { flex: 1; padding: 32px 40px; max-width: 900px; }
main h1 { font-size: 24px; margin-bottom: 4px; color: #1e293b; }
main h2 { font-size: 18px; margin: 28px 0 10px; padding-bottom: 6px; border-bottom: 1px solid var(--border); color: #334155; }
main h3 { font-size: 15px; margin: 20px 0 8px; color: #475569; }
main p, main li { margin-bottom: 6px; }
main a { color: var(--accent); text-decoration: none; }
main a:hover { text-decoration: underline; }
main ul, main ol { padding-left: 24px; margin-bottom: 12px; }
main code { background: var(--code-bg); padding: 2px 6px; border-radius: 4px; font-size: 13px; font-family: "SF Mono", "Fira Code", "Cascadia Code", Consolas, monospace; }
main pre { background: #1e293b; color: #e2e8f0; padding: 16px 20px; border-radius: 8px; overflow-x: auto; margin: 12px 0; font-size: 13px; line-height: 1.5; }
main pre code { background: none; padding: 0; color: inherit; }
main table { border-collapse: collapse; width: 100%; margin: 12px 0; }
main th, main td { border: 1px solid var(--border); padding: 8px 12px; text-align: left; font-size: 14px; }
main th { background: #f1f5f9; font-weight: 600; }
main hr { border: none; border-top: 1px solid var(--border); margin: 24px 0; }
main blockquote { border-left: 3px solid var(--accent); padding: 4px 16px; margin: 12px 0; color: var(--muted); background: #f8fafc; border-radius: 0 6px 6px 0; }
main strong { font-weight: 600; color: #1e293b; }
/* meta bar */
.meta-bar { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; font-size: 13px; color: var(--muted); }
.meta-bar span { background: var(--code-bg); padding: 2px 10px; border-radius: 12px; }
.domain-tag { background: #dbeafe !important; color: #1e40af !important; font-weight: 500; }
/* stats grid */
.stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; margin: 16px 0; }
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; text-align: center; }
.stat-card .num { font-size: 28px; font-weight: 700; color: var(--accent); }
.stat-card .label { font-size: 12px; color: var(--muted); margin-top: 4px; }
"""

# ── markdown converter ──

MD = markdown.Markdown(extensions=["extra", "codehilite", "toc"])


def _strip_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter. Returns (meta_dict, body_text)."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    try:
        meta = yaml.safe_load(text[4:end])
    except Exception:
        meta = {}
    return (meta or {}), text[end + 5:]


def _depth(path: Path, base: Path) -> int:
    """Compute directory depth relative to base. Used for relative path prefixes."""
    try:
        rel = path.parent.relative_to(base)
        return len(rel.parts)
    except ValueError:
        return 0


def _rel_prefix(depth: int) -> str:
    """Relative path prefix to reach root from given depth."""
    return "/".join([".."] * depth) if depth > 0 else "."


def _md_to_html(text: str, current_depth: int) -> str:
    """Convert markdown body to HTML, fixing .md→.html in internal wiki links.

    Only source-file links (containing 'source/' in the path) keep .md extension.
    All other .md links — wiki-internal paths like sections/, domains/, indexes/,
    and bare relative filenames like code_unit.md — are converted to .html.
    """
    def _fix_link(m: re.Match) -> str:
        url = m.group(1)
        # Keep external, anchor-only, and non-.md links unchanged
        if url.startswith("http://") or url.startswith("https://") or url.startswith("#"):
            return m.group(0)
        # split off any anchor fragment so ".md#..." is detected as an .md link
        path_part, _, frag = url.partition("#")
        if not path_part.endswith(".md"):
            return m.group(0)
        # Source file references (e.g. ../../../source/chat/example.md) keep .md
        if "source/" in path_part:
            return m.group(0)
        # All other .md links are wiki-internal → convert to .html (keep fragment)
        new_url = path_part[:-3] + ".html" + (f"#{frag}" if frag else "")
        return f"]({new_url})"

    text = re.sub(r"\]\(([^)]+)\)", _fix_link, text)
    return MD.convert(text)


# ── page templates ──

def _nav_html(current_path: str, domains: list[str], domain_counts: dict,
              entity_types: list[str],
              entity_hrefs: dict, entity_counts: dict,
              source_types: list[str]) -> str:
    """Build sidebar navigation HTML: Domains / Entities / Code / Source."""

    def _link(label: str, href: str, count: str = "") -> str:
        cls = ' class="active"' if href == current_path else ""
        ct = f' <span class="count">{count}</span>' if count else ""
        return f'<a href="{href}"{cls}>{label}{ct}</a>'

    parts = [
        '<nav>',
        f'<h2><a href="index.html">📚 Knowledge Index</a></h2>',
        '<div class="nav-section">',
        '<div class="nav-label">Domains</div>',
    ]
    for d in domains:
        dc = domain_counts.get(d, 0)
        parts.append(_link(d, f"domains/{d}/index.html", str(dc)))
    parts.append('</div>')

    # Entities — each entity type links to its canonical page
    parts.append('<div class="nav-section"><div class="nav-label">Entities</div>')
    for et in entity_types:
        href = entity_hrefs.get(et, f"index.html#by-entity-type")
        cnt = entity_counts.get(et, "")
        parts.append(_link(et, href, str(cnt) if cnt else ""))
    parts.append('</div>')

    # Glossary
    parts.append('<div class="nav-section"><div class="nav-label">Glossary</div>')
    parts.append(_link("Terms", "TERMS.html"))
    parts.append('</div>')

    # Code
    parts.append('<div class="nav-section"><div class="nav-label">Code</div>')
    parts.append(_link("Code structure", "code/index.html"))
    parts.append('</div>')

    # Source types
    parts.append('<div class="nav-section"><div class="nav-label">Source</div>')
    parts.append(_link("Source types", "indexes/by-source-type/index.html"))
    parts.append('</div>')

    parts.append('</nav>')
    return "\n".join(parts)


def _page_html(title: str, meta: dict, body_html: str, nav: str, depth: int,
               meta_html: str = "") -> str:
    """Wrap body in full page template."""
    prefix = _rel_prefix(depth)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — Phase 1 Knowledge Index</title>
<style>{CSS}</style>
</head>
<body>
{nav}
<main>
<h1>{title}</h1>
{meta_html}
{body_html}
</main>
</body>
</html>"""


def _meta_html(meta: dict) -> str:
    """Render frontmatter metadata as a tag bar."""
    tags = []
    if meta.get("type"):
        tags.append(f'<span class="domain-tag">{meta["type"]}</span>')
    if meta.get("domains"):
        if isinstance(meta["domains"], list):
            tags.append(f'<span>Domains: {", ".join(meta["domains"])}</span>')
        else:
            tags.append(f'<span>Domains: {meta["domains"]}</span>')
    if meta.get("description"):
        tags.append(f"<span>{meta['description']}</span>")
    if meta.get("timestamp"):
        ts = meta["timestamp"]
        if hasattr(ts, "strftime"):
            tags.append(f"<span>{ts.strftime('%Y-%m-%d')}</span>")
        else:
            tags.append(f"<span>{str(ts)[:10]}</span>")
    if not tags:
        return ""
    return f'<div class="meta-bar">{"".join(tags)}</div>'


# ── sidebar data collection ──

def _collect_nav_data(index_dir: Path) -> tuple[list[str], dict[str, int], list[str], list[str], list[str]]:
    """Scan the index directory to collect domain names, counts, and index types."""
    domain_counts: dict[str, int] = {}
    domains = []

    # Read root index for domain list
    root_md = index_dir / "index.md"
    if root_md.exists():
        text = root_md.read_text(encoding="utf-8")
        in_domains = False
        for line in text.splitlines():
            if line.startswith("## Domains"):
                in_domains = True
                continue
            if line.startswith("##") and in_domains:
                in_domains = False
            if in_domains and "](domains/" in line:
                # Extract name from markdown link: **name** or name
                import re
                m = re.search(r'\[(?:\*\*)?([^*\]]+)(?:\*\*)?\]\(domains/([^/]+)/', line)
                if m:
                    name = m.group(2)
                    domains.append(name)
                # Extract count: (NNN sections)
                c = re.search(r'\((\d+)\s+sections?\)', line)
                if c:
                    domain_counts[name] = int(c.group(1))

    # Canonical entity pages, in display order. Each entity type lives at its own
    # canonical page; counts are read from the root index's "By entity type" hub.
    entity_types = ["person", "project", "artifact", "task", "error", "workflow",
                    "protocol", "doc_section", "code_unit", "resource"]
    entity_hrefs = {
        "person": "indexes/by-person.html",
        "project": "projects/index.html",
        "artifact": "artifacts/index.html",
        "task": "infos/tasks/index.html",
        "error": "infos/errors/index.html",
        "workflow": "infos/workflows/index.html",
        "protocol": "infos/protocols/index.html",
        # doc_section has no standalone index; point to the root entity hub
        "doc_section": "index.html#by-entity-type",
        "code_unit": "code/code-units.html",
        "resource": "indexes/by-resource.html",
    }
    entity_counts = {}
    if root_md.exists():
        text = root_md.read_text(encoding="utf-8")
        in_entities = False
        for line in text.splitlines():
            if line.startswith("### By entity type"):
                in_entities = True
                continue
            if line.startswith("###") and in_entities:
                break
            if in_entities:
                # parse "- **[task](infos/tasks/index.md)** (1618) — ..."
                # and "- **[person](indexes/by-person.md)** (182 items) — ..."
                m = re.match(r"- \*\*\[([a-z_]+)\]\(([^)]+)\)\*\* \((\d+)( items)?\)", line)
                if m:
                    name = m.group(1)
                    entity_counts[name] = int(m.group(3))

    # Source types
    source_types = []

    return domains, domain_counts, entity_types, entity_hrefs, entity_counts, source_types


# ── main conversion ──

def convert_index_to_html() -> None:
    index_dir = INDEX_DIR
    out_dir = HTML_OUT

    # Clean and recreate output
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (domains, domain_counts, entity_types,
     entity_hrefs, entity_counts, source_types) = _collect_nav_data(index_dir)

    # Walk all .md files
    md_files = list(index_dir.rglob("*.md"))
    total = len(md_files)
    print(f"Converting {total} markdown files to HTML...")

    for i, md_path in enumerate(md_files):
        if i % 500 == 0:
            print(f"  {i}/{total}...")

        rel = md_path.relative_to(index_dir)
        out_path = out_dir / rel.with_suffix(".html")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        text = md_path.read_text(encoding="utf-8")
        meta, body = _strip_frontmatter(text)

        # The page template already renders the title as <h1>; drop the body's
        # leading "# " H1 so it doesn't appear twice (duplicate-title bug).
        body = re.sub(r"^#\s+.*$", "", body, count=1, flags=re.MULTILINE)

        depth = _depth(out_path, out_dir)
        body_html = _md_to_html(body, depth)

        # current page for nav active state
        current = str(rel.with_suffix(".html"))
        # adjust nav links for depth
        prefix = _rel_prefix(depth)
        nav = _nav_html(current, domains, domain_counts,
                        entity_types, entity_hrefs, entity_counts, source_types)
        # fix nav links for depth — cover every root-level nav target
        nav = re.sub(r'href="(domains|indexes|infos|projects|artifacts|registries|code|sections|TERMS|index\.html)',
                     rf'href="{prefix}/\1', nav)

        title = meta.get("title", md_path.stem) if meta else md_path.stem
        page = _page_html(title, meta, body_html, nav, depth, _meta_html(meta))
        out_path.write_text(page, encoding="utf-8")

    # copy CSS is inline, no extra assets needed

    # ── Copy AGENTS.md to HTML output ──
    agents_md = index_dir / "AGENTS.md"
    if agents_md.exists():
        import shutil, re as _re2
        shutil.copy2(agents_md, out_dir / "AGENTS.md")
        # Also plain-text version
        content = agents_md.read_text(encoding="utf-8")
        plain = _re2.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
        (out_dir / "AGENTS.txt").write_text(plain, encoding="utf-8")

    print(f"Done → {out_dir}")
    print(f"  {total} HTML pages written")
    print(f"  Open: {out_dir / 'index.html'}")


if __name__ == "__main__":
    convert_index_to_html()
