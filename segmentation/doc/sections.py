"""Domain-aware segmentation — produces DocSectionRecord and CodeUnitRecord."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from util import model_data
from util.ontology import DocSectionRecord, SourceRecord
from util.paths import ROOT

HEADING_RE = re.compile(r"^(#+)\s+(.*)$")
MSG_RE = re.compile(r"^(.+?)\s+(\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2})$")
MAX_FALLBACK_SECTIONS = 20
MIN_SECTION_CHARS = 200  # raised from 50 — merge tiny sections into neighbors
MIN_SECTION_LINES = 5    # raised from 3 — preserve meaningful content units

# 非人名短语(如「本周总结」「因子」):单一事实源 model_data/ontology/nonperson_stoplist.json
_NONPERSON_WORDS = model_data.nonperson_stoplist()


def _sid(prefix: str, key: str) -> str:
    return f"{prefix}-{hashlib.sha1(key.encode()).hexdigest()[:12]}"


_pdf_text_cache: dict[str, str] = {}


def _read_text(path: Path) -> str:
    """Read file content, extracting text from PDFs when needed."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        key = str(path)
        if key not in _pdf_text_cache:
            try:
                import fitz
                doc = fitz.open(path)
                pages = [page.get_text() for page in doc]
                doc.close()
                _pdf_text_cache[key] = "\n".join(pages)
            except Exception:
                _pdf_text_cache[key] = path.read_text(encoding="utf-8", errors="ignore")
        return _pdf_text_cache[key]
    if suffix in (".docx",):
        try:
            import docx
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            pass
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_lines(path: Path) -> list[str]:
    return _read_text(path).splitlines()


def _doc_section(source_id: str, idx: int, name: str, kind: str,
                 start: int, end: int, spath: list[str] | None = None,
                 people: list[str] | None = None, msg_count: int = 0,
                 time_range: str = "", msg_ranges: list[dict] | None = None,
                 content: str = "") -> DocSectionRecord:
    loc = {"messages": msg_ranges} if msg_ranges else {"start_line": start, "end_line": end}
    return DocSectionRecord(
        id=f"{source_id}-sec-{idx}", source_id=source_id,
        name=name, kind=kind,
        section_path=spath or [name],
        locator=loc,
        people=people or [], message_count=msg_count, time_range=time_range,
        content=content,
    )


# ── Markdown ──

def segment_markdown(source_id: str, path: Path) -> list[DocSectionRecord]:
    lines = _read_lines(path)
    sections: list[DocSectionRecord] = []
    name = source_id.rsplit("-", 1)[-1] if "-" in source_id else path.stem
    current_name = name
    current_path = [name]
    start_line, idx = 1, 0
    for line_no, line in enumerate(lines, start=1):
        m = HEADING_RE.match(line)
        if not m: continue
        if line_no > start_line:
            content = "\n".join(lines[start_line - 1 : line_no - 1])
            sections.append(_doc_section(source_id, idx, current_name, "heading_block", start_line, line_no - 1, list(current_path), content=content))
            idx += 1
        current_name = m.group(2).strip()
        current_path = [name, current_name]
        start_line = line_no
    if lines:
        content = "\n".join(lines[start_line - 1 :])
        sections.append(_doc_section(source_id, idx, current_name, "heading_block", start_line, len(lines), list(current_path), content=content))
    return sections


# ── Chat (late-chunk embedding + HDBSCAN clustering → named segments) ──

_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("BAAI/bge-m3", device="mps")
    return _embed_model


def _late_chunk_embeddings(model, messages: list[str], max_window_tokens: int = 7000):
    """Compute per-message embeddings via late chunking.

    Joins messages into windows ≤ max_window_tokens, runs one forward pass
    per window (messages attend to each other), then mean-pools token
    embeddings for each message's span.

    Returns: np.ndarray of shape [len(messages), embedding_dim]
    """
    import torch
    import numpy as np

    tokenizer = model.tokenizer

    # Token counts per message (without special tokens)
    msg_token_counts = []
    for msg in messages:
        ids = tokenizer.encode(msg, add_special_tokens=False)
        msg_token_counts.append(len(ids))

    # Build windows — each window holds as many messages as fit
    windows = []  # list of (start_idx, end_idx)
    i = 0
    while i < len(messages):
        win_tokens = 0
        j = i
        # If a single message exceeds max_window_tokens, it gets its own window
        # and will be truncated to max_window_tokens by the tokenizer
        if msg_token_counts[i] > max_window_tokens:
            windows.append((i, i + 1))
            i += 1
            continue
        while j < len(messages):
            needed = msg_token_counts[j] + 1  # +1 for newline separator
            if win_tokens + needed > max_window_tokens and j > i:
                break
            win_tokens += needed
            j += 1
        windows.append((i, j))
        i = j

    all_embeddings = [None] * len(messages)

    for win_start, win_end in windows:
        win_messages = messages[win_start:win_end]
        win_text = '\n'.join(win_messages)

        # One forward pass for the entire window → token embeddings
        token_embeds = model.encode(
            [win_text],
            output_value='token_embeddings',
            show_progress_bar=False,
        )[0]  # torch.Tensor [seq_len, hidden_dim]

        # Mean-pool per message: find each message's token span
        pos = 1  # skip CLS token
        for k, msg in enumerate(win_messages):
            msg_len = msg_token_counts[win_start + k]
            end = min(pos + msg_len, token_embeds.shape[0])

            if pos < token_embeds.shape[0]:
                all_embeddings[win_start + k] = (
                    token_embeds[pos:end].mean(dim=0).cpu().numpy()
                )

            pos = end + 1  # +1 for newline separator

    valid = [e for e in all_embeddings if e is not None]
    if not valid:
        return np.array([])
    return np.array(valid)


def segment_chat(source_id: str, path: Path) -> list[DocSectionRecord]:
    import numpy as np
    import hdbscan

    lines = _read_lines(path)
    if not lines: return []

    # Parse messages — MSG_RE matches "Speaker YYYY-MM-DD HH:MM" at line start
    messages: list[dict] = []
    cur_speaker, cur_dt, cur_body, cur_start = "", "", [], 1
    for line_no, line in enumerate(lines, start=1):
        m = MSG_RE.match(line)
        if m:
            if cur_body:
                messages.append({"speaker": cur_speaker, "dt": cur_dt, "body": cur_body, "start": cur_start, "end": line_no - 1})
            cur_speaker, cur_dt = m.group(1).strip(), m.group(2).strip()
            cur_body, cur_start = [], line_no
        else:
            cur_body.append(line)
    if cur_body:
        messages.append({"speaker": cur_speaker, "dt": cur_dt, "body": cur_body, "start": cur_start, "end": len(lines)})

    if len(messages) < 5:
        return []

    # Format messages for embedding
    formatted = [f"{m['speaker']}: {' '.join(m['body']).strip()}" for m in messages]

    # Late-chunk embedding: context-aware per-message embeddings
    try:
        model = _get_embed_model()
        embeddings = _late_chunk_embeddings(model, formatted)
        if len(embeddings) == 0:
            print(f"  chat [{source_id}]: embedding returned 0 — SKIPPED", flush=True)
            return []
    except Exception as e:
        # Surface the failure instead of silently dropping the entire chat source.
        print(f"  chat [{source_id}]: embedding failed ({type(e).__name__}: {e}) — SKIPPED", flush=True)
        return []

    ms = max(5, min(15, len(formatted) // 5))
    clusterer = hdbscan.HDBSCAN(min_cluster_size=ms, min_samples=3, metric="euclidean")
    labels = clusterer.fit_predict(embeddings)

    clean = list(labels)
    for i in range(len(clean)):
        if clean[i] == -1:
            for j in range(i + 1, len(clean)):
                if clean[j] != -1: clean[i] = clean[j]; break
            if clean[i] == -1:
                for j in range(i - 1, -1, -1):
                    if clean[j] != -1: clean[i] = clean[j]; break

    sections: list[DocSectionRecord] = []
    idx, prev, seg_start = 0, None, 0
    for i, label in enumerate(clean):
        if label != prev and prev is not None:
            seg_msgs = messages[seg_start:i]
            if len(seg_msgs) >= 2:
                msg_ranges = [{"speaker": m["speaker"], "dt": m["dt"], "start": m["start"], "end": m["end"]} for m in seg_msgs]
                content = "\n".join(f"{m['speaker']}: {' '.join(m['body']).strip()}" for m in seg_msgs)
                sections.append(_doc_section(source_id, idx,
                    _chat_name(seg_msgs), "embedding_seg",
                    seg_msgs[0]["start"], seg_msgs[-1]["end"],
                    people=list({m["speaker"] for m in seg_msgs}),
                    msg_count=len(seg_msgs),
                    time_range=f"{seg_msgs[0]['dt'][:16]} – {seg_msgs[-1]['dt'][:16]}" if seg_msgs[0]["dt"] and seg_msgs[-1]["dt"] else "",
                    msg_ranges=msg_ranges, content=content))
                idx += 1
            seg_start = i
        prev = label
    if seg_start < len(messages):
        seg_msgs = messages[seg_start:]
        if len(seg_msgs) >= 2:
            msg_ranges = [{"speaker": m["speaker"], "dt": m["dt"], "start": m["start"], "end": m["end"]} for m in seg_msgs]
            content = "\n".join(f"{m['speaker']}: {' '.join(m['body']).strip()}" for m in seg_msgs)
            sections.append(_doc_section(source_id, idx,
                _chat_name(seg_msgs), "embedding_seg",
                seg_msgs[0]["start"], seg_msgs[-1]["end"],
                people=list({m["speaker"] for m in seg_msgs}),
                msg_count=len(seg_msgs),
                time_range=f"{seg_msgs[0]['dt'][:16]} – {seg_msgs[-1]['dt'][:16]}" if seg_msgs[0]["dt"] and seg_msgs[-1]["dt"] else "",
                msg_ranges=msg_ranges, content=content))

    print(f"  chat: {len(messages)} msgs → {len(sections)} segments", flush=True)
    return sections


def _chat_name(msgs: list[dict]) -> str:
    """Initial name for a chat segment — domain/asset filled in by classifier later."""
    speaker = msgs[0]["speaker"]
    dt = msgs[0]["dt"][:16] if msgs[0]["dt"] else ""
    for m in msgs:
        body = " ".join(m["body"]).strip()
        if len(body) > 10:
            return f"[{dt}] {speaker}: {body[:120]}"
    return f"[{dt}] {speaker}"


# ── Config ──

def segment_config(source_id: str, path: Path) -> list[DocSectionRecord]:
    import csv
    from io import StringIO
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".csv":
        try:
            rows = list(csv.reader(StringIO(text)))
            if rows:
                name = f"{path.stem}: {', '.join(rows[0][:6])} ({len(rows)-1} rows)"
                return [_doc_section(source_id, 0, name, "config_table", 1, len(text.splitlines()), content=text[:5000])]
        except Exception: pass
    return segment_fallback(source_id, path)


# ── DOCX (style-aware, structure-preserving) ──

# Patterns for detecting section boundaries in weekly report DOCX files
_PERSON_NAME_RE = re.compile(r"^[一-鿿]{2,4}$")  # 2-4 Chinese chars = likely name
_SECTION_HEADER_RE = re.compile(r"(本周|上周|下周|这周|工作总结|工作内容|工作进展|本周总结|本周内容|本周工作|上周总结|下周计划|下周工作|工作计划|工作安排|工作总结|任务进展|项目进展)")

def _get_font_pt(paragraph) -> float | None:
    """Extract font size in points from the first run with size info."""
    for run in paragraph.runs:
        if run.font.size:
            # python-docx stores size in EMU (English Metric Units)
            # 1 pt = 12700 EMU
            return run.font.size / 12700
    return None


def _is_likely_person(text: str) -> bool:
    """Check if a line looks like a person's name using structural heuristics only.

    No hard-coded name list. Relies on document structure:
    A short Chinese-character line immediately before a section header
    in a DOCX weekly report is almost certainly a person name.
    """
    if _SECTION_HEADER_RE.search(text):
        return False
    if ":" in text or "：" in text:
        return False
    if any(w in text for w in _NONPERSON_WORDS):
        return False
    # Short Chinese name (2-4 chars) — the structural heuristic
    if _PERSON_NAME_RE.match(text):
        return True
    return False


def _is_section_header(text: str) -> bool:
    """Check if a line is a weekly report section header.

    Must be a short line where the header phrase is the primary content,
    not buried in a longer sentence.
    """
    if not _SECTION_HEADER_RE.search(text):
        return False
    # The header phrase should dominate the line — either short overall,
    # or the match is at the beginning
    if len(text) <= 15:
        return True
    # Check if it starts with a section phrase
    if _SECTION_HEADER_RE.match(text):
        return True
    # Longer line with header phrase buried inside → not a real header
    return False


def segment_docx(source_id: str, path: Path) -> list[DocSectionRecord]:
    """Parse .docx weekly reports preserving person→section→items hierarchy.

    These reports follow a consistent pattern:
        [Person Name]
        本周总结/本周内容：
        - item 1
        - item 2
        下周计划：
        - item 1
        ...

    Each person+section combo becomes one DocSectionRecord with full content.
    """
    import docx as docx_lib

    try:
        doc = docx_lib.Document(str(path))
    except Exception:
        return segment_fallback(source_id, path)

    # Collect non-empty paragraphs with metadata
    entries: list[dict] = []  # {text, font_pt, is_bold, line_no}
    line_no = 1
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            font_pt = _get_font_pt(p)
            is_bold = any(run.bold for run in p.runs if run.bold)
            entries.append({"text": text, "font_pt": font_pt, "bold": is_bold, "line": line_no})
        line_no += 1

    if not entries:
        return []

    # ── Inline embedded PDF content into entries ──
    embedded_texts = _extract_embedded_pdf_texts(path)
    if embedded_texts:
        # Insert embedded text as content entries at the OLE paragraph position
        new_entries = []
        for line_num in sorted(embedded_texts.keys()):
            # Split the extracted text into pseudo-paragraphs and insert after the OLE line
            pdf_lines = embedded_texts[line_num].split('\n')
            for offset, pdf_line in enumerate(pdf_lines):
                pdf_line = pdf_line.strip()
                if pdf_line:
                    new_entries.append((line_num + offset * 0.001, {
                        "text": pdf_line, "font_pt": None, "bold": False,
                        "line": line_num, "role_note": "embedded"
                    }))

        # Merge new entries into the main list, maintaining line order
        if new_entries:
            all_entries = [(e["line"], e) for e in entries]
            all_entries.extend(new_entries)
            all_entries.sort(key=lambda x: x[0])
            entries = [e for _, e in all_entries]

    # ── Phase 1: identify structural boundaries ──
    # Classify each entry: person_name, section_header, content, or standalone_header
    typed: list[dict] = []
    for e in entries:
        t = e.copy()
        text = e["text"]
        if _is_likely_person(text):
            t["role"] = "person"
        elif _is_section_header(text):
            t["role"] = "section_header"
        elif e.get("bold") or (e.get("font_pt") and e["font_pt"] >= 14):
            # Bold or large-font line = likely header-like
            t["role"] = "standalone_header"
        else:
            t["role"] = "content"
        typed.append(t)

    # ── Phase 2: build sections ──
    sections: list[DocSectionRecord] = []
    idx = 0
    current_person = ""
    current_section = ""
    current_lines: list[str] = []
    current_start = 1
    current_people: list[str] = []

    # Detect if a line looks like content (starts with number, bullet, or is long)
    def _is_content_line(text: str) -> bool:
        """Lines starting with numbers, bullets, or very long text are content, not headers."""
        if len(text) > 40:
            return True
        if re.match(r"^\d+[\.\)、）]", text):
            return True
        if re.match(r"^[\(（]?\d+[\)）]?", text):
            return True
        if text.startswith(("+", "-", "•", "·", "a.", "b.", "c.", "d.")):
            return True
        return False

    def _flush():
        nonlocal idx, current_lines, current_start, current_people
        if not current_lines:
            return
        content = "\n".join(current_lines)
        label_parts = []
        if current_person:
            label_parts.append(current_person)
        section_label = current_section if current_section else "本周工作"
        label_parts.append(section_label)
        name = ": ".join(label_parts) if label_parts else path.stem
        spath = [current_person, section_label] if current_person else [name]
        sections.append(_doc_section(
            source_id, idx, name, "weekly_section",
            current_start, current_start + len(current_lines) - 1,
            spath=[p for p in spath if p],
            people=list(current_people),
            content=content,
        ))
        idx += 1
        current_lines = []
        current_start = 1
        current_people = []

    for t in typed:
        if t["role"] == "person":
            _flush()
            current_person = t["text"]
            current_section = ""
            current_start = t["line"]
            current_people = {current_person}
            continue

        if t["role"] in ("section_header", "standalone_header"):
            if _is_content_line(t["text"]):
                # This looks like content despite formatting — accumulate it
                current_lines.append(t["text"])
                if not current_section and current_person:
                    current_section = "本周工作"
                    current_start = t["line"]
                continue
            # Real section header
            if current_lines and current_section:
                _flush()
                current_people = {current_person} if current_person else set()
            current_section = t["text"].rstrip("：:。.")
            current_start = t["line"]
            if not current_person:
                if _is_likely_person(t["text"]):
                    current_person = t["text"]
                    current_section = ""
                    current_people = {current_person}
                else:
                    current_person = t["text"][:20]
                    current_people = set()
            continue

        # content line
        if not current_person and not current_section:
            current_section = t["text"][:80] if len(t["text"]) > 40 else t["text"]
            current_start = t["line"]
        current_lines.append(t["text"])

    _flush()

    if not sections:
        return segment_fallback(source_id, path)

    # ── Extract embedded PDFs from OLE objects ──
    # Map each OLE paragraph to its enclosing section's person
    ole_person_map = {}  # rId → person name
    for t in typed:
        # After flush, current_person is reset; track OLE paragraph positions
        pass  # We'll use the sections' line ranges instead

    # Build a map: paragraph line number → the section it belongs to
    def _find_section_person(line_num):
        for s in sections:
            start = s.locator.get('start_line', 0)
            end = s.locator.get('end_line', 0)
            if start <= line_num <= end:
                return s.people[0] if s.people else None
        return None

    embedded_sections = _extract_embedded_objects(source_id, path, idx, _find_section_person)
    if embedded_sections:
        sections.extend(embedded_sections)
        idx += len(embedded_sections)

    print(f"  docx: {len(entries)} paragraphs → {len(sections)} sections"
          f"{f' (+{len(embedded_sections)} embedded)' if embedded_sections else ''}", flush=True)
    return sections


# ── Embedded PDF text extraction (inlines into DOCX entries) ──

def _extract_embedded_pdf_texts(path: Path) -> dict[int, str]:
    """Extract text from embedded PDFs in a DOCX, keyed by paragraph line number.

    Returns dict: line_number → extracted_text for inline insertion.
    """
    import zipfile, io as _io, re as _re

    _MAX_PAGES = 80
    _MAX_PAGE_CHARS = 50000
    result = {}

    try:
        with zipfile.ZipFile(path) as zf:
            rels_path = 'word/_rels/document.xml.rels'
            if rels_path not in zf.namelist():
                return result

            rels_xml = zf.read(rels_path).decode('utf-8', errors='ignore')
            ole_targets = {}
            for m in _re.finditer(r'<Relationship[^>]*/>', rels_xml):
                rtype = _re.search(r'Type="([^"]*)"', m.group())
                rid = _re.search(r'Id="([^"]*)"', m.group())
                target = _re.search(r'Target="([^"]*)"', m.group())
                if rtype and rid and target and 'oleObject' in rtype.group(1):
                    ole_targets[rid.group(1)] = f"word/{target.group(1)}"

            if not ole_targets:
                return result

            doc_xml = zf.read('word/document.xml').decode('utf-8', errors='ignore')
            ole_lines = {}
            line_num = 0
            for para in _re.finditer(r'<w:p[ >].*?</w:p>', doc_xml, _re.DOTALL):
                line_num += 1
                if 'OLEObject' in para.group():
                    # Find ALL rIds (OLE paragraphs also contain an image rId)
                    for rid_match in _re.finditer(r'r:id="(rId\d+)"', para.group()):
                        if rid_match.group(1) in ole_targets:
                            ole_lines[line_num] = rid_match.group(1)
                            break

            if not ole_lines:
                return result

            for line_num, rid in ole_lines.items():
                emb_path = ole_targets[rid]
                if emb_path not in zf.namelist():
                    continue
                try:
                    ole_data = zf.read(emb_path)
                    import olefile
                    ole = olefile.OleFileIO(_io.BytesIO(ole_data))
                    pdf_data = None
                    if ['\x01Ole10Native'] in ole.listdir():
                        native = ole.openstream('\x01Ole10Native').read()
                        pdf_magic = native.find(b'%PDF')
                        if pdf_magic >= 0:
                            pdf_data = bytes(native[pdf_magic:])
                    ole.close()
                    if not pdf_data:
                        continue

                    import fitz
                    doc = fitz.open(stream=pdf_data, filetype='pdf')
                    page_count = min(doc.page_count, _MAX_PAGES)
                    pages_text = []
                    for page_num in range(page_count):
                        try:
                            text = doc.load_page(page_num).get_text()
                        except Exception:
                            continue
                        if len(text) > _MAX_PAGE_CHARS:
                            text = text[:_MAX_PAGE_CHARS]
                        if text.strip():
                            pages_text.append(text.strip())
                    doc.close()
                    if pages_text:
                        result[line_num] = "\n\n".join(pages_text)
                except Exception as e:
                    print(f"    embedded PDF failed: {e}", flush=True)
                    continue
            return result
    except Exception as e:
        print(f"  embedded extraction error: {e}", flush=True)
        return result


# ── (Legacy) Embedded object extraction ──

def _extract_embedded_objects(source_id: str, path: Path, start_idx: int,
                              find_person: callable = None) -> list:
    """Extract embedded PDF files from OLE objects in a DOCX.

    Reads word/_rels/document.xml.rels to find OLE object references,
    extracts the raw PDF stream from each OLE binary using olefile,
    then parses with pymupdf (fitz) page-by-page.

    Args:
        source_id: Source identifier for section IDs
        path: Path to the DOCX file
        start_idx: Starting index for section numbering
        find_person: callable(paragraph_line_num) → person name or None.
                     Used to propagate person context from the enclosing
                     DOCX section to the embedded PDF sections.

    Robustness guards:
      - Max 80 pages per embedded PDF
      - Max 50K chars per page
      - Skips non-PDF / unparseable objects
    """
    import zipfile, io as _io

    _MAX_PAGES = 80
    _MAX_PAGE_CHARS = 50000

    try:
        with zipfile.ZipFile(path) as zf:
            # Find OLE relationships
            rels_path = 'word/_rels/document.xml.rels'
            if rels_path not in zf.namelist():
                return []

            import xml.etree.ElementTree as ET
            rels_xml = zf.read(rels_path)
            rels_root = ET.fromstring(rels_xml)
            ns = '{http://schemas.openxmlformats.org/package/2006/relationships}'

            ole_targets = {}
            for rel in rels_root:
                rtype = rel.get('Type', '')
                if 'oleObject' in rtype:
                    rid = rel.get('Id', '')
                    target = rel.get('Target', '')
                    if target.startswith('embeddings/'):
                        ole_targets[rid] = target

            if not ole_targets:
                return []

            # Find which paragraphs reference these OLE objects and their line numbers
            doc_xml = zf.read('word/document.xml')
            doc_root = ET.fromstring(doc_xml)
            r_ns = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'

            # Map paragraph line numbers → rIds by scanning the raw XML for OLE paragraphs
            import re
            doc_text = doc_xml.decode('utf-8', errors='ignore') if isinstance(doc_xml, bytes) else doc_xml
            ole_line_map = {}  # rId → line_number
            para_pattern = re.compile(r'<w:p[ >].*?</w:p>', re.DOTALL)
            line_num = 0
            for para in para_pattern.finditer(doc_text):
                line_num += 1
                p_text = para.group()
                if 'OLEObject' in p_text:
                    rid_match = re.search(r'r:id="(rId\d+)"', p_text)
                    if rid_match:
                        rid = rid_match.group(1)
                        if rid in ole_targets:
                            ole_line_map[rid] = line_num

            if not ole_line_map:
                return []

            ole_refs = set(ole_line_map.keys())

            # Extract and parse each embedded object
            sections = []
            for rid in ole_refs:
                emb_path = f'word/{ole_targets[rid]}'
                if emb_path not in zf.namelist():
                    continue

                ole_data = zf.read(emb_path)

                try:
                    import olefile
                    ole = olefile.OleFileIO(_io.BytesIO(ole_data))
                    # The embedded file is typically in the '\x01Ole10Native' stream
                    # or as a package in the root storage
                    streams = ole.listdir()
                    pdf_data = None

                    # Try to find the PDF content
                    # Ole10Native stream contains the raw embedded file
                    if ['\x01Ole10Native'] in streams:
                        native = ole.openstream('\x01Ole10Native').read()
                        # Ole10Native format: 4 bytes size + embedded file data
                        # The actual file starts after a variable-length header
                        # Look for PDF magic
                        pdf_magic = native.find(b'%PDF')
                        if pdf_magic >= 0:
                            pdf_data = native[pdf_magic:]
                    if not pdf_data:
                        # Try reading the CONTENTS stream
                        try:
                            content = ole.openstream('CONTENTS').read()
                            pdf_magic = content.find(b'%PDF')
                            if pdf_magic >= 0:
                                pdf_data = content[pdf_magic:]
                        except:
                            pass
                    if not pdf_data:
                        # Try package approach: look for any stream containing PDF
                        for stream_path in streams:
                            try:
                                data = ole.openstream(stream_path).read()
                                if b'%PDF' in data:
                                    pdf_data = data[data.find(b'%PDF'):]
                                    break
                            except:
                                continue

                    if not pdf_data:
                        ole.close()
                        continue

                    # Parse PDF with pymupdf page-by-page
                    # Copy bytes to ensure they survive OLE/zip closure
                    pdf_bytes = bytes(pdf_data)
                    import fitz
                    doc = fitz.open(stream=pdf_bytes, filetype='pdf')
                    page_count = min(doc.page_count, _MAX_PAGES)

                    for page_num in range(page_count):
                        try:
                            page = doc.load_page(page_num)
                            text = page.get_text()
                        except Exception:
                            continue
                        if len(text) > _MAX_PAGE_CHARS:
                            text = text[:_MAX_PAGE_CHARS] + f"\n... [truncated, original page {len(text)} chars]"
                            print(f"    embedded PDF page {page_num+1} truncated ({_MAX_PAGE_CHARS} chars)", flush=True)

                        if text.strip():
                            name = f"[embedded] {path.stem} p{page_num+1}"
                            # Get person context from the enclosing DOCX paragraph
                            emb_people = []
                            if find_person and rid in ole_line_map:
                                person = find_person(ole_line_map[rid])
                                if person:
                                    emb_people = [person]
                            sections.append(_doc_section(
                                source_id, start_idx + len(sections),
                                name, "font_heading_block",
                                page_num + 1, page_num + 1,
                                spath=[f"[embedded PDF {page_num+1}]"],
                                people=emb_people,
                                content=text,
                            ))

                    if doc.page_count > _MAX_PAGES:
                        print(f"    embedded PDF truncated: {doc.page_count} → {_MAX_PAGES} pages", flush=True)
                    doc.close()
                    ole.close()

                except Exception as e:
                    print(f"    embedded {emb_path}: extraction failed ({e})", flush=True)
                    continue

            return sections

    except Exception as e:
        print(f"  embedded extraction error: {e}", flush=True)
        return []


# ── PDF (font-aware, heading-preserving) ──

def segment_pdf(source_id: str, path: Path) -> list[DocSectionRecord]:
    """Parse PDF files preserving heading hierarchy from font size and positioning.

    Uses pymupdf's dict-based extraction to access font metadata.
    Groups text lines by y-coordinate, detects headings by larger font size
    or bold text, and builds hierarchical sections.
    """
    try:
        import fitz
        doc = fitz.open(str(path))
    except Exception:
        return segment_fallback(source_id, path)

    # Collect all text lines across all pages with their formatting
    lines: list[dict] = []  # {text, font_size, is_bold, x, y, page}
    MIN_FONT_FOR_CONTENT = 7.0  # ignore tiny fonts (footers, page numbers)

    for page_idx, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                # Collect all spans in this line
                spans = line.get("spans", [])
                if not spans:
                    continue

                # Merge spans into one text line, tracking max font size
                text_parts = []
                max_size = 0.0
                has_bold = False
                min_x = float("inf")

                for span in spans:
                    t = span.get("text", "").strip()
                    if t:
                        text_parts.append(t)
                    sz = span.get("size", 0)
                    if sz > max_size:
                        max_size = sz
                    if "Bold" in span.get("font", ""):
                        has_bold = True
                    x0 = span.get("bbox", [0, 0, 0, 0])[0]
                    if x0 < min_x:
                        min_x = x0

                full_text = "".join(text_parts).strip()
                if not full_text or max_size < MIN_FONT_FOR_CONTENT:
                    continue

                y_pos = round(spans[0].get("bbox", [0, 0, 0, 0])[1])
                lines.append({
                    "text": full_text,
                    "font_size": max_size,
                    "bold": has_bold,
                    "x": round(min_x),
                    "y": y_pos,
                    "page": page_idx,
                })

    doc.close()

    if not lines:
        return segment_fallback(source_id, path)

    # ── Determine the "body" font size ──
    # The most common font size is the body; significantly larger = heading
    from collections import Counter
    size_counts = Counter(round(l["font_size"], 1) for l in lines)
    if size_counts:
        body_size = size_counts.most_common(1)[0][0]
    else:
        body_size = 12.0

    # ── Classify each line ──
    BULLET_CHARS = {"●", "○", "•", "·", "", "", "", "", "ii.", "iii.", "iv.", "v."}
    for l in lines:
        text = l["text"].strip()
        sz = round(l["font_size"], 1)

        # Filter out lines that look like headings but are actually bullets/garbled
        is_bullet = (
            len(text) <= 5
            and (text in BULLET_CHARS or text.rstrip(".").isascii() and len(text.rstrip(".")) <= 3)
        )

        if is_bullet:
            l["role"] = "body"  # force body — these are garbled bullet points
        elif sz >= body_size + 2.0 or l["bold"]:
            l["role"] = "heading"
        elif sz >= body_size + 1.0:
            l["role"] = "subheading"
        else:
            l["role"] = "body"

    # ── Group lines into sections ──
    sections: list[DocSectionRecord] = []
    idx = 0
    current_heading = ""
    current_lines: list[str] = []
    current_start = 1
    seen_headings: set[str] = set()

    for l in lines:
        if l["role"] in ("heading", "subheading"):
            # Flush previous section
            if current_lines:
                content = "\n".join(current_lines)
                name = current_heading if current_heading else path.stem
                sections.append(_doc_section(
                    source_id, idx, name, "font_heading_block",
                    current_start, current_start + len(current_lines) - 1,
                    spath=[name],
                    content=content,
                ))
                idx += 1
                current_lines = []

            heading_text = l["text"]
            # Avoid duplicate headings (e.g., same header on multiple pages)
            if heading_text in seen_headings:
                heading_text = f"{heading_text} (p{l['page']+1})"
            seen_headings.add(heading_text)
            current_heading = heading_text
            current_start = l["y"]
        else:
            current_lines.append(l["text"])

    # Flush last section
    if current_lines:
        content = "\n".join(current_lines)
        name = current_heading if current_heading else path.stem
        sections.append(_doc_section(
            source_id, idx, name, "font_heading_block",
            current_start, current_start + len(current_lines) - 1,
            spath=[name],
            content=content,
        ))

    # ── Post-processing: merge orphaned sections ──
    # Small sections get merged into the previous section (thresholds at top of file)

    merged: list[DocSectionRecord] = []
    for s in sections:
        body_lines = [l for l in s.content.split("\n") if l.strip()]
        is_small = len(s.content) < MIN_SECTION_CHARS or len(body_lines) < MIN_SECTION_LINES
        if merged and is_small:
            # Merge into previous section
            prev = merged[-1]
            prev.content = prev.content + "\n" + s.content
            prev.locator["end_line"] = s.locator.get("end_line", prev.locator.get("start_line", 1))
            prev.name = prev.name  # keep the parent heading
        else:
            merged.append(s)

    # Fall back if result is extreme: too few sections (undetected structure)
    # or too many (over-fragmentation from bold-heavy group reports)
    if len(merged) < 2 or len(merged) > 30:
        # Try pymupdf4llm → markdown conversion as intermediate fallback
        md_sections = _pdf_via_markdown(source_id, path)
        if md_sections:
            print(f"  pdf: {len(lines)} lines → pymupdf4llm→md ({len(md_sections)} sections)", flush=True)
            return md_sections
        print(f"  pdf: fallback ({len(merged)} sections, body_size={body_size}pt)", flush=True)
        return segment_fallback(source_id, path)

    print(f"  pdf: {len(lines)} lines → {len(sections)}→{len(merged)} sections (body_size={body_size}pt)", flush=True)
    return merged


def _pdf_via_markdown(source_id: str, path: Path) -> list[DocSectionRecord] | None:
    """Try pymupdf4llm → segment_markdown as intermediate PDF fallback."""
    try:
        import pymupdf4llm
        import tempfile, os
        md_text = pymupdf4llm.to_markdown(str(path))
        if len(md_text) < 100:  # too little content extracted
            return None
        # Write to temp file and feed to our markdown segmenter
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(md_text)
            tmp_path = f.name
        try:
            sections = segment_markdown(source_id, Path(tmp_path))
            return sections if len(sections) >= 2 else None
        finally:
            os.unlink(tmp_path)
    except ImportError:
        return None
    except Exception:
        return None


# ── Fallback ──

def segment_fallback(source_id: str, path: Path) -> list[DocSectionRecord]:
    lines = _read_lines(path)
    if not lines: return []
    sections: list[DocSectionRecord] = []
    start_line, idx, in_block = 1, 0, False
    for line_no, line in enumerate(lines, start=1):
        if idx >= MAX_FALLBACK_SECTIONS: break
        if line.strip() and not in_block: start_line, in_block = line_no, True
        elif not line.strip() and in_block:
            content = "\n".join(lines[start_line - 1 : line_no - 1])
            sections.append(_doc_section(source_id, idx, f"{path.stem}:{start_line}", "text_block", start_line, line_no - 1, content=content))
            idx += 1; in_block = False
    if in_block and idx < MAX_FALLBACK_SECTIONS:
        content = "\n".join(lines[start_line - 1 :])
        sections.append(_doc_section(source_id, idx, f"{path.stem}:{start_line}", "text_block", start_line, len(lines), content=content))
    return sections


# ── Dispatcher ──

def build_section_inventory(sources: list[SourceRecord]) -> list[DocSectionRecord]:
    """Segment all non-code sources into DocSectionRecord."""
    doc_secs: list[DocSectionRecord] = []
    for source in sources:
        path = ROOT / source.path
        if source.source_type == "chat":
            doc_secs.extend(segment_chat(source.id, path))
        elif source.source_type == "code_file":
            continue
        elif source.source_type == "config":
            doc_secs.extend(segment_config(source.id, path))
        elif source.source_type == "weekly_report":
            # LLM-based segmentation for weekly reports (optional, with checkpoint)
            from util.paths import _SETTINGS
            if _SETTINGS.get("llm_seg_enabled", False):
                from util.llm_seg import segment_llm, load_llm_seg_checkpoint, save_llm_seg_checkpoint
                cached = load_llm_seg_checkpoint(source.id)
                if cached:
                    from util.ontology import DocSectionRecord
                    doc_secs.extend([DocSectionRecord(**c) for c in cached])
                    print(f"  LLM_seg: {path.name} → {len(cached)} sections (checkpoint)", flush=True)
                else:
                    sections = segment_llm(source.id, path, idx_offset=len(doc_secs))
                    if sections:
                        doc_secs.extend(sections)
                        save_llm_seg_checkpoint(source.id, sections)
                    else:
                        if path.suffix.lower() == ".pdf":
                            doc_secs.extend(segment_pdf(source.id, path))
                        else:
                            doc_secs.extend(segment_docx(source.id, path))
            else:
                # LLM_seg disabled — use format-based segmenters
                if path.suffix.lower() == ".pdf":
                    doc_secs.extend(segment_pdf(source.id, path))
                else:
                    doc_secs.extend(segment_docx(source.id, path))
        elif path.suffix.lower() == ".pdf":
            # Single-page check: don't segment 1-page PDFs
            try:
                import fitz
                doc = fitz.open(str(path))
                page_count = doc.page_count
                doc.close()
                if page_count <= 1:
                    doc_secs.extend(segment_fallback(source.id, path))
                else:
                    doc_secs.extend(segment_pdf(source.id, path))
            except Exception:
                doc_secs.extend(segment_pdf(source.id, path))
        elif path.suffix.lower() == ".docx":
            # Single-page check: estimate from paragraph count
            try:
                import docx
                doc = docx.Document(str(path))
                para_count = sum(1 for p in doc.paragraphs if p.text.strip())
                if para_count <= 20:  # roughly 1 page
                    doc_secs.extend(segment_fallback(source.id, path))
                else:
                    doc_secs.extend(segment_docx(source.id, path))
            except Exception:
                doc_secs.extend(segment_docx(source.id, path))
        elif path.suffix.lower() in {".md"}:
            doc_secs.extend(segment_markdown(source.id, path))
        else:
            doc_secs.extend(segment_fallback(source.id, path))
    # ── Post-processing: backfill people + iso_week from section metadata ──
    import re as _re
    from datetime import datetime as _dt
    source_by_id = {s.id: s for s in sources}

    # Known non-person section headers (avoid false positives)
    # ── Extract source-level metadata from filenames ──
    # Person: "张三-工作周报0117.pdf" or "李四 5.28 周报.pdf"
    # Week:   "原始周报(2025.1.13-2025.1.17).docx" or "0117" (MMDD)
    import os as _os
    _source_person = {}
    _source_week = {}
    for src in sources:
        fname = Path(src.path).stem
        # Person from filename prefix (2-4 Chinese chars at start)
        pm = _re.match(r'^([一-鿿]{2,4})[\s\-_\-]', fname)
        if pm:
            _source_person[src.id] = pm.group(1)

        # Year inference helper: use file mtime year as fallback for M.DD-only patterns
        def _get_file_year() -> int | None:
            fp = ROOT / src.path
            try:
                return _dt.fromtimestamp(_os.path.getmtime(fp)).year
            except (OSError, ValueError):
                return None

        week = None

        # Pattern 1: (YYYY.M.D-YYYY.M.D) — e.g., 原始周报(2025.1.13-2025.1.17).docx
        wm = _re.search(r'\((\d{4})\.(\d{1,2})\.(\d{1,2})-(\d{4})\.(\d{1,2})\.(\d{1,2})\)', src.path)
        if wm:
            try:
                week = _dt(int(wm.group(4)), int(wm.group(5)), int(wm.group(6))).strftime("%Y-W%V")
            except ValueError:
                pass

        # Pattern 2: YYYYMMDD-YYYYMMDD — e.g., 20250113-20250117
        if not week:
            wm = _re.search(r'(\d{4})(\d{2})(\d{2})-(\d{4})(\d{2})(\d{2})', src.path)
            if wm:
                try:
                    week = _dt(int(wm.group(4)), int(wm.group(5)), int(wm.group(6))).strftime("%Y-W%V")
                except ValueError:
                    pass

        # Pattern 3: MMDD-MMDD range — e.g., 王五-周报-0524-0530
        if not week:
            wm = _re.search(r'(\d{2})(\d{2})-(\d{2})(\d{2})', src.path)
            if wm:
                yr = _get_file_year()
                if yr:
                    try:
                        week = _dt(yr, int(wm.group(3)), int(wm.group(4))).strftime("%Y-W%V")
                    except ValueError:
                        pass

        # Pattern 4: M.DD-M.DD range — e.g., 黄嘉璇 2.17-2.21, 李昊1.12-1.17
        if not week:
            wm = _re.search(r'(\d{1,2})\.(\d{1,2})[-_](\d{1,2})\.(\d{1,2})', src.path)
            if wm:
                yr = _get_file_year()
                if yr:
                    try:
                        week = _dt(yr, int(wm.group(3)), int(wm.group(4))).strftime("%Y-W%V")
                    except ValueError:
                        pass

        # Pattern 5: Single YYYYMMDD — e.g., 李四 周报 20250510
        if not week:
            wm = _re.search(r'(\d{4})(\d{2})(\d{2})', src.path)
            if wm and not _re.search(r'(\d{4})(\d{2})(\d{2})-', src.path):
                # Only match standalone YYYYMMDD, not ranges (caught by pattern 2)
                try:
                    week = _dt(int(wm.group(1)), int(wm.group(2)), int(wm.group(3))).strftime("%Y-W%V")
                except ValueError:
                    pass

        # Pattern 6: M.DD (single date, with year from mtime) — e.g., 赵六 5.9周报, 李四 5.28
        if not week:
            wm = _re.search(r'(\d{1,2})\.(\d{1,2})', src.path)
            if wm:
                yr = _get_file_year()
                if yr:
                    try:
                        week = _dt(yr, int(wm.group(1)), int(wm.group(2))).strftime("%Y-W%V")
                    except ValueError:
                        pass

        # Pattern 7: MMDD suffix — e.g., 张三-工作周报0117
        if not week:
            wm = _re.search(r'(\d{2})(\d{2})(?:\.\w+)?$', fname)
            if wm:
                yr = _get_file_year()
                if yr:
                    try:
                        week = _dt(yr, int(wm.group(1)), int(wm.group(2))).strftime("%Y-W%V")
                    except ValueError:
                        pass

        if week:
            _source_week[src.id] = week

    _NON_PERSON = {
        "本周内容", "本周工作", "本周总结", "本周小结", "本周计划",
        "下周计划", "下周工作", "本周主要工作", "本周工作总结",
        "本周主要完成了以下工作", "周报", "周总结",
    }

    for ds in doc_secs:
        src = source_by_id.get(ds.source_id)
        # ── People ──
        if not ds.people:
            # Try section name: "李四: 本周总结"
            m = _re.match(r'^([^:：]+)[：:]', ds.name)
            if m:
                name = m.group(1).strip()
                # Only a bare Chinese personal name (2-4 hanzi) is a plausible person;
                # config filenames like "labels"/"my_path"/"test" must not become people.
                if name not in _NON_PERSON and _re.fullmatch(r'[一-鿿]{2,4}', name):
                    ds.people = [name]
        if not ds.people and src:
            # Fallback: source-level person from filename
            sp = _source_person.get(ds.source_id)
            if sp:
                ds.people = [sp]

        # ── ISO week ──
        if not ds.iso_week:
            # Chat time_range "2025-01-17 15:30 – ..."
            m = _re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', ds.time_range)
            if m:
                try:
                    ds.iso_week = _dt(int(m.group(1)), int(m.group(2)), int(m.group(3))).strftime("%Y-W%V")
                except ValueError:
                    pass
        if not ds.iso_week and src:
            # Source-level week from filename
            sw = _source_week.get(ds.source_id)
            if sw:
                ds.iso_week = sw

    return doc_secs
