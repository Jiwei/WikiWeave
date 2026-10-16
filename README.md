# WikiWeave

> A config-driven, provenance-anchored pipeline that turns an arbitrary corpus of source materials — **code, docs, chat logs, reports, specs, anything** — into a **linked, self-contained knowledge base**: an OKF-compliant Markdown wiki + structured JSON + navigable HTML with meta ontology schema and mining functions, bridging code intelligence and document intelligence.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

**The self-building knowledge graph for your code and docs.** Rather than hand-curating a
knowledge base or throwing everything into a vector store, WikiWeave **discovers the
structure itself** — from a config-driven ontology template and a fixed set of *meta-functions* —
and mines an arbitrary corpus into a queryable, cross-linked, auditable graph.

---

## What it does

Teams often accumulate knowledge in messy, unstructured form: chat threads, one-off READMEs, weekly
reports, scripts whose "why" lives only in someone's head. **Vector stores store embeddings, not
meaning** — context you can't explain and links you can't trace. WikiWeave turns that corpus
into a *queryable, cross-linked* knowledge graph where:

- **Every entity** (a task, an error-and-fix, a workflow, a
  person) carries a **provenance trail** back to the exact source section that produced it.
- **Code ↔ doc linking** connects source files to the narrative that explains them.
- The output is **OKF-compliant Markdown** — readable by humans *and* loadable by AI agents,
  following [Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
  and Google's Open Knowledge Format (OKF). No database or runtime required to *read* the result.

It is *generic by design*: the entity schema, domain taxonomy, keyword tables, code-domain
vocabulary, and source-type routing are **all data, not code** — re-theming it for a new domain
means editing JSON, not Python.

---

## Why WikiWeave

|                           | Plain RAG / vector store | Hand-curated wiki         | **WikiWeave**                      |
| ------------------------- | ------------------------ | ------------------------- | ---------------------------------- |
| **How links form**        | embedding similarity     | manual, best-effort       | deterministic + LLM, traced        |
| **Provenance**            | none                     | anecdotal                 | every entity → source section      |
| **Entity model**          | none (flat chunks)       | ad hoc                    | config-driven ontology             |
| **Domain adaptation**     | retrain / re-embed       | rewrite docs              | edit JSON (`model_data/`)          |
| **Code ↔ doc bridge**     | no                       | no                        | yes (AST + tree-sitter)            |
| **Auditability**          | "why" is opaque          | "why" is someone's memory | every claim traces to raw material |
| **Agent-readable output** | only via API             | no                        | OKF Markdown, grep-able by anyone  |

---

## Who it's for

- **Platform / infra teams** that want a self-maintaining internal knowledge base over mixed code + docs.
- **R&D teams** (quant, ML, engineering) whose "why" lives in codes, chats and weekly reports, while the knowledge landscape being updated constantly.
- **AI-agent builders** that need a clean, provenance-anchored context layer instead of raw chunks.
- **Anyone inheriting a messy repo** who needs to reconstruct *what exists and why* from the corpus itself.

---

## What WikiWeave gives you

- **Config-driven ontology** — the entity schema is declared in JSON and turned into dataclasses at
  import time; adding an entity type is a JSON edit, never a code change.
- **Fixed meta-functions** — `extract · seed · group · index · synthesize`, a small static vocabulary
  that *mines* discovered entities from the base entities, all dispatched from JSON declarations.
- **Base ↔ discovered bidirectional linking** — five base entities (`Source`, `Domain`, `DocSection`,
  `CodeModule`, `CodeUnit`) are the deterministic foundation; discovered entities link back to them
  via `source_sections` / `domains` / `ref`, and base pages link forward to what they produced.
- **Provenance-anchored** — nothing in the output is asserted without a path back to raw material.
- **Deterministic-first, LLM-enhanced** — every LLM step has a keyword/heuristic fallback, so the
  pipeline *degrades instead of dying* when the model is unavailable.
- **Self-contained output** — OKF Markdown wiki + HTML + structured JSON, browsable in any editor and
  grep-able by humans and agents alike.

---

## Quick start

```bash
# 1. (optional) install per-source-type dependencies
pip install pymupdf pymupdf4llm python-docx tree-sitter-language-pack

# 2. point it at a corpus and run (runs both phases end-to-end by default)
python run.py /path/to/your/corpus
`python run.py --explore` — invokes `claude -p` (headless, `--dangerously-skip-permissions`
  #for non-interactive write-back) with the [`explore_skill`](exploration/explore_skill/SKILL.md).

#   or via env var / config (see Configuration)
KE_INPUT_PATH=/path/to/corpus python run.py
```

## Pipeline

A four-stage pipeline, each stage an isolated package with a narrow contract — any stage can be
replaced or extended independently.

```
                         ┌────────────────────────────────────────────────┐
   raw sources ─────────▶│  1. EXPLORATION          (exploration/)        │
   (any directory)       │     inventory · content-hash dedup ·           │
                         │     data-driven source-type routing .          │
                         │     Agentic discovery of knowledge ontology    │
                         └───────────────────────┬────────────────────────┘
                                                 ▼
                         ┌────────────────────────────────────────────────┐
                         │  2. SEGMENTATION         (segmentation/)       │
                         │     doc → sections · code → units (AST/        │
                         │     tree-sitter) · multi-label domain          │
                         │     classification (LLM + keyword fallback)    │
                         └───────────────────────┬────────────────────────┘
                                                 ▼
                         ┌────────────────────────────────────────────────┐
                         │  3. EXTRACTION           (extraction/)         │
                         │     LLM-structured entities: tasks · errors ·  │
                         │     workflows · protocols · projects ·         │
                         │     artifacts · people (alias-normalized)      │
                         └───────────────────────┬────────────────────────┘
                                                 ▼
                         ┌────────────────────────────────────────────────┐
                         │  4. LINKING + RENDER      (render/)            │
                         │     code↔doc links · entity↔source backlinks · │
                         │     OKF Markdown wiki · HTML · code map ·      │
                         │     call/dependency graphs · glossary          │
                         └────────────────────────────────────────────────┘
```

## License

[GNU Affero General Public License v3.0](LICENSE).

---

## Topics

`knowledge-base` · `knowledge-extraction` · `knowledge-graph` · `okf` · `llm` · `rag` ·
`indexing` · `entity-extraction` · `code-analysis` · `tree-sitter` · `provenance` ·
`markdown-wiki` · `second-brain` · `data-driven` · `python`

*Suggested GitHub topics (set on the repo):* `knowledge-base`, `knowledge-graph`, `llm`,
`okf`, `rag`, `code-analysis`, `entity-extraction`, `markdown`.
