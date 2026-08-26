# Auto-Archivist

> A config-driven, provenance-anchored pipeline that turns an arbitrary corpus of source materials — **code, docs, chat logs, reports, specs, anything** — into a **linked, self-contained knowledge base**: an OKF-compliant Markdown wiki + structured JSON + navigable HTML with meta ontology schema and mining functions, bridging code intelligence and document intelligence.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

**The self-building knowledge graph for your code and docs.** Rather than hand-curating a
knowledge base or throwing everything into a vector store, Auto-Archivist **discovers the
structure itself** — from a config-driven ontology template and a fixed set of *meta-functions* —
and mines an arbitrary corpus into a queryable, cross-linked, auditable graph.

---

## What it does

Individual and teams often accumulate knowledge in messy, unstructured form: chat threads, one-off READMEs, weekly
reports, scripts whose "why" lives only in someone's head. **Vector stores store embeddings, not
meaning** — context you can't explain and links you can't trace. Auto-Archivist turns that corpus
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

## Why Auto-Archivist

|                           | Plain RAG / vector store | Hand-curated wiki         | **Auto-Archivist**                 |
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
- **R&D teams** (quant, ML, engineering) whose "why" lives in codes, chats and weekly reports, while the knowledge landscape keep being updated.
- **AI-agent builders** that need a clean, provenance-anchored context layer instead of raw chunks.
- **Anyone inheriting a messy repo** who needs to reconstruct *what exists and why* from the corpus itself.

---

## What Auto-Archivist gives you

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

# 2. point it at a corpus and run
python -m knowledge_extractor.run /path/to/your/corpus --phase all

#   or via env var / config (see Configuration)
KE_INPUT_PATH=/path/to/corpus python -m knowledge_extractor.run --phase 1
```

Phases: `--phase all` (default) | `1` | `2` | `check`.

The orchestrator first checks that `model_data/` is defined; if not, it points you to
[exploration](#exploration--the-agent). Then it segments, extracts, links, and renders:

```
artifact/
├── knowledge_bits/          # structured JSON (the machine-readable layer)
│   ├── sources.json         # deduped source inventory
│   ├── doc_sections.json    # segmented document sections (+ domain classification)
│   ├── code_units.json      # code units (modules/classes/functions)
│   ├── people.json          # people index (alias-normalized, confidence-bucketed)
│   ├── phase2_tasks.json    # extracted tasks (+ errors/workflows/protocols/…)
│   └── …
├── index/                   # self-contained OKF Markdown wiki (human/agent-readable)
└── index_html/              # navigable HTML wiki
```

---

## Pipeline

A four-stage pipeline, each stage an isolated package with a narrow contract — any stage can be
replaced or extended independently.

```
                         ┌────────────────────────────────────────────────┐
   raw sources ─────────▶│  1. EXPLORATION          (exploration/)        │
   (any directory)       │     inventory · content-hash dedup ·           │
                         │     data-driven source-type routing            │
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

| Stage                | Package                                   | What it does                                                                                                                                                                                                                                       |
| -------------------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Exploration**      | `exploration/inventory.py`                | Recursively walks the input directory, dedups by content hash, and routes each file to a source type using **data-driven rules** (`model_data/sources/source_types.json`).                                                                         |
| **Segmentation**     | `segmentation/doc/`, `segmentation/code/` | Splits documents into sections (headings, chat speaker turns, font-aware PDF, style-aware DOCX) and code into units (Python `ast`, tree-sitter for other languages); classifies each into one or more **domains** (LLM-primary, keyword fallback). |
| **Extraction**       | `extraction/`                             | Groups related sections, then runs **structured LLM extraction** of tasks, errors, workflows, protocols, projects, and artifacts; builds a normalized **people index** with aliases and confidence.                                                |
| **Linking + Render** | `render/`, `segmentation/code/scan.py`    | Bakes **links** (code↔doc, entity↔source backlinks, search tags), then renders the self-contained **OKF Markdown wiki**, HTML, code map, call/dependency graphs, and a glossary (`TERMS.md`).                                                      |

---

## Design philosophy

### 1. Data-driven over hard-coded

The single most important decision. Everything domain-specific lives in JSON along **four fixed
dimensions** (`code/`, `domains/`, `ontology/`, `sources/`), split between **`model_schema/`** (the
pre-defined structure — entity schema, base entities, generic vocabularies) and **`model_data/`**
(the discovered content, written back by exploration); the two are merged at load time.

**Re-theming for a new domain = edit JSON, not Python.** This is what makes the pipeline "generic"
rather than "a one-off tool that happens to be configurable."

### 2. Deterministic-first, LLM-enhanced, graceful degradation

Every LLM step has a **deterministic fallback** (keyword matching, heuristics). If the LLM is
unavailable or a call fails, the pipeline **degrades instead of dying** — it emits lower-fidelity
but still-valid output. LLM calls are batched, checkpointed, and retried with backoff.

### 3. Provenance-anchored

Every claim is traceable. Entities carry `source_sections` (the section IDs that produced them),
sections carry `source_id` + `locator` (file/line), and the inventory records content hashes.
Nothing in the output is asserted without a path back to the raw material.

### 4. Config-driven ontology

The entity schema is the single source of truth for the data model. `util/ontology.py` generates the
record dataclasses at import time, so the schema is data — field additions/removals are a JSON edit,
not a code change. A field may declare a **cross-entity reference** (`ref` + `ref_key`), e.g. a task's
`people` → `PersonRecord` by name.

### 5. Self-contained, human- *and* agent-friendly output

The rendered wiki follows the OKF convention (markdown files with `type` frontmatter, standard links
= knowledge graph). It is browsable in any editor/Obsidian, grep-able by humans, and loadable by
agents — no database or runtime required to *read* the result.

### 6. Immutable sources, idempotent pipeline

Raw sources are never modified. The pipeline is resumable (phase-level checkpoints keyed by content
hash + schema version), so re-running after adding materials only does incremental work.

### 7. Meta entity modeling and meta-functions

Entities are discovered dynamically, but the **transformations** on them are a small fixed set —
**extract · seed · group · index · synthesize** — declared per-entity as a `meta` array in
`ontology/entities.json` and dispatched by `extraction/meta.py`:

- `extract` — LLM pulls instance records out of grouped content (driven by `definition` + `list_key`)
- `seed` — deterministic discovery from content (`spec` parsing / `regex` patterns)
- `group` — rolls a child entity up into a parent (e.g. task → project, with weighted scoring)
- `index` — aggregates cross-source mentions into one entity (e.g. the people index)
- `synthesize` — LLM-refines what's been found (titles, backfill, detail synthesis)

The *how* (scoring weights, thresholds, regex patterns) lives in `model_data/`; the *what-to-do* is
generic code. The canonical registries are [`model_schema/ontology/meta_functions.json`](model_schema/ontology/meta_functions.json)
and [`model_schema/ontology/entity_schema.json`](model_schema/ontology/entity_schema.json).

### 8. Base entities + bidirectional linking

The graph rests on five **base entities** — **Source, Domain, DocSection, CodeModule, CodeUnit** —
produced deterministically by exploration/segmentation. Everything else (**Task, Error, Workflow,
Protocol, Project, Artifact, Person, Resource**, …) is a **discovered** entity, mined from the base
entities and declared in `ontology/entities.json`.

The two layers link **bidirectionally**:

- **discovered → base** — each discovered entity carries `source_sections` (the DocSections it came
  from), `domains` (the Domains it was classified into), and `ref` fields (CodeUnits / other entities);
  its page renders these as *Source sections*, *Domains*, and cross-links.
- **base → discovered** — DocSection and Domain pages link back to the entities extracted from them via
  a generic cross-linking pass; Source and Code associate with discovered entities indirectly through
  their sections.

---

## Configuration

Two config files under `config/`:

- **`config/settings.json`** — `input_path` (corpus root), `output_dir`, output subdir names, and
  pipeline flags (`enable_exploration`, `checkpoint`, …).
- **`config/llm.json`** — the LLM endpoint (`base_url`, `model`, OpenAI/Anthropic `protocol`),
  per-phase batch sizing, and retry policy. **Credentials come from env vars** (`LLM_API_KEY`), never
  from the file.

Input path precedence: **CLI arg** → **`KE_INPUT_PATH`** env var → **`settings.input_path`**.

**Language** — the artifact's primary language, applied to both the static UI (headings, labels) *and*
the LLM-generated content (summaries, extracted entities — via an "output language" instruction
injected into every extraction prompt). Default **`en`**; switch to **`zh`** with `settings.language`,
the `KE_LANG` env var, or `--lang zh`. Only English and Chinese are supported for now.

All knowledge configuration lives in **`model_schema/`** (pre-defined) + **`model_data/`** (discovered),
organized along the four dimensions — that is where you go to re-theme the extractor, add domains, or
change the entity schema.

---

## Exploration & the agent

The pipeline needs a defined `model_data/` before it can run. `model_schema/` ships with the tool
(pre-defined schema + base entities); `model_data/` is what **exploration** discovers from the corpus
and writes back. When the required discovered files are missing, the orchestrator refuses and points
you to exploration.

Exploration is **automated via a Claude Code agent** (currently the only supported harness):

- `python -m knowledge_extractor.run --explore` — invokes `claude -p` (headless, `--dangerously-skip-permissions`
  for non-interactive write-back) with the [`explore_skill`](exploration/explore_skill/SKILL.md).
- `python -m knowledge_extractor.run` — skips exploration automatically when `model_data/` is already complete.

The skill discovers sources, domains, code objects + `code_domain`, and entities — reusing
`exploration/inventory.py` for the corpus scan — then writes the results back to `model_data/`.
Afterward the gate re-checks and the pipeline proceeds.

> **Supported agents**: Claude Code only for now. To add another harness, extend `exploration/agent.py`
> — the invocation is isolated there.

---

## Project structure

```
knowledge_extractor/
├── run.py                 # top-level orchestrator (exploration → … → check)
├── exploration/           # 1. source discovery, dedup, routing
│   └── explore_skill/     #    the agentic EDA → model_data skill (SKILL.md)
├── segmentation/          # 2. doc/code splitting + domain classification
│   ├── doc/               #    (chat/markdown/PDF/DOCX sectioning)
│   └── code/              #    (AST + tree-sitter scan, code map, call graph/CFG)
├── extraction/            # 3. entity extraction + deterministic enrichment
│   ├── entities/          #    generic LLM extractor (extract meta-function)
│   └── meta_ops/          #    seed / group / index / synthesize implementations
├── render/                # 4. OKF markdown, HTML, code map, glossary, clusters
├── validation/            # quality-check gate
├── util/                  # ontology, model_data loader, paths, LLM client
├── model_schema/          # pre-defined: schema + base entities + generic vocabularies
│   ├── code/              #    CodeModule/CodeUnit (base)
│   ├── domains/           #    DomainRecord (base) + empty domain templates
│   ├── ontology/          #    DocSectionRecord (base) + entity_schema/meta_functions
│   └── sources/           #    SourceRecord (base) + default routing
├── model_data/            # ★ discovered data (exploration output — merged with model_schema)
│   ├── code/  domains/  ontology/  sources/   # same layout; filled by exploration
└── config/                # settings.json + llm.json
```

---

## Testing

```bash
python -m pytest            # from the repo root (knowledge_builder/)
# or, without pytest installed:
python tests/test_ontology.py
python tests/test_inventory.py
```

Tests cover: entity-schema generation, keyword/domain data integrity, source-type routing, and
recursive inventory + deferred-file collection.

---

## License

[GNU Affero General Public License v3.0](LICENSE).

---

## Topics

`knowledge-base` · `knowledge-extraction` · `knowledge-graph` · `okf` · `llm` · `rag` ·
`indexing` · `entity-extraction` · `code-analysis` · `tree-sitter` · `provenance` ·
`markdown-wiki` · `second-brain` · `data-driven` · `python`

*Suggested GitHub topics (set on the repo):* `knowledge-base`, `knowledge-graph`, `llm`,
`okf`, `rag`, `code-analysis`, `entity-extraction`, `markdown`.
