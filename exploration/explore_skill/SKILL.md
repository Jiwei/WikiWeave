---
name: explore
description: Exploration (EDA) — scan the input corpus, discover the model_data four dimensions (sources, domains, code, ontology), and write them back so the extraction pipeline can run.
---

# Exploration Skill — EDA → model_data

> **What**: The extraction pipeline needs a defined `model_data/` before it can run. This skill
> discovers the **four dimensions** of `model_data/` from the input corpus and writes them back.
> **Invoked by**: `python run.py --explore` (via a Claude Code agent), or manually.
> **State**: `model_data/` — persisted; the pipeline's gate skips exploration when it's already complete.

> **Two layers (merged at load time)**:
> - **`model_schema/`** — the *pre-defined* structure that ships with the tool: the entity schema
>   (`entity_schema.json`, `meta_functions.json`) and the **base entities** (`SourceRecord`,
>   `DomainRecord`, `DocSectionRecord`, `CodeModuleRecord`, `CodeUnitRecord`). **Do not overwrite.**
> - **`model_data/`** — the *discovered* content this skill writes back: source-type `rules`, domain
>   `definitions`/`keywords`/`sub_domains`/`valid_domains`, code `code_domain`, the **non-base
>   entities** in `ontology/entities.json`, the **enums** in `ontology/enums.json`, and the **seed
>   patterns** in `ontology/seed_patterns.json`. The two are merged per-file when the pipeline loads
>   them. (Do **not** put enums into `seed_patterns.json`, nor base entities into `entities.json`.)

---

## The four dimensions (fixed)

`model_data/` is organized along four **fixed** dimensions — invariant across domains, even though
the JSON *inside* each folder changes per corpus:

| Dimension | Question it answers | Files to produce |
|---|---|---|
| **sources** | where does knowledge come from? | `sources/entities.json` (SourceRecord), `sources/source_types.json` |
| **domains** | what is this about? | `domains/definitions.json`, `domains/sub_domains.json`, `domains/valid_domains.json`, `domains/keywords.json` |
| **code** | how do we read the code? | `code/entities.json` (CodeModule/CodeUnit), `code/code_domain.json`, `code/naming_keys.json` |
| **ontology** | what kinds of things exist? | `ontology/entities.json`, `ontology/entity_schema.json`, `ontology/enums.json`, `ontology/seed_patterns.json`, `ontology/meta_functions.json`, `ontology/asset_classes.json`, `ontology/asset_keywords.json`, `ontology/alias_map.json`, `ontology/honorifics.json`, `ontology/nonperson_stoplist.json` |

## Completeness contract (the gate)

The orchestrator checks the **merged** model data before running the pipeline. `--explore` must
produce at least:

- **`sources/source_types.json`** has a non-empty `rules` list;
- **`domains/definitions.json`** has content;
- **non-code corpus**: additionally **`ontology/entities.json`** contains at least one entity
  **beyond** the default base entity `DocSectionRecord`.

## Base entities (the deterministic foundation)

Base entities are **pre-defined in `model_schema/`** (not discovered by this skill) — they are
"given by default", owned by their dimension:

| Entity | Owned by (in `model_schema/`) | Produced by |
|---|---|---|
| `SourceRecord` | `sources/entities.json` | exploration (inventory) |
| `DocSectionRecord` | `ontology/entities.json` | segmentation (doc) |
| `CodeModuleRecord`, `CodeUnitRecord` | `code/entities.json` | segmentation (code scan) |

Everything else (`Task`, `Error`, `Workflow`, `Protocol`, `Project`, `Artifact`, …) is a
**discovered** entity — defined in `ontology/entities.json` and extracted by the LLM.

`code_domain` (in `code/code_domain.json`) links each code domain to a `domain` (+ optional
`sub_domain`) in the domains taxonomy — **code_domain is a subset of domains**.

## Meta-functions (the fixed transformation vocabulary)

Entities are discovered dynamically, but the **transformations** that operate on them are a small,
pre-defined, static vocabulary — the **meta-functions**. Their canonical declaration (purpose,
params, `depends_on`, `kind` variants, usage) lives in
**`model_schema/ontology/meta_functions.json`** — read it before designing entities.

Each discovered entity may declare a `meta` array in `ontology/entities.json`:

```json
"meta": [
  {"op": "seed", "params": {"kind": "spec", "source_type": "project", "spec_file": "任务说明文档.md"}},
  {"op": "group", "depends_on": ["tasks"], "params": {"weights": {"key_match": 3.0, "domain_overlap": 1.5, "title_token": 2.0, "code_stem": 1.0}, "min_score": 3.5, "margin": 1.0}},
  {"op": "synthesize", "params": {"kind": "title"}}
]
```

Entry fields:

| Field | Meaning |
|---|---|
| `op` | one of `extract · seed · group · index · synthesize` |
| `params.kind` | selects a sub-implementation when an op has several (`seed:spec` vs `seed:regex`); the runtime registry key is `op:kind` |
| `depends_on` | list_keys this entity's meta-functions read; the entity runs only after those are produced (topological order) |
| `params.*` | op-specific data (weights, thresholds, sources) — the *how* stays in model_data; the *what-to-do* is generic code |

The five ops:

| Op | LLM | What it does | Typical `kind` / params |
|---|---|---|---|
| `extract` | ✅ | structured extraction from grouped content | driven by `definition` + `list_key` (not a `meta` entry) |
| `seed` | — | deterministic discovery from content | `spec` (parse spec docs) · `regex` (patterns in `seed_patterns.json`) |
| `group` | — | roll up a child entity into a parent (e.g. task→project) | `weights` + `min_score` + `margin` |
| `index` | — | aggregate cross-source mentions into one entity | `sources` (which sources to merge) |
| `synthesize` | ✅ | LLM-refine instances (titles / backfill / detail) | `title`, or backfill+detail via `definition_backfill`/`definition_synthesis` |

When you define a discovered entity, decide its lifecycle:

1. **Pure LLM** → write `definition` + `list_key` only (`extract`).
2. **Deterministic-first, LLM-later** → `seed` (or `index`) + optional `synthesize`.
3. **Child→parent aggregation** → `group` on the parent (with `depends_on` the child).

Scoring weights / thresholds are declared in `params` (never hard-coded); the entity-specific
regex & controlled vocabularies live in `model_data/ontology/seed_patterns.json`.

## Entity definition schema

The complete shape of an entity object (a value in `ontology/entities.json`) is **pre-designed and
fixed**, declared canonically in **`model_schema/ontology/entity_schema.json`**. Every entity object may
carry these fields:

| Field | Kind | What it declares |
|---|---|---|
| `layer` / `ontology_group` | required | phase + grouping tag (`phase1\|phase2`) |
| `base` | bool | base entity (given by default) vs discovered |
| `doc` | string | one-line description |
| `fields` | required array | the dataclass fields — each `{name, type, default?, doc?, ref?, ref_key?}`; `type ∈ str\|int\|float\|bool\|list\|dict\|str?`; required (no `default`) fields must precede defaulted ones |
| `definition` + `list_key` | string | the LLM extraction prompt (with `{items}`/`{lang}`/`__DOMAINS__` placeholders) and the JSON key that carries the extracted list |
| `definition_backfill` / `definition_synthesis` | string | extra prompts for multi-prompt entities (e.g. Artifact backfill / detail) |
| `meta` | array | the meta-functions to run (see [Meta-functions](#meta-functions-the-fixed-transformation-vocabulary)); entry shape in `meta_functions.json` |
| `render` | object | how to render the entity's OKF pages (see [Rendering](#rendering-entities-pages)) |

The runtime generates the record dataclasses from this schema (`util/ontology.py`); field
additions/removals are JSON edits, never code changes.

**Cross-entity references.** A field may declare `ref` (the target entity type its values point to) and
`ref_key` (`id` or `name`) — e.g. a task's `people` field → `ref: "PersonRecord", ref_key: "name"`. This
is how entity→entity links are declared as data: the renderer's `links` sections and reverse indexes
(e.g. the people index) derive their targets from `ref`, instead of hard-coding them.

## Rendering entities (pages)

Each discovered entity may declare a `render` object (shape in `entity_schema.json.render_schema`) that
tells the generic renderer how to produce its OKF pages — `page_dir`, `kind_label`, `slug_field`,
`title_field`, an ordered `detail` list (`bullets` / `text` / `list` / `links` sections), and an `index`
(table, or `group_by` ∈ `null|week|scope|domain`). This is what removes the last per-entity rendering
code: re-theming a corpus = edit the `render` JSON, not the renderer.

## Procedure

1. **Inventory the corpus** — reuse `exploration.inventory` (don't hand-scan): run `detect_content_types()` to learn whether the corpus has code/non-code, and `build_source_inventory()` to list sources with their `source_type` (via `infer_source_type`). These results drive `sources/source_types.json` and decide which dimensions to discover (code vs non-code vs both).
2. **Discover domains** — from section content derive a small multi-label taxonomy: `valid_domains.json` (names + `others`), rich `definitions.json` (per domain: `label`, `concept`, `typical_activities`, `not`, `discriminating_signals`), `keywords.json` (domain → keywords), and `sub_domains.json` (domain → `[zh, en, [keywords]]`).
3. **Discover code objects + code_domain** (if code) — define `CodeModule`/`CodeUnit` fields in `code/entities.json`, a `code_domain.json` list (`name`, `domain` ∈ valid_domains, optional `sub_domain`, `keywords`, `description`) plus `unclassified_label`, and a `naming_keys.json` glossary of team abbreviations.
4. **Discover entities** (if non-code) — define the entity schema in `ontology/entities.json` beyond `DocSectionRecord`, plus actor/asset vocabularies (asset classes + keywords, alias map, honorifics, non-person stoplist).
5. **Write back** `model_data/` in the schemas below; then verify with `python run.py --skip-check --phase 1`.

## JSON schemas

Each entity object: `{"layer": "phase1|phase2", "ontology_group": "…", "base": bool, "fields": [{name,type,default?,doc?}]}`.
`type` ∈ `str|int|float|bool|list|dict|str?`; required fields (no `default`) must precede defaulted ones.

### sources/entities.json
```json
{
  "SourceRecord": {
    "layer": "phase1", "ontology_group": "provenance", "base": true,
    "fields": [
      {"name": "id", "type": "str"}, {"name": "path", "type": "str"},
      {"name": "source_type", "type": "str"}, {"name": "file_type", "type": "str"},
      {"name": "title", "type": "str"}, {"name": "hash", "type": "str"},
      {"name": "domains", "type": "list", "default": []}
    ]
  }
}
```

### sources/source_types.json
```json
{"rules": [{"path_contains": "/聊天记录/", "source_type": "chat"}], "default": "doc"}
```

### domains/definitions.json
```json
{"model-research": {"label": "Model & Alpha Research", "concept": "…", "typical_activities": "…", "not": "…", "discriminating_signals": "…"}}
```

### code/code_domain.json
```json
{"code_domains": [{"name": "5. Label Generation", "domain": "target-engineering", "sub_domain": "Label Definition", "keywords": ["y_", "label"], "description": "…"}], "unclassified_label": "Other / Unclassified", "unclassified_description": "…"}
```

### ontology/entities.json (discovered entities, beyond DocSectionRecord)
```json
{"TaskRecord": {"layer": "phase2", "ontology_group": "entity", "base": false, "fields": [{"name": "id", "type": "str"}, {"name": "title", "type": "str"}, {"name": "progress", "type": "str"}, {"name": "source_sections", "type": "list", "default": []}]}}
```

> Re-theming a corpus = rewrite the JSON **inside** these four folders — never the folder structure or the Python.
