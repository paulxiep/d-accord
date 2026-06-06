# `data/` — pipeline stages & data map

The pipeline writes through these subdirectories in sequence. The committed-vs-ignored split
([.gitignore](../.gitignore)) follows one rule: **commit the smallest load-bearing ancestor on each
derivation path — the smallest artifact that still captures information you can't cheaply or freely
regenerate — then gitignore everything above it (huge/expensive, already captured by the cut) and
everything below it (cheaply rebuildable from the cut).**

Committed cut-points: `ingest/` (parsing distillate of the source PDFs — re-parsing needs Marker+GPU),
`registry/` (citation universe; needed to rebuild `clauses/`), and `gold/` + `training/` (the small
distillates of the ~$32 ensemble run). Gitignored: cheaply rebuilt *below* the cut (`clauses/`,
`indices/`), or huge/expensive *above* it but already captured by gold+training (`raw/`, `ensemble/`,
`splits/`). `training/` in particular is committed because its summaries live only in 84 MB `tiered/`
(gitignored) — it is itself the smallest artifact carrying them, and version-controls the MLflow
`dataset_hash`.

## Top-level stages

| Dir | Committed? | Holds | Schema | Produced by |
|---|---|---|---|---|
| `raw/` | no | Source PDFs | — | tier 1D (download) |
| `ingest/` | **yes** | Parsed markdown + `manifest.jsonl` | — | tier 4 (Marker / Thai parser) |
| `registry/{fw}.json` | **yes** | Per-framework **valid citation IDs** | `FrameworkRegistry` | tier 5 (`scripts/extract_registry.py`) |
| `clauses/{fw}.json` | no | Per-framework **citation_id → body text** | `FrameworkClauses` | tier 7A prep (`scripts/extract_clauses.py`) |
| `indices/target_clauses/{fw}.{faiss,jsonl}` | no | FAISS vectors + metadata for the RAG seat | — | tier 6B++ (`envs/eval/scripts/build_target_indices.py`) |
| `ensemble/` | no | Candidate generation → tiering → cross-check → verdicts | see below | tiers 7A / 6B / 6B+ / 7C |
| `splits/` | no | Gold-eligible rows partitioned train/val/test (still `TieredPair`) | `TieredPair` | tier 7B (`scripts/build_splits.py`) |
| `gold/` | **yes** | Frozen gold pairs + dataset SHA | `GoldPair` | tier 9 (`scripts/freeze_gold.py`) |
| `training/{train,val,test}.jsonl` | **yes** | SFT chat examples — 1-2 sentence **summary** target | `TrainingExample` (`messages[]`) | tier 10A (`scripts/build_training_data.py`) |

Schemas live in `src/daccord/`: registry/clauses → [`registry/schema.py`](../src/daccord/registry/schema.py);
ensemble shapes → [`ensemble/schema.py`](../src/daccord/ensemble/schema.py),
[`ensemble/bidirectional.py`](../src/daccord/ensemble/bidirectional.py),
[`ensemble/validated.py`](../src/daccord/ensemble/validated.py); gold → [`gold/schema.py`](../src/daccord/gold/schema.py).

## Regenerating the gitignored derivatives

A fresh clone has the committed cut-points (`ingest/`, `registry/`, `gold/`, `training/`) but **not**
`clauses/` or `indices/`. They're byte-deterministic from committed inputs — rebuild them locally
(both CPU-only, no GPU, fast):

```bash
# clauses/{fw}.json  ← data/registry/ + data/ingest/   (byte-identical re-runs)
docker compose run --rm root uv run python scripts/extract_clauses.py

# indices/target_clauses/{fw}.{faiss,jsonl}  ← data/clauses/
docker compose run --rm eval uv run python scripts/build_target_indices.py
```

The expensive upstream — `ensemble/{raw,tiered}/` (144 MB + 84 MB) and `splits/` — is gitignored and
**not** freely reproducible: it requires the paid ensemble run (`run_ensemble.py run-paid`, ~$32 +
~6.5 h). You almost never need it: committed `gold/` + `training/` are the durable distillates. Only
rebuild the ensemble if you're re-deriving the gold set itself. If you do re-run it, regenerate the
trainer input afterward (and re-commit `training/`):

```bash
# training/{split}.jsonl  ← data/gold/ + data/ensemble/tiered/  (needs tiered present locally)
docker compose run --rm root uv run python scripts/build_training_data.py
```

## The `ensemble/` subtree (the part that's easy to get lost in)

| Dir | One row = | Schema | Written by |
|---|---|---|---|
| `ensemble/raw/{src}__{tgt}__{model}.jsonl` | one **LLM seat's** candidate for one source clause | `EnsembleCandidate` | `run_ensemble.py run-paid` (4 paid seats) |
| `ensemble/raw_local/{src}__{tgt}__{model}.jsonl` | the **RAG seat's** candidate for one source clause | `EnsembleCandidate` | `run_ensemble.py run-rag` (CPU FAISS) |
| `ensemble/tiered/{src}__{tgt}.jsonl` | one source clause + its **HIGH/MED/LOW/SALVAGE** label after agreement scoring | `TieredPair` | `tier_ensemble.py` |
| `ensemble/bidirectional/{src}__{tgt}.jsonl` | does the **reverse** pair confirm this mapping? | `BidirectionalResult` | `cross_check_ensemble.py` |
| `ensemble/validated/{src}__{tgt}.jsonl` | a **human reviewer's** verdict (write-once) | `ValidatedPair` | tier-7C labeler (`consumer/labeler/app.py`) |

**Currently `validated/` is empty** — the provisional M2 freeze used zero hand-validation (see
[development_plan.md](../docs/development_plan.md) M2).

## How rows relate (the two keys)

- **Pair key** `{source_framework}__{target_framework}` — e.g. `gdpr__pdpa_sg`. 9 frameworks → **72
  directional pairs** (9×8). The reverse of `gdpr__pdpa_sg` is `pdpa_sg__gdpr`; bidirectional checks join the two.
- **Row key** `source_id` — e.g. `gdpr-1`. **Not globally unique**: `gdpr-1` recurs in every pair where GDPR
  is the source. So joins are always keyed by **(pair, source_id)**, and the frozen `GoldPair.id` prefixes the
  pair (`gdpr__pdpa_sg::gdpr-1`) to de-collide.

## Flow (and where each field comes from)

```
registry/{fw}.json + clauses/{fw}.json   (citation universe + source clause text)
        │
        ├─ run_ensemble run-paid  → ensemble/raw/        (4 LLM votes per source clause)
        └─ run_ensemble run-rag   → ensemble/raw_local/  (1 RAG vote per source clause)
                                          │
              tier_ensemble.py  ──────────┘   group by source_id, score LLM agreement
                                          ▼
                                 ensemble/tiered/         TieredPair: tier, consensus_citation_id,
                                          │                agreement_score, votes[], rag_concurs
                          ┌───────────────┼───────────────┐
        cross_check_ensemble.py     build_splits.py   (labeler → validated/, currently skipped)
                          ▼                 │
                 ensemble/bidirectional/    │  gold-eligibility + jurisdiction partition
                 (consistent / inconsistent)│  (test={th,ph}, val={my}, train=rest)
                          └──── overlay ─────┤
                                          ▼
                                  splits/{train,val,test}.jsonl   (TieredPair, gold-eligible only)
                                          │
                            freeze_gold.py │  project TieredPair → GoldPair:
                                          │   • target_citation_id = consensus_citation_id
                                          │   • target_mechanism  = clauses[tgt][consensus]  (registry text)
                                          │                          ↳ fallback: consensus vote's summary
                                          │   • {src,tgt}_language = gold/languages.py map
                                          ▼
                                  gold/gold_v1.jsonl (+ train/val/test) + gold_v1_manifest.json (dataset SHA)
```

## Tiering & gold-eligibility cheat-sheet

`tier` is derived from LLM-seat agreement (RAG is side-info, not counted):

| tier | agreement | enters gold when… |
|---|---|---|
| HIGH | 4/4 agree | **always** (regardless of bidirectional status — incl. 134 HIGH+inconsistent) |
| MED | ≥60% agree | bidirectional `consistent` **or** `rag_concurs` **or** a `validated/` verdict |
| LOW | <60% | only a `validated/` verdict (none currently) → excluded |
| SALVAGE | 0 valid votes | only a `validated/` verdict → excluded |

Current scale: 8,888 tiered rows (HIGH 840 / MED 1,113 / LOW 6,633 / SALVAGE 302) → **1,002 frozen gold
pairs** (auto-promotion only). Of the gold target text, 859 rows use registry body text and 143 fall back to
the consensus seat's summary (registry body-recall gap); 5 rows were excluded for citing a non-registry id.
See [`gold/gold_v1_manifest.json`](gold/) for the exact freeze stats + SHA.

## Citation normalization

Citation IDs are normalized via `daccord.eval.scoring.normalize_citation_id` (M0-locked). `registry/` and
`clauses/` keys are canonical (normalized + uppercase letter suffix); `tiered` `consensus_citation_id` is
already normalized; joins re-normalize both sides so they meet in the same space.
