# D'accord

Cross-jurisdiction regulatory clause mapping. Privacy MVP across SEA + EU; legal-domain extensible.

D'accord ("agreed" in French) is a small specialized language model fine-tuned to find the parallel provisions of a given regulatory clause across other jurisdictions, with article-level citations. Given a control requirement in (e.g.) GDPR, it returns the analogous requirement in PDPA-SG / PDPA-TH / Loi Informatique et Libertés / BDSG with the exact section ID and a one-sentence justification.

**MVP scope (privacy)**: 8 data-privacy framework families spanning SEA-4 + EU-spine + UK + DE + FR. Operational-resilience deferred to v1; other legal domains (employment, AML/KYC, consumer protection) extensible via the same pipeline.

**Why this exists**: Cross-jurisdiction compliance teams currently re-map controls manually across 4–10 frameworks when a business expands to new jurisdictions. Frontier LLMs hallucinate citations and miss SEA-specific framework references. A small specialized model with citation-faithful output + native validation (Thai + French via the author) closes that gap.

**Interface**: a side-by-side comparison demo — input a privacy clause, see the fine-tuned d'accord output, the retrieval-baseline output, and (when input is in the eval set) the gold answer in parallel columns, each with a provenance tag (`gold-retrieval` / `fine-tune-generalization` / `no-confident-match`) and clickable citations. One-click CSV export of the comparison row, so the output drops straight into a compliance team's control matrix.

## Scope (MVP)

**SEA-4**:
| Jurisdiction | Frameworks | Native validation |
|---|---|---|
| Singapore | PDPA-SG | English |
| Thailand | PDPA-TH + Royal Gazette amendments | **Thai (author reads natively)** |
| Philippines | DPA 2012 | English |
| Malaysia | PDPA-MY | English |

**EU-spine + UK + DE + FR**:
| Jurisdiction | Frameworks | Native validation |
|---|---|---|
| EU-level | GDPR | English |
| United Kingdom | UK-GDPR, DPA 2018, ICO guidance | English |
| Germany | BDSG | German + English translations |
| France | Loi Informatique et Libertés + CNIL guidance | **French (author reads partially)** |

## Architecture

| Stage | Responsibility |
|---|---|
| `data/ingest/` | PDF → markdown via Marker (locked for both EN and TH; Thai bake-off result: marker recall=1.0 precision=1.0 reading-order=5.0 vs typhoon reading-order=4.0 — see [data/parser_bakeoff/summary.md](data/parser_bakeoff/summary.md)) |
| `data/registry/` | Per-framework valid citation IDs extracted from parsed markdown — deterministic key for ensemble filtering |
| `data/ensemble/` | Multi-model candidate generation — 4 paid-API seats (Claude Haiku 4.5, GPT-5-mini, Gemini 3.1 Flash Lite, Qwen 3-235B via Together) + a 5th CPU RAG seat; chain-of-thought JSON output; citation IDs constrained to registry. Subtree map in [data/README.md](data/README.md) |
| `data/ensemble/tiered/` | HIGH/MEDIUM/LOW/SALVAGE classification (deterministic via registry agreement) + bidirectional reverse-direction cross-check |
| `data/gold/` | Frozen mapping pairs (`gold_v1.jsonl`, 1,002) — committed. **Provisional: auto-promoted, zero hand-validation** |
| `training/` | QLoRA training (Qwen3-8B base, 4-bit NF4, all-linear LoRA) — MLflow-tracked via `report_to=["mlflow"]`; 1–2 sentence summary target (full clause body served via retrieval) |
| `eval/` | Three-tier scoring (citation exact match + LLM-as-judge semantic + ~100-example human spot-check) across six comparators including a retrieval baseline; stratified by in-domain vs out-of-domain ([eval/README.md](eval/README.md)) |
| `publish/` | Model-packaging + cloud inference handler — **deferred to a future phase** (only needed if live generation on out-of-corpus clauses is added; v0 needs no cloud hosting) |
| `src/daccord/serving/` | `HybridRouter` (retrieval-first, QLoRA fallback, per-response provenance tagging) — runs at **precompute time** to generate the static demo's saved outputs; the same router would back any future live endpoint |
| `consumer/` | Side-by-side comparison + CSV export. **A static, no-backend interactive demo** (`consumer/demo/`) over precomputed model outputs — select any corpus clause, compare columns, filter, export CSV, all client-side; self-hostable anywhere (no streamlit.io, no server, no GPU). The source-clause space is closed (every clause is a corpus citation), so everything is precomputable; optional free-text input is a RAG hop (embed → FAISS nearest clause → precomputed mapping), needing only a free/cheap embedder, not a GPU endpoint |

## Methodology

**Dataset construction (~50% of effort)**:
1. **Seed from public crosswalks** — NIST 800-53 ↔ ISO 27001 (published by NIST, free) as authoritative anchors.
2. **Citation registry extraction** — per-framework structured TOC of valid section/article IDs from parsed markdown. Avoids the "LLM says `GDPR Art. 32` but Marker emitted `### 32. Security of processing`" mismatch under naive substring matching.
3. **Multi-model ensemble with constrained citations + chain-of-thought** — each model emits `{source_mechanism, target_mechanism, mapping_justification, citation_id}` with citation_id restricted to the target framework's registry. Four diverse paid-API seats, one per family (Anthropic Claude Haiku 4.5 / OpenAI GPT-5-mini / Google Gemini 3.1 Flash Lite / Alibaba Qwen 3-235B via Together), plus an independent 5th CPU **RAG seat** (MPNet + FAISS over target clauses).
4. **Tier classification** — HIGH (4/4 agree), MEDIUM (≥60%), LOW (<60%), SALVAGE (no valid votes), plus a bidirectional reverse-direction cross-check and the RAG seat as independent corroboration signals.
5. **Gold promotion** — designed for human hand-validation of MED/LOW/SALVAGE with a 10% HIGH spot-check. **As executed for M2, gold was frozen provisionally from auto-promotion only (HIGH + bidirectional-consistent MED + rag-concurs MED), ZERO hand-validation** — reversible because raw data is immutable; hand-validation revisited via the tier-7C labeler if M3/M4 quality bites.

**Training**: QLoRA on Qwen3-8B, 4-bit NF4, all-linear LoRA, MLflow-tracked, local on RTX 5080. The M3 small-run validated the pipeline (200 × 1 epoch); the trainer targets a 1–2 sentence mechanism summary (full clause text is served via the retrieval path), which keeps sequences within 16 GB VRAM at micro-batch 1.

**Eval** — three-tier scoring across four comparator models:
- **Tier 1 — Citation exact match**: deterministic, cheap; top-1 and top-3.
- **Tier 2 — LLM-as-judge semantic match**: scores substance match, mitigating the exact-match penalty for valid paraphrasings. The M4 judge is **Claude Haiku 4.5 via the direct Anthropic API**, deliberately chosen from outside the comparator pool to avoid self-judging bias (the M0 baseline used a Groq judge).
- **Tier 3 — Human spot-check** (~100 examples): quantifies judge accuracy; calibrates Tier 2 scores.

**Comparators**: fine-tuned d'accord vs base Qwen 3-8B vs Llama 4 Scout (Groq) vs Qwen 3-32B (Groq) vs Gemini 3.1 Flash Lite (Google AI Studio) vs **retrieval baseline** (sentence-transformers MPNet + FAISS over train-split source clauses). The retrieval baseline answers the architectural question "could you have just done retrieval?" with data. The Qwen-3-32B comparator additionally asks "would the newer-and-bigger same-family model already beat us without fine-tune?".

**Stratification**: each eval pass runs twice — once on **in-domain** pairs (val/test inputs whose source clauses are cosine-near a train-split clause; retrieval-friendly) and once on **out-of-domain** pairs (jurisdiction-disjoint held-out requirement areas; where the fine-tune's generalization should pay off). Slice tag goes to MLflow run metadata, not the per-row CSV (CSV row contract stays stable per [eval/README.md](eval/README.md)).

Per-jurisdiction + per-language breakdowns are aggregated from CSV rows at read time to quantify the SEA/FR/DE differentiation.

## Current State

Phase 1 (local validation) in progress. Phase 2 (optional, provider-agnostic cloud hosting) triggered separately when M4 lands a publishable delta and a live demo is actually needed. See [development plan](docs/development_plan.md) for the milestone gate definitions + cut criteria; the table below is the live status.

### Phase 1 — Local validation

- **M0 — Eval bar locked (⚠ partial, 2026-05-25)**
  - 20-pair toy gold built; 9/20 author-verified, 11/20 claude-extract-only pending paraphrase semantic pass (deferred but no longer gates downstream — only affects toy `eval/baseline_toy.csv`).
  - Tokenizer audit on **Qwen3-8B** (locked as the QLoRA base): PASS for th (0.575 tok/char), fr (0.520), de (0.303), en (0.213). R4 resolved.
  - Thai parser bake-off: Marker locked (recall=1.0, precision=1.0, reading-order=5.0 vs Typhoon-OCR 4.0 on 5-page PDPA-TH sample with 61 hand-verified Thai citations). R1 resolved.
  - 4 baseline comparators on toy gold judged by Llama 4 Scout: Qwen 3-8B (local NF4), Llama 4 Scout (Groq), Qwen 3-32B (Groq), Gemini 3.1 Flash Lite. Numbers in [eval/baseline_toy.csv](eval/baseline_toy.csv).
  - Eval harness: citation-exact-match + LLM-as-judge, 14-column CSV contract, MLflow nested-run logging, pair-major iteration so per-provider RPM density stays under free-tier ceilings.
  - **Strict bar pending**: 11 claude-extract-only rows pending paraphrase pass; FR Loi I+L native-validation coverage at 2/3 pairs (decide accept-vs-repoint). Both decoupled from M1+ — toy gold re-eval is a 5-min run.

- **M1 — Corpus + registry frozen (✓ 2026-05-26)**
  - ✓ 13-PDF corpus parsed to markdown via Marker (tier 4): 10 parses + 3 cached, **~37 min wall-time, 0 failures**. Manifest at [data/ingest/manifest.jsonl](data/ingest/manifest.jsonl).
  - ✓ R8 PASS: browser-print PDFs (UK-GDPR, UK DPA 2018, FR Loi I+L) hit **1.29× the regulator-baseline citation density** — no R8 fallback needed. Report at [data/ingest/r8_spotcheck.txt](data/ingest/r8_spotcheck.txt).
  - ✓ Per-framework citation-registry extraction (tier 5): 9/9 frameworks extracted from the parsed markdown, **100% toy-gold base-section recall** on every framework. Registries at [data/registry/](data/registry/) (gdpr 92, uk_gdpr 103, dpa_2018 288, bdsg 97, loi_il 126, pdpa_sg 91, pdpa_th 96, pdpa_my 146, dpa_2012_ph 72 citation IDs). R8 follow-up resolved: PDPA-MY Malay regex (`Seksyen`) catches what the EN-only baseline missed.

- **M2 — Gold set frozen (✓ provisional, 2026-06-05)**
  - ✓ Ensemble generated via **Path 2 paid direct API** (Claude Haiku 4.5 / GPT-5-mini / Gemini 3.1 Flash Lite / Qwen 3-235B via Together) + a 5th CPU **RAG seat** — 35,552 raw candidates, 99.955% success, ~$32. Tiered to 8,888 rows (HIGH 840 / MED 1,113 / LOW 6,633 / SALVAGE 302) + bidirectional reverse-direction cross-check.
  - ✓ Gold frozen at **1,002 pairs** (2× the ≥500 floor) — jurisdiction-disjoint splits (test={th,ph}, val={my}) committed at [data/gold/](data/gold/) with dataset SHA in `gold_v1_manifest.json`.
  - ⚠ **Provisional — ZERO hand-validation.** Gold is auto-promotion only (HIGH + bidirectional-consistent MED + rag-concurs MED); 134 HIGH+inconsistent rows ship unreviewed. Reversible (raw data immutable); revisit via the tier-7C labeler if M3/M4 quality bites. See [development_plan.md](docs/development_plan.md) M2.
- **M3 — Small-run validated (✓ 2026-06-06)**
  - ✓ Tier 10A `training/` sub-project (QLoRA on Qwen3-8B, 4-bit NF4, all-linear LoRA) + `scripts/build_training_data.py` + `training` compose service. MLflow via `report_to=["mlflow"]` (not global autolog).
  - ✓ Small-run 200 × 1 epoch, ~10 min — train loss 1.306→0.862, eval_loss 0.967→0.963; adapter (87 MB) reloads via `LocalAdapterClient` + emits coherent mappings; MLflow logs git_commit·seed·dataset_hash·adapter_sha256 (sha verified == on-disk file).
  - **Training target = ensemble 1–2 sentence summary** (the full clause body is served via the retrieval path instead) — aligns train with the eval prompt and keeps sequences within 16 GB VRAM.
  - **R5 finding**: on 16 GB Windows/WSL the 5080 *silently spills* to system RAM at seq-4096 × micro-batch 2 (NVIDIA sysmem fallback — no clean CUDA OOM, just a crawl); watch step-time/shared-memory, not an exception. Resolved at micro-batch 1 + the summary target. Qwen3 `<think>`-block parser fix also landed (else M4 would score every fine-tune row as a parse error).
- **M4 — Eval delta proven (Phase 1 done)** ⏳ — next gate (M3 closed). Three-tier eval × **6 comparators** × 2 slices (in-domain + out-of-domain) per-jurisdiction breakdown; judge = Claude Haiku 4.5 via direct Anthropic API. Tier-12 implementation steps in [docs/tier12_plan.md](docs/tier12_plan.md).

### Phase 2 — cloud GPU hosting (deferred to future scope)

- **M5 — Demo captured (no live endpoint needed)** ⏳ — the source-clause space is **closed** (every clause is a corpus citation), so the precomputed static demo covers the entire MVP, including free-text input via a RAG hop (embed → FAISS nearest clause → precomputed mapping, free/cheap embedder, no GPU). A live GPU endpoint is therefore **unnecessary for v0** — it only earns its cost for clauses outside the 9-framework corpus (future frameworks). If ever wanted, it deploys behind the cloud-agnostic `HybridRouter` to a scale-to-zero provider (**HF Endpoints / Modal**; SageMaker faces the same AWS approval wall as Bedrock and is not pre-warmed). The durable artifact is the recording, not a running endpoint.

### Cross-milestone infrastructure (in place since 2026-05-25)

- **Dev environment**: Docker Compose, 7 services (`root`, `eval`, `audit`, `bakeoff`, `baseline`, `ingest`, `consumer`); shared uv wheel cache + HF model cache + surya datalab cache via named volumes; per-env Python split (3.13 for marker-pdf-using `bakeoff` + `ingest`; 3.14 elsewhere).
- **Cost discipline**: per-provider RPD caps (Groq 14400 / Gemini 1500 / Cerebras 1000 / DeepSeek 1000) wired into the cost layer; shared 10-RPM throttle, Gemini transient-error retry, Groq APIError safety net in all clients; eval runner uses pair-major iteration so per-provider density stays well under any single provider's free-tier cap.
- **Hybrid serving**: `HybridRouter` (retrieval-first + QLoRA fallback, per-response provenance tagging) shared between the demo-data precompute step and the optional cloud inference handler.

## Development environment

**All development runs in Linux containers via Docker Compose.** Two thin Dockerfiles (CPU + CUDA) back six compose services (`root`, `eval`, `audit`, `bakeoff`, `baseline`, `consumer`). Host requirements: Docker Desktop (WSL2 backend) + an NVIDIA Windows driver for the GPU services — no CUDA Toolkit install needed on the host (bundled in the CUDA image).

Quick start:

```bash
docker compose build root eval                       # CPU image (first time only)
docker compose run --rm root uv sync                 # shared daccord lib
docker compose run --rm eval uv sync                 # tier 2B eval harness
docker compose run --rm eval uv run pytest           # 78/78 should pass
```

Per-env Python split: root/eval/audit/baseline/consumer on 3.14; bakeoff on 3.13 (held back by marker-pdf's `pillow<11` ceiling). Each service's `working_dir` is set to its env folder in `docker-compose.yml`, so `pytest`/`ruff`/`pyright` pick up the right `pyproject.toml` without a `cd`.

## Tech Stack

- **Base model**: Qwen3-8B (multilingual: Thai, French, German, English — tokenizer audit PASS on all four; see [eval/tokenizer_audit.md](eval/tokenizer_audit.md))
- **Training**: QLoRA via PEFT
- **MLOps tracking**: MLflow
- **PDF processing**: Marker (locked for both EN and TH after a 5-page Thai bake-off vs Typhoon-OCR — both hit perfect citation extraction; Marker preferred for ~2× faster wall time and noise-free body output free of Royal Gazette page-header chrome)
- **Ensemble labelers**: four paid-API seats, one per family — Claude Haiku 4.5 (Anthropic), GPT-5-mini (OpenAI), Gemini 3.1 Flash Lite (Google), Qwen 3-235B-A22B via Together (Alibaba) — plus a 5th CPU **RAG seat** (MPNet + FAISS). This is the Path-2 lineup actually run for M2 (the Bedrock-only Path-1 lineup was preserved in code but AWS denied the Bedrock quota)
- **Eval judge (M4)**: **Claude Haiku 4.5 via the direct Anthropic API** — kept outside the comparator pool to avoid self-judging bias (Bedrock was the original judge route but its quota was unavailable; the free-tier Groq route is a comparator, so it isn't used as judge). M0 baseline used a Groq judge with a documented self-judging-bias note
- **Retrieval baseline**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` + `faiss-cpu` (FAISS index over train-split source clauses; also reused at serving time by the hybrid router)
- **Hybrid serving**: `HybridRouter` (retrieval-first, QLoRA fallback, per-response provenance tagging) — shared between the demo-data precompute step and the optional cloud inference handler
- **Demo consumer**: static, no-backend interactive side-by-side comparison UI (client-side, over precomputed outputs) with CSV export — self-hosted, no server cost
- **Deployment**: precomputed static demo — no live endpoint, no GPU, self-hosted anywhere. Cloud GPU hosting (HF Endpoints / Modal / SageMaker behind `HybridRouter`) is **deferred to a future phase**, justified only by out-of-corpus generation

## Roadmap

- [Development plan (Phase 1 → Phase 2)](docs/development_plan.md) — tiers, milestones, gates, risk register

### Milestone history

| Milestone | Date | What landed |
|---|---|---|
| **1A–1D** | 2026-05-22 | Repo skeleton + MLflow plumbing + per-provider RPD caps (Groq/Gemini/Cerebras/DeepSeek) + 13-PDF corpus on disk |
| **2B** | 2026-05-23 | Eval harness end-to-end (citation match + LLM-as-judge, 14-col CSV contract, MLflow nested-run logging) |
| **2C** | 2026-05-24 | Tokenizer audit PASS on th/fr/de/en for Qwen3-8B; R4 resolved |
| **2D** | 2026-05-24 | Thai parser bake-off — Marker locked (5-page sample, 61 hand-verified Thai citations); R1 resolved |
| **Docker migration** | 2026-05-25 | Dev env moved to Docker Compose (6→7 Linux services); consumer pivoted chatbot → side-by-side comparison + CSV; shared 10-RPM throttle + retry layer in API clients; pair-major eval iteration |
| **3A (M0)** | 2026-05-25 | 4 baseline comparators on toy gold judged by Llama 4 Scout; `envs/baseline/` + `LocalHFClient` + `GroqJudge` shipped; Qwen3-8B locked as QLoRA base |
| **4 (M1)** | 2026-05-26 | 13-PDF corpus parsed to markdown via Marker; `envs/ingest/` + 7th compose service; surya weights cached + `hf_transfer` (~65× download speedup); R8 PASS at 1.29× regulator-baseline citation density |
| **5 (M1)** | 2026-05-26 | Per-framework citation-registry extraction — 9/9 frameworks, 100% toy-gold base-section recall, idempotent reruns; closes M1 |
| **6–9 (M2)** | 2026-06-05 | Path-2 paid ensemble + RAG seat (35,552 candidates) → tiering + bidirectional → **provisional gold freeze 1,002 pairs (zero hand-val)** + jurisdiction-disjoint splits + dataset SHA |
| **10–11 (M3)** | 2026-06-06 | `training/` QLoRA sub-project + summary-target build; small-run (200×1ep) validated — loss curve sane, adapter saves/reloads, MLflow SHA-linked; R5 sysmem-spill + Qwen3 `<think>` parser fixes |
| **12–13 (M4)** | TBD | Full QLoRA train (small sweep) + three-tier eval × 6 comparators × 2 slices (judge: Claude Haiku 4.5 direct) — Phase 1 done. See [tier 12 plan](docs/tier12_plan.md) |
| **14–18 (M5)** | TBD (Phase 2, deferred) | Precomputed static demo capture (no endpoint, no GPU). Live cloud GPU endpoint deferred to future scope (only needed for out-of-corpus clauses) |

### Release targets

| Version | Date | Focus |
|---|---|---|
| **v0 MVP** | 2026 Q2 | Privacy: SEA-4 + EU(spine+UK+DE+FR); QLoRA fine-tune on Qwen3-8B; three-tier eval (citation exact match + LLM-as-judge + human spot-check) with retrieval baseline + in/out-of-domain stratification; hybrid serving (retrieval + fine-tune fallback with provenance tagging); static, no-backend interactive side-by-side comparison + CSV export |
| **v1** | TBD | Operational-resilience extension: MAS TRM, BOT IT, OJK POJK, BNM RMiT, BSP, DORA, EBA, PRA SS1/21, BaFin BAIT/MaRisk |
| **v2+** | TBD | Additional legal domains (employment, AML/KYC, consumer protection) and additional jurisdictions |

## Known Limitations

- **Browser-print PDFs for UK + FR**: UK-GDPR, UK DPA 2018, and FR Loi Informatique et Libertés have no scraper-friendly consolidated PDFs (legislation.gov.uk and Légifrance expose only HTML). The corpus falls back to browser print-to-PDF for these three sources — 5–60× larger files with embedded page chrome. Risk R8 in the development plan; tier-4 spot-check **PASS** at 1.29× regulator-baseline citation density (report at [data/ingest/r8_spotcheck.txt](data/ingest/r8_spotcheck.txt)).
- **Gemini free-tier daily cap**: `gemini-3.1-flash-lite` ships 15 RPM / 500 RPD on the free tier. The earlier `gemini-2.5-flash` daily cap was as low as 20 RPD on some accounts, which exhausted mid-baseline; the project standardised on 3.1 Flash Lite and the Llama 4 Scout judge sidesteps any per-minute spikes via the 10-RPM global throttle + transient-error retry layer + pair-major iteration (per-provider density stays at ~1 call per pair-cycle, well under any single provider's cap).
- **16 GB VRAM is tight for 8B QLoRA at long sequences**: the M3 small-run validated training end-to-end, but seq-4096 × micro-batch 2 *silently spills* to system RAM on Windows/WSL (the NVIDIA driver's sysmem fallback — no clean CUDA OOM, just a slowdown). The working config is **micro-batch 1** with a **1–2 sentence summary target** (the full registry clause bodies hit ~22 K tokens and overflowed VRAM; full text is served via retrieval instead). Unsloth remains the documented fallback if the 12A full train needs more headroom.

---

### Keywords

- **Language**: `Python`
- **Domain**: `Cross-Jurisdiction Regulatory Mapping` · `Privacy Compliance` · `RegTech` · `Legal NLP` · `Citation-Faithful Output`
- **Frameworks (MVP)**: `GDPR` · `UK-GDPR` · `DPA 2018` · `BDSG` · `Loi Informatique et Libertés` · `CNIL guidance` · `PDPA-SG` · `PDPA-TH` · `DPA 2012 (PH)` · `PDPA-MY`
- **ML & MLOps**: `QLoRA Fine-Tuning` · `PEFT` · `MLflow` · `LoRA Adapter` · `Multi-Model Ensemble Labeling` · `Weak Supervision` · `Chain-of-Thought Structured Output` · `Citation Registry Constraint` · `Three-Tier Evaluation` · `LLM-as-Judge` · `Human Spot-Check Calibration`
- **Base / Ensemble Models**: `Qwen3-8B (base)` · `Qwen3-32B` · `Llama 4 Scout (17B × 16E MoE)` · `Gemini 3.1 Flash Lite` · `DeepSeek V3` (all open-weight; free-tier-served)
- **PDF / Layout**: `Marker (ViT layout, locked for EN + TH)` · `LlamaParse (fallback)` · `Citation Registry Extraction`
- **Deployment**: `Static no-backend interactive demo (precomputed, self-hosted)` · `Cloud GPU hosting deferred to a future phase (HF Endpoints / Modal / SageMaker behind HybridRouter)`
