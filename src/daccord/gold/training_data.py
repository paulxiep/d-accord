"""Tier-10A training-data build — project frozen `GoldPair` rows into chat-format
SFT examples for the QLoRA trainer.

The trainer ([training/train.py]) consumes one clean file per split,
`data/training/{split}.jsonl`, each a `{"messages": [...]}` row. This module does
the joins the trainer should not — pulling the **target_mechanism summary** and
the **mapping_justification** from the consensus ensemble vote in
`data/ensemble/tiered/{framework_pair}.jsonl`:

  - **target_mechanism**: the consensus vote's 1-2 sentence summary, NOT the
    frozen `GoldPair`'s full registry clause body. The eval prompt asks for "a
    one-to-two sentence summary"; the summary keeps train/eval aligned and
    collapses sequence length (full clause bodies run to 20K+ tokens, which
    overflows 16 GB VRAM). The authoritative full clause text is preserved in
    `data/clauses/` + the eval gold and is the source for the retrieval/verbatim
    serving path (`HybridRouter`, cloud) — it is *not* discarded, just not used
    as the fine-tune target. Falls back to the `GoldPair` body if a vote summary
    is unavailable.
  - **mapping_justification**: the consensus vote's justification.

Reads only `data/gold/` and `data/ensemble/tiered/` (a derivative of raw);
writes only `data/training/`. **`data/ensemble/raw/` is never opened.**
See [docs/m3_gate.md].

The prompt (system+user) comes from `build_eval_prompt` — byte-identical to what
the eval harness + `LocalAdapterClient` feed the model. Logic lives here
(root-env importable, unit-tested); `scripts/build_training_data.py` is the CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

from daccord.ensemble.schema import TieredPair
from daccord.eval.prompts import build_eval_prompt
from daccord.eval.scoring import normalize_citation_id
from daccord.gold.schema import GoldPair, GoldSet
from daccord.validation import ValidatedModel, validated


class TrainingExample(ValidatedModel):
    """One chat-format SFT row: `messages = [system, user, assistant]`.

    The assistant turn is the supervised target; `SFTTrainer`'s
    `assistant_only_loss` masks the system+user tokens.
    """

    messages: list[dict[str, str]]


class BuildStats(ValidatedModel):
    """Counts from one build pass — logged by the CLI + asserted in tests."""

    input_rows: int
    emitted: int
    mechanism_from_vote: int
    mechanism_fallback: int
    justification_from_vote: int
    justification_missing: int


@validated
def source_id_of(gold: GoldPair) -> str:
    """Recover the tiered `source_id` from `gold.id` (`{src}__{tgt}::{source_id}`)."""
    _, sep, source_id = gold.id.partition("::")
    return source_id if sep else gold.id


def load_tiered_index(tiered_dir: Path) -> dict[str, dict[str, TieredPair]]:
    """`{framework_pair: {source_id: TieredPair}}` from `tiered_dir/*.jsonl`."""
    index: dict[str, dict[str, TieredPair]] = {}
    for path in sorted(tiered_dir.glob("*.jsonl")):
        rows: dict[str, TieredPair] = {}
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            tp = TieredPair.model_validate_json(line)
            rows[tp.source_id] = tp
        index[path.stem] = rows
    return index


def _consensus_field(row: TieredPair, field: str) -> str:
    """First non-empty `field` from a seat that voted the consensus citation.

    Used for both `target_mechanism` (the 1-2 sentence summary) and
    `mapping_justification` — same consensus-vote selection as
    `freeze.py::_consensus_vote_summary`.
    """
    norm = normalize_citation_id(row.consensus_citation_id) if row.consensus_citation_id else ""
    if not norm:
        return ""
    for v in row.votes:
        if v.citation_id_normalized == norm:
            value = getattr(v, field)
            if value:
                return value
    return ""


@validated
def build_example(gold: GoldPair, target_mechanism: str, justification: str) -> TrainingExample:
    """One `GoldPair` + recovered (summary, justification) → a 3-field SFT example.

    `citation_id` is the authoritative `GoldPair` target (the Tier-1 signal);
    `target_mechanism` + `justification` are the ensemble's concise text.
    """
    prompt = build_eval_prompt(gold)
    completion = json.dumps(
        {
            "citation_id": gold.target_citation_id,
            "target_mechanism": target_mechanism,
            "mapping_justification": justification,
        },
        ensure_ascii=False,
    )
    return TrainingExample(
        messages=[
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
            {"role": "assistant", "content": completion},
        ]
    )


@validated
def build_training_examples(
    gold_set: GoldSet,
    tiered_index: dict[str, dict[str, TieredPair]],
) -> tuple[list[TrainingExample], BuildStats]:
    """Project every `GoldPair` → `TrainingExample` using consensus-vote text.

    `target_mechanism` prefers the consensus vote's summary, falling back to the
    `GoldPair`'s (registry-body) mechanism when no vote summary exists. Missing
    justification is non-fatal (empty string). `citation_id` is always present.
    """
    examples: list[TrainingExample] = []
    mech_vote = mech_fallback = just_vote = just_missing = 0
    for gold in gold_set.pairs:
        framework_pair = f"{gold.source_framework}__{gold.target_framework}"
        tiered = tiered_index.get(framework_pair, {}).get(source_id_of(gold))
        summary = _consensus_field(tiered, "target_mechanism") if tiered is not None else ""
        justification = (
            _consensus_field(tiered, "mapping_justification") if tiered is not None else ""
        )
        if summary:
            mechanism = summary
            mech_vote += 1
        else:
            mechanism = gold.target_mechanism
            mech_fallback += 1
        if justification:
            just_vote += 1
        else:
            just_missing += 1
        examples.append(build_example(gold, mechanism, justification))
    stats = BuildStats(
        input_rows=len(gold_set.pairs),
        emitted=len(examples),
        mechanism_from_vote=mech_vote,
        mechanism_fallback=mech_fallback,
        justification_from_vote=just_vote,
        justification_missing=just_missing,
    )
    return examples, stats
