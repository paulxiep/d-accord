"""Tier-10A training-data build — project frozen `GoldPair` rows into
chat-format SFT examples for the QLoRA trainer.

The trainer ([training/train.py]) consumes a single clean file per split —
`data/training/{split}.jsonl`, one `{"messages": [...]}` row each. This
module does the join the trainer should not: it recovers the
`mapping_justification` (absent from the frozen `GoldPair` — its `notes`
field holds the freeze provenance string) from the consensus ensemble vote
in `data/ensemble/tiered/{framework_pair}.jsonl`, so the supervised target
is the same 3-field JSON object the eval prompt asks the model to produce.

The prompt (system + user) is built by `build_eval_prompt` — the *exact*
prompt the M0/M4 eval harness and `LocalAdapterClient` feed the model — so
train-time and eval-time inputs can never drift. See [docs/m3_gate.md].

Logic lives here (root-env importable, unit-tested in `tests/`);
`scripts/build_training_data.py` is the thin CLI. Pattern mirrors
`daccord.gold.freeze` + `scripts/freeze_gold.py`.
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

    The assistant turn is the supervised target — `SFTTrainer`'s
    `assistant_only_loss` masks the system+user tokens, so loss is computed
    on the JSON answer alone.
    """

    messages: list[dict[str, str]]


class BuildStats(ValidatedModel):
    """Counts from one build pass — logged by the CLI + asserted in tests."""

    input_rows: int
    emitted: int
    justification_from_vote: int
    justification_missing: int


@validated
def source_id_of(gold: GoldPair) -> str:
    """Recover the tiered `source_id` from `gold.id` (`{src}__{tgt}::{source_id}`).

    `GoldPair` drops `source_id` (freeze.py embeds it in the composite id);
    the tiered files are keyed by it, so we parse it back out for the join.
    """
    _, sep, source_id = gold.id.partition("::")
    return source_id if sep else gold.id


def load_tiered_index(tiered_dir: Path) -> dict[str, dict[str, TieredPair]]:
    """`{framework_pair: {source_id: TieredPair}}` from `tiered_dir/*.jsonl`.

    The file stem is the framework_pair (`gdpr__pdpa_sg`), matching the
    `{source_framework}__{target_framework}` key reconstructed per gold row.
    """
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


def _consensus_justification(row: TieredPair) -> str:
    """`mapping_justification` from a seat that voted the consensus citation.

    Sibling of `freeze.py::_consensus_vote_summary` (which pulls
    `target_mechanism`); same consensus-vote selection, different field.
    """
    norm = normalize_citation_id(row.consensus_citation_id) if row.consensus_citation_id else ""
    if not norm:
        return ""
    for v in row.votes:
        if v.citation_id_normalized == norm and v.mapping_justification:
            return v.mapping_justification
    return ""


@validated
def build_example(gold: GoldPair, justification: str) -> TrainingExample:
    """One `GoldPair` + its recovered justification → a 3-field SFT example."""
    prompt = build_eval_prompt(gold)
    completion = json.dumps(
        {
            "citation_id": gold.target_citation_id,
            "target_mechanism": gold.target_mechanism,
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
    """Project every `GoldPair` in `gold_set` → `TrainingExample`.

    Missing justification (no tiered match, or no consensus-voting seat
    carried one) is not fatal — the row is still emitted with an empty
    justification and counted in `justification_missing`. Citation +
    mechanism, the gold-grounded fields, are always present.
    """
    examples: list[TrainingExample] = []
    from_vote = 0
    missing = 0
    for gold in gold_set.pairs:
        framework_pair = f"{gold.source_framework}__{gold.target_framework}"
        tiered = tiered_index.get(framework_pair, {}).get(source_id_of(gold))
        justification = _consensus_justification(tiered) if tiered is not None else ""
        if justification:
            from_vote += 1
        else:
            missing += 1
        examples.append(build_example(gold, justification))
    stats = BuildStats(
        input_rows=len(gold_set.pairs),
        emitted=len(examples),
        justification_from_vote=from_vote,
        justification_missing=missing,
    )
    return examples, stats
