"""Tier-10A training-data build tests — the gold → SFT-example projection.

Covers the join that recovers `mapping_justification` from the consensus
ensemble vote, the 3-field completion shape, and graceful handling when no
tiered match exists. See `daccord.gold.training_data` + [docs/m3_gate.md].
"""

from __future__ import annotations

import json
from pathlib import Path

from daccord.ensemble.schema import ModelVote, TieredPair
from daccord.eval.scoring import normalize_citation_id
from daccord.gold.schema import GoldPair, GoldSet
from daccord.gold.training_data import (
    TrainingExample,
    build_example,
    build_training_examples,
    load_tiered_index,
    source_id_of,
)


def _gold(source_id: str = "gdpr-1", **over: object) -> GoldPair:
    base: dict[str, object] = {
        "id": f"gdpr__pdpa_sg::{source_id}",
        "source_jurisdiction": "eu",
        "source_framework": "gdpr",
        "source_citation_id": "Article 5",
        "source_mechanism": "Principles relating to processing of personal data.",
        "source_language": "en",
        "target_jurisdiction": "sg",
        "target_framework": "pdpa_sg",
        "target_citation_id": "Section 13",
        "target_mechanism": "Consent required before collection, use or disclosure.",
        "target_language": "en",
        "notes": "provisional M2 freeze — auto-promoted; NOT hand-validated",
    }
    base.update(over)
    return GoldPair.model_validate(base)


def _vote(citation_id: str, justification: str, model: str = "claude-haiku-4-5") -> ModelVote:
    return ModelVote(
        model=model,
        citation_id_raw=citation_id,
        citation_id_normalized=normalize_citation_id(citation_id),
        target_mechanism="Consent is the lawful basis.",
        mapping_justification=justification,
    )


def _tiered(source_id: str, consensus: str, votes: list[ModelVote]) -> TieredPair:
    return TieredPair(
        source_id=source_id,
        source_jurisdiction="eu",
        source_framework="gdpr",
        source_citation_id="Article 5",
        source_mechanism="Principles relating to processing of personal data.",
        target_jurisdiction="sg",
        target_framework="pdpa_sg",
        tier="HIGH",
        consensus_citation_id=consensus,
        valid_vote_count=len(votes),
        consensus_vote_count=len(votes),
        agreement_score=1.0,
        votes=votes,
    )


class TestSourceIdOf:
    def test_parses_source_id_from_composite_gold_id(self) -> None:
        assert source_id_of(_gold("gdpr-1")) == "gdpr-1"

    def test_falls_back_to_whole_id_when_no_separator(self) -> None:
        assert source_id_of(_gold(id="legacy-noprefix")) == "legacy-noprefix"


class TestBuildExample:
    def test_three_field_completion_is_valid_json(self) -> None:
        ex = build_example(_gold(), justification="Both gate processing on consent.")
        assert isinstance(ex, TrainingExample)
        roles = [m["role"] for m in ex.messages]
        assert roles == ["system", "user", "assistant"]

        completion = json.loads(ex.messages[2]["content"])
        assert set(completion) == {"citation_id", "target_mechanism", "mapping_justification"}
        assert completion["citation_id"] == "Section 13"
        assert completion["target_mechanism"].startswith("Consent required")
        assert completion["mapping_justification"] == "Both gate processing on consent."

    def test_prompt_matches_eval_prompt(self) -> None:
        from daccord.eval.prompts import build_eval_prompt

        gold = _gold()
        ex = build_example(gold, justification="x")
        expected = build_eval_prompt(gold)
        assert ex.messages[0]["content"] == expected.system
        assert ex.messages[1]["content"] == expected.user

    def test_empty_justification_still_emits(self) -> None:
        ex = build_example(_gold(), justification="")
        assert json.loads(ex.messages[2]["content"])["mapping_justification"] == ""

    def test_round_trip_model_dump_json(self) -> None:
        ex = build_example(_gold(), justification="j")
        reloaded = TrainingExample.model_validate_json(ex.model_dump_json())
        assert reloaded.messages == ex.messages


class TestBuildTrainingExamples:
    def _index(self, tmp_path: Path, rows: list[TieredPair]) -> dict[str, dict[str, TieredPair]]:
        tiered_dir = tmp_path / "tiered"
        tiered_dir.mkdir()
        (tiered_dir / "gdpr__pdpa_sg.jsonl").write_text(
            "".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8"
        )
        return load_tiered_index(tiered_dir)

    def _gold_set(self, pairs: list[GoldPair]) -> GoldSet:
        return GoldSet(pairs=pairs, dataset_hash="deadbeef", source_path="mem")

    def test_justification_recovered_from_consensus_vote(self, tmp_path: Path) -> None:
        votes = [
            _vote("Section 13", "Consensus seat reasoning."),
            _vote("Section 99", "Dissenting seat reasoning.", model="gpt-5-mini"),
        ]
        index = self._index(tmp_path, [_tiered("gdpr-1", "Section 13", votes)])
        examples, stats = build_training_examples(self._gold_set([_gold("gdpr-1")]), index)

        assert stats.justification_from_vote == 1
        assert stats.justification_missing == 0
        completion = json.loads(examples[0].messages[2]["content"])
        assert completion["mapping_justification"] == "Consensus seat reasoning."

    def test_missing_tiered_match_emits_with_empty_justification(self, tmp_path: Path) -> None:
        index = self._index(tmp_path, [_tiered("gdpr-1", "Section 13", [_vote("Section 13", "j")])])
        # gold-2 has no tiered row → missing, but still emitted
        examples, stats = build_training_examples(self._gold_set([_gold("gdpr-2")]), index)

        assert stats.emitted == 1
        assert stats.justification_from_vote == 0
        assert stats.justification_missing == 1
        assert json.loads(examples[0].messages[2]["content"])["mapping_justification"] == ""

    def test_consensus_with_no_voting_seat_justification_is_missing(self, tmp_path: Path) -> None:
        # consensus citation has no seat carrying a non-empty justification for it
        votes = [_vote("Section 13", "")]
        index = self._index(tmp_path, [_tiered("gdpr-1", "Section 13", votes)])
        _, stats = build_training_examples(self._gold_set([_gold("gdpr-1")]), index)
        assert stats.justification_missing == 1

    def test_stats_total_accounting(self, tmp_path: Path) -> None:
        index = self._index(tmp_path, [_tiered("gdpr-1", "Section 13", [_vote("Section 13", "j")])])
        gold_set = self._gold_set([_gold("gdpr-1"), _gold("gdpr-2")])
        examples, stats = build_training_examples(gold_set, index)
        assert stats.input_rows == stats.emitted == len(examples) == 2
        assert stats.justification_from_vote + stats.justification_missing == 2
