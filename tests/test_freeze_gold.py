"""Tier-9 gold-freeze guards.

Pins the TieredPair→GoldPair projection: registry-body resolution, the
consensus-vote-summary fallback for body-recall misses, exclusion of
non-registry consensus citations, id de-collision across pairs, and the
language mapping.
"""

from __future__ import annotations

import json
from pathlib import Path

from daccord.ensemble.schema import ModelVote, Tier, TieredPair
from daccord.gold.freeze import (
    freeze_gold_rows,
    gold_id,
    load_clause_bodies,
    load_registry_ids,
)


def _make_pair(
    source_id: str,
    *,
    source_framework: str = "gdpr",
    source_jurisdiction: str = "eu",
    target_framework: str = "pdpa_sg",
    target_jurisdiction: str = "sg",
    tier: Tier = "HIGH",
    consensus_citation_id: str = "24",
    agreement_score: float = 1.0,
    rag_concurs: bool = False,
    vote_target_mechanism: str = "Singapore PDPA security obligation.",
    vote_citation_normalized: str = "24",
) -> TieredPair:
    return TieredPair(
        source_id=source_id,
        source_jurisdiction=source_jurisdiction,
        source_framework=source_framework,
        source_citation_id="Art. 32",
        source_mechanism="The controller shall implement appropriate measures.",
        target_jurisdiction=target_jurisdiction,
        target_framework=target_framework,
        tier=tier,
        consensus_citation_id=consensus_citation_id,
        valid_vote_count=4,
        consensus_vote_count=4,
        agreement_score=agreement_score,
        votes=[
            ModelVote(
                model="m1",
                citation_id_raw=vote_citation_normalized,
                citation_id_normalized=vote_citation_normalized,
                target_mechanism=vote_target_mechanism,
                mapping_justification="analogous security duty",
            )
        ],
        rag_concurs=rag_concurs,
    )


# Minimal in-memory registry/clause fixtures: pdpa_sg has citation 24 with body,
# citation 26 in registry but NO body (body-recall miss). 99 is in nothing.
_REGISTRY = {"pdpa_sg": {"24", "26"}, "gdpr": {"32"}}
_CLAUSES = {"pdpa_sg": {"24": "24.—(1) An organisation shall protect personal data..."}}


class TestRegistryResolution:
    def test_registry_body_becomes_target_mechanism(self) -> None:
        rows = [_make_pair("gdpr-1", consensus_citation_id="24")]
        gold, stats = freeze_gold_rows(rows, clause_bodies=_CLAUSES, registry_ids=_REGISTRY)
        assert stats.emitted == 1
        assert stats.target_from_registry == 1
        assert stats.target_from_ensemble_summary == 0
        g = gold[0]
        assert g.target_citation_id == "24"
        assert g.target_mechanism.startswith("24.—(1)")
        assert "target_text=registry" in (g.notes or "")
        assert "NOT hand-validated" in (g.notes or "")

    def test_body_recall_miss_falls_back_to_vote_summary(self) -> None:
        # citation 26 is in the registry but has no extracted clause body.
        rows = [
            _make_pair(
                "gdpr-2",
                consensus_citation_id="26",
                vote_citation_normalized="26",
                vote_target_mechanism="PDPA consent withdrawal mechanism.",
            )
        ]
        gold, stats = freeze_gold_rows(rows, clause_bodies=_CLAUSES, registry_ids=_REGISTRY)
        assert stats.emitted == 1
        assert stats.target_from_ensemble_summary == 1
        assert gold[0].target_mechanism == "PDPA consent withdrawal mechanism."
        assert "target_text=ensemble-summary" in (gold[0].notes or "")

    def test_non_registry_consensus_is_excluded(self) -> None:
        rows = [_make_pair("gdpr-3", consensus_citation_id="99")]
        gold, stats = freeze_gold_rows(rows, clause_bodies=_CLAUSES, registry_ids=_REGISTRY)
        assert gold == []
        assert stats.excluded_non_registry == 1
        assert stats.emitted == 0

    def test_empty_consensus_is_excluded(self) -> None:
        rows = [_make_pair("gdpr-4", consensus_citation_id="")]
        gold, stats = freeze_gold_rows(rows, clause_bodies=_CLAUSES, registry_ids=_REGISTRY)
        assert gold == []
        assert stats.excluded_non_registry == 1

    def test_in_registry_no_body_no_vote_summary_excluded(self) -> None:
        # citation 26 in registry, no body, and the only vote summary is for a
        # different citation → nothing to fall back on.
        rows = [
            _make_pair(
                "gdpr-5",
                consensus_citation_id="26",
                vote_citation_normalized="24",
                vote_target_mechanism="wrong-citation summary",
            )
        ]
        gold, stats = freeze_gold_rows(rows, clause_bodies=_CLAUSES, registry_ids=_REGISTRY)
        assert gold == []
        assert stats.excluded_no_target_text == 1


class TestIdAndLanguages:
    def test_id_decollides_same_source_across_pairs(self) -> None:
        rows = [
            _make_pair("gdpr-1", target_framework="pdpa_sg", target_jurisdiction="sg"),
            _make_pair(
                "gdpr-1",
                target_framework="pdpa_my",
                target_jurisdiction="my",
                consensus_citation_id="24",
            ),
        ]
        registry = {"pdpa_sg": {"24"}, "pdpa_my": {"24"}}
        clauses = {"pdpa_sg": {"24": "SG body"}, "pdpa_my": {"24": "MY body"}}
        gold, stats = freeze_gold_rows(rows, clause_bodies=clauses, registry_ids=registry)
        assert stats.emitted == 2
        ids = {g.id for g in gold}
        assert ids == {"gdpr__pdpa_sg::gdpr-1", "gdpr__pdpa_my::gdpr-1"}

    def test_languages_filled_from_framework_map(self) -> None:
        # bdsg (de) → pdpa_th (th)
        rows = [
            _make_pair(
                "bdsg-1",
                source_framework="bdsg",
                source_jurisdiction="de",
                target_framework="pdpa_th",
                target_jurisdiction="th",
                consensus_citation_id="20",
            )
        ]
        registry = {"pdpa_th": {"20"}}
        clauses = {"pdpa_th": {"20": "Thai section 20 body"}}
        gold, _ = freeze_gold_rows(rows, clause_bodies=clauses, registry_ids=registry)
        assert gold[0].source_language == "de"
        assert gold[0].target_language == "th"

    def test_gold_id_helper(self) -> None:
        row = _make_pair("gdpr-7", target_framework="pdpa_sg")
        assert gold_id(row) == "gdpr__pdpa_sg::gdpr-7"


class TestLoaders:
    def test_load_clause_bodies_normalizes_keys(self, tmp_path: Path) -> None:
        d = tmp_path / "clauses"
        d.mkdir()
        (d / "pdpa_sg.json").write_text(
            json.dumps(
                {
                    "framework": "pdpa_sg",
                    "jurisdiction": "sg",
                    "clauses": {"24": "body 24", "26": ""},
                }
            ),
            encoding="utf-8",
        )
        bodies = load_clause_bodies(d)
        assert bodies["pdpa_sg"]["24"] == "body 24"
        # empty body is dropped
        assert "26" not in bodies["pdpa_sg"]

    def test_load_registry_ids_skips_non_registry_files(self, tmp_path: Path) -> None:
        d = tmp_path / "registry"
        d.mkdir()
        (d / "pdpa_sg.json").write_text(
            json.dumps(
                {"framework": "pdpa_sg", "jurisdiction": "sg", "citation_ids": ["24", "26"]}
            ),
            encoding="utf-8",
        )
        # a non-registry json (no citation_ids) must be skipped, not crash
        (d / "summary.json").write_text(json.dumps({"note": "not a registry"}), encoding="utf-8")
        ids = load_registry_ids(d)
        assert ids == {"pdpa_sg": {"24", "26"}}
