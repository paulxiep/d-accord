"""Tier-9 gold freeze — project tier-6B `TieredPair` rows into `GoldPair`.

`scripts/build_splits.py` partitions gold-eligible tiered rows into
jurisdiction-disjoint train/val/test (still as `TieredPair`). This module
converts those into the canonical `GoldPair` shape consumed by the tier-10A
trainer and the tier-2B eval harness — resolving the **target clause body**
that a `TieredPair` does not carry (it only has the consensus citation_id).

Target-mechanism resolution, per row, in priority order:
  1. **Registry clause body** — `data/clauses/{target_framework}.json` keyed by
     the normalized consensus citation_id. Authoritative law text.
  2. **Consensus-vote summary fallback** — when the citation is in the target
     registry but its body text was not extracted (`body_recall < 1.0`, a
     documented gap; see `daccord.registry.clauses`), use the ensemble seat's
     one-to-two sentence `target_mechanism` for the consensus citation. Flagged
     `target_text=ensemble-summary` in `GoldPair.notes`.
  3. **Exclude** — when the consensus citation_id is NOT in the target registry
     at all (the agreed answer cites something outside our citation universe; a
     registry-constrained gold answer would be incoherent). Counted in
     `FreezeStats.excluded_non_registry`, never silently dropped.

Provenance: every emitted row's `notes` records that it was auto-promoted and
NOT hand-validated (provisional M2 freeze). See development_plan.md M2.
"""

from __future__ import annotations

import json
from pathlib import Path

from daccord.ensemble.schema import TieredPair
from daccord.eval.scoring import normalize_citation_id
from daccord.gold.languages import FRAMEWORK_TO_LANGUAGE
from daccord.gold.schema import GoldPair
from daccord.validation import ValidatedModel, validated


class FreezeStats(ValidatedModel):
    """Counts from one freeze pass — printed by the CLI + asserted in tests."""

    input_rows: int
    emitted: int
    excluded_non_registry: int
    excluded_no_target_text: int
    target_from_registry: int
    target_from_ensemble_summary: int


def load_clause_bodies(clauses_dir: Path) -> dict[str, dict[str, str]]:
    """`{framework: {normalized_citation_id: body_text}}` from data/clauses/*.json."""
    out: dict[str, dict[str, str]] = {}
    for path in sorted(clauses_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        bodies: dict[str, str] = {}
        for cid, body in data["clauses"].items():
            norm = normalize_citation_id(cid)
            if norm and body:
                bodies.setdefault(norm, body)
        out[data["framework"]] = bodies
    return out


def load_registry_ids(registry_dir: Path) -> dict[str, set[str]]:
    """`{framework: {normalized_citation_id}}` from data/registry/*.json.

    Skips non-registry files (manifest.jsonl, summary.md) that lack
    `citation_ids`.
    """
    out: dict[str, set[str]] = {}
    for path in sorted(registry_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "citation_ids" not in data:
            continue
        out[data["framework"]] = {
            n for c in data["citation_ids"] if (n := normalize_citation_id(c))
        }
    return out


def gold_id(row: TieredPair) -> str:
    """Globally-unique id — `source_id` alone collides across pairs.

    `gdpr-1` appears in `gdpr__pdpa_sg` and `gdpr__pdpa_my`; the pair prefix
    de-collides it so the gold set has unique ids.
    """
    return f"{row.source_framework}__{row.target_framework}::{row.source_id}"


def _consensus_vote_summary(row: TieredPair, norm_consensus: str) -> str:
    """The `target_mechanism` summary from a seat that voted the consensus citation."""
    for v in row.votes:
        if v.citation_id_normalized == norm_consensus and v.target_mechanism:
            return v.target_mechanism
    return ""


def _provenance_note(row: TieredPair, target_source: str) -> str:
    return (
        f"provisional M2 freeze — auto-promoted "
        f"(tier={row.tier}, agreement={row.agreement_score:.2f}, "
        f"rag_concurs={row.rag_concurs}); NOT hand-validated; "
        f"target_text={target_source}"
    )


def _to_gold_pair(
    row: TieredPair,
    clause_bodies: dict[str, dict[str, str]],
    registry_ids: dict[str, set[str]],
) -> tuple[GoldPair | None, str]:
    """Project one tiered row → (GoldPair, target_source) or (None, exclude_reason)."""
    norm = normalize_citation_id(row.consensus_citation_id) if row.consensus_citation_id else ""
    if not norm or norm not in registry_ids.get(row.target_framework, set()):
        return None, "non_registry"
    body = clause_bodies.get(row.target_framework, {}).get(norm)
    if body:
        target_mechanism, target_source = body, "registry"
    else:
        target_mechanism = _consensus_vote_summary(row, norm)
        if not target_mechanism:
            return None, "no_target_text"
        target_source = "ensemble-summary"
    pair = GoldPair(
        id=gold_id(row),
        source_jurisdiction=row.source_jurisdiction,
        source_framework=row.source_framework,
        source_citation_id=row.source_citation_id,
        source_mechanism=row.source_mechanism,
        source_language=FRAMEWORK_TO_LANGUAGE[row.source_framework],
        target_jurisdiction=row.target_jurisdiction,
        target_framework=row.target_framework,
        target_citation_id=row.consensus_citation_id,
        target_mechanism=target_mechanism,
        target_language=FRAMEWORK_TO_LANGUAGE[row.target_framework],
        notes=_provenance_note(row, target_source),
    )
    return pair, target_source


@validated
def freeze_gold_rows(
    rows: list[TieredPair],
    *,
    clause_bodies: dict[str, dict[str, str]],
    registry_ids: dict[str, set[str]],
) -> tuple[list[GoldPair], FreezeStats]:
    """Convert gold-eligible tiered rows → GoldPair, with exclusion accounting.

    Rows are returned in input order; the CLI sorts the union by id for a
    stable dataset SHA.
    """
    gold: list[GoldPair] = []
    excluded_non_registry = 0
    excluded_no_target_text = 0
    from_registry = 0
    from_summary = 0
    for row in rows:
        pair, source = _to_gold_pair(row, clause_bodies, registry_ids)
        if pair is None:
            if source == "non_registry":
                excluded_non_registry += 1
            else:
                excluded_no_target_text += 1
            continue
        gold.append(pair)
        if source == "registry":
            from_registry += 1
        else:
            from_summary += 1
    stats = FreezeStats(
        input_rows=len(rows),
        emitted=len(gold),
        excluded_non_registry=excluded_non_registry,
        excluded_no_target_text=excluded_no_target_text,
        target_from_registry=from_registry,
        target_from_ensemble_summary=from_summary,
    )
    return gold, stats
