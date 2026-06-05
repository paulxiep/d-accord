"""Tier-9 gold freeze CLI — project tier-7B splits into the gold-set shape.

Reads the jurisdiction-disjoint splits written by `scripts/build_splits.py`
(`data/splits/{train,val,test}.jsonl`, each a `TieredPair` per row) and emits:

  - `data/gold/{train,val,test}.jsonl` — one `GoldPair` per row (trainer-ready)
  - `data/gold/gold_v1.jsonl`          — union of the three, sorted by id (the
                                         M2 artifact; its SHA256 is the dataset hash)
  - `data/gold/gold_v1_manifest.json`  — freeze stats + dataset SHA + provenance

Target clause text is resolved from `data/clauses/*.json` (registry body) with a
consensus-vote-summary fallback; rows whose consensus citation is not in the
target registry are excluded. See `daccord.gold.freeze` for the rules.

Provisional M2 freeze: gold rows are AUTO-PROMOTED (HIGH + bidirectional-consistent
MED + rag-concurs MED) with ZERO hand-validation. Every row says so in `notes`.
Revisit via the tier-7C labeler if M3/M4 quality bites. See development_plan.md M2.

Run (docker):
    docker compose run --rm root uv run python scripts/freeze_gold.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from daccord.ensemble.schema import TieredPair
from daccord.gold.freeze import (
    FreezeStats,
    freeze_gold_rows,
    load_clause_bodies,
    load_registry_ids,
)
from daccord.gold.schema import GoldPair
from daccord.tracking import compute_file_sha256

log = logging.getLogger("freeze_gold")

SPLIT_NAMES = ("train", "val", "test")


def _read_tiered_file(path: Path) -> list[TieredPair]:
    if not path.exists():
        raise FileNotFoundError(f"split file missing: {path} — run scripts/build_splits.py first")
    rows: list[TieredPair] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            rows.append(TieredPair.model_validate_json(line))
        except Exception as exc:  # noqa: BLE001 — annotate the offending line
            raise ValueError(f"{path}:{lineno}: invalid TieredPair row: {exc}") from exc
    return rows


def _write_gold(path: Path, rows: list[GoldPair]) -> None:
    ordered = sorted(rows, key=lambda r: r.id)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in ordered:
            f.write(row.model_dump_json())
            f.write("\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    clause_bodies = load_clause_bodies(args.clauses_dir)
    registry_ids = load_registry_ids(args.registry_dir)

    per_split: dict[str, tuple[list[GoldPair], FreezeStats]] = {}
    union: list[GoldPair] = []
    for name in SPLIT_NAMES:
        rows = _read_tiered_file(args.splits_dir / f"{name}.jsonl")
        gold, stats = freeze_gold_rows(rows, clause_bodies=clause_bodies, registry_ids=registry_ids)
        per_split[name] = (gold, stats)
        union.extend(gold)
        log.info(
            "%-5s  in=%4d  emitted=%4d  (registry=%4d  summary=%3d)  "
            "excluded: non_registry=%d no_text=%d",
            name,
            stats.input_rows,
            stats.emitted,
            stats.target_from_registry,
            stats.target_from_ensemble_summary,
            stats.excluded_non_registry,
            stats.excluded_no_target_text,
        )

    total = len(union)
    log.info("TOTAL gold pairs: %d", total)
    if total < 500:
        log.error("gold pool %d < 500 (M2 floor) — aborting", total)
        return 1

    if args.dry_run:
        log.info("--dry-run: no files written")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, (gold, _) in per_split.items():
        _write_gold(args.out_dir / f"{name}.jsonl", gold)
    gold_v1 = args.out_dir / "gold_v1.jsonl"
    _write_gold(gold_v1, union)
    dataset_sha = compute_file_sha256(gold_v1)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "gold_v1_sha256": dataset_sha,
        "total_pairs": total,
        "hand_validated": False,
        "provenance": (
            "AUTO-PROMOTED from tier-7B splits (HIGH + bidirectional-consistent MED "
            "+ rag-concurs MED); ZERO hand-validation (provisional M2 freeze). "
            "134 HIGH+inconsistent rows ship unreviewed — see development_plan.md M2."
        ),
        "splits": {
            name: {
                "input_rows": stats.input_rows,
                "emitted": stats.emitted,
                "target_from_registry": stats.target_from_registry,
                "target_from_ensemble_summary": stats.target_from_ensemble_summary,
                "excluded_non_registry": stats.excluded_non_registry,
                "excluded_no_target_text": stats.excluded_no_target_text,
            }
            for name, (_, stats) in per_split.items()
        },
    }
    (args.out_dir / "gold_v1_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log.info("Wrote %s/{train,val,test}.jsonl + gold_v1.jsonl + manifest", args.out_dir)
    log.info("dataset SHA256 (gold_v1.jsonl) = %s", dataset_sha)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--clauses-dir", type=Path, default=Path("data/clauses"))
    parser.add_argument("--registry-dir", type=Path, default=Path("data/registry"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/gold"))
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute + print stats; don't write files."
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
