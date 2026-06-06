"""Tier-10A training-data build CLI — frozen gold splits → SFT chat examples.

Reads `data/gold/{train,val,test}.jsonl` (the M2 freeze) and emits
`data/training/{train,val,test}.jsonl`, one `{"messages": [...]}` row each,
ready for `training/train.py`. The `mapping_justification` is recovered by
joining each gold row back to its consensus vote in
`data/ensemble/tiered/`; see `daccord.gold.training_data` and
[docs/m3_gate.md].

Idempotent (pure function of committed gold + tiered data). Prints the
justification-coverage stat per split.

Run (docker):
    docker compose run --rm root uv run python scripts/build_training_data.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from daccord.gold.schema import GoldSet
from daccord.gold.training_data import (
    TrainingExample,
    build_training_examples,
    load_tiered_index,
)

log = logging.getLogger("build_training_data")

SPLIT_NAMES = ("train", "val", "test")


def _write(path: Path, examples: list[TrainingExample]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json())
            f.write("\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    tiered_index = load_tiered_index(args.tiered_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    grand_rows = grand_mech = grand_just = 0
    for name in SPLIT_NAMES:
        gold_path = args.gold_dir / f"{name}.jsonl"
        if not gold_path.exists():
            raise FileNotFoundError(f"{gold_path} missing — run scripts/freeze_gold.py first")
        gold_set = GoldSet.from_jsonl(gold_path)
        examples, stats = build_training_examples(gold_set, tiered_index)
        log.info(
            "%-5s  rows=%4d  mechanism: vote=%4d fallback=%4d  justification: vote=%4d missing=%4d",
            name,
            stats.input_rows,
            stats.mechanism_from_vote,
            stats.mechanism_fallback,
            stats.justification_from_vote,
            stats.justification_missing,
        )
        grand_rows += stats.input_rows
        grand_mech += stats.mechanism_from_vote
        grand_just += stats.justification_from_vote
        if not args.dry_run:
            _write(args.out_dir / f"{name}.jsonl", examples)

    log.info(
        "TOTAL rows=%d  summary-target coverage=%.1f%%  justification coverage=%.1f%%",
        grand_rows,
        100.0 * grand_mech / max(grand_rows, 1),
        100.0 * grand_just / max(grand_rows, 1),
    )
    if args.dry_run:
        log.info("--dry-run: no files written")
    else:
        log.info("Wrote %s/{train,val,test}.jsonl", args.out_dir)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", type=Path, default=Path("data/gold"))
    parser.add_argument("--tiered-dir", type=Path, default=Path("data/ensemble/tiered"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/training"))
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute + print stats; don't write files."
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
