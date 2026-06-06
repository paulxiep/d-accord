"""Chat-example loading + deterministic subsetting for the small run.

Reads `data/training/{split}.jsonl` (built by `scripts/build_training_data.py`),
each row `{"messages": [system, user, assistant]}`. Torch-free so it stays
unit-testable in the lightweight root env.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def load_chat_examples(path: Path) -> list[dict[str, object]]:
    """Load `{"messages": [...]}` rows from a JSONL file, skipping blank lines."""
    examples: list[dict[str, object]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        examples.append(json.loads(line))
    return examples


def subset(examples: list[dict[str, object]], n: int, seed: int) -> list[dict[str, object]]:
    """Deterministic seeded `n`-row subsample, returned in original on-disk order.

    `n <= 0` or `n >= len` → return all (a copy). The seeded `sample` picks
    *which* rows; re-sorting by original index keeps batching reproducible.
    """
    if n <= 0 or n >= len(examples):
        return list(examples)
    picked = sorted(random.Random(seed).sample(range(len(examples)), n))
    return [examples[i] for i in picked]
