"""Chat-example load + deterministic-subset tests. Torch-free."""

from __future__ import annotations

import json
from pathlib import Path

from data import load_chat_examples, subset


def _write(path: Path, n: int) -> None:
    path.write_text(
        "".join(
            json.dumps({"messages": [{"role": "user", "content": str(i)}]}) + "\n" for i in range(n)
        ),
        encoding="utf-8",
    )


def _order(rows: list[dict[str, object]]) -> list[int]:
    out: list[int] = []
    for r in rows:
        messages = r["messages"]
        assert isinstance(messages, list)
        out.append(int(messages[0]["content"]))
    return out


class TestLoad:
    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "t.jsonl"
        path.write_text('{"messages": []}\n\n{"messages": []}\n', encoding="utf-8")
        assert len(load_chat_examples(path)) == 2


class TestSubset:
    def test_is_deterministic_for_a_seed(self, tmp_path: Path) -> None:
        path = tmp_path / "t.jsonl"
        _write(path, 100)
        rows = load_chat_examples(path)
        assert subset(rows, 10, seed=42) == subset(rows, 10, seed=42)
        assert len(subset(rows, 10, seed=42)) == 10

    def test_preserves_on_disk_order(self, tmp_path: Path) -> None:
        path = tmp_path / "t.jsonl"
        _write(path, 50)
        order = _order(subset(load_chat_examples(path), 10, seed=1))
        assert order == sorted(order)

    def test_n_at_or_above_len_returns_all(self, tmp_path: Path) -> None:
        path = tmp_path / "t.jsonl"
        _write(path, 5)
        rows = load_chat_examples(path)
        assert len(subset(rows, 10, seed=1)) == 5
        assert len(subset(rows, 0, seed=1)) == 5

    def test_different_seed_picks_different_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "t.jsonl"
        _write(path, 200)
        rows = load_chat_examples(path)
        assert _order(subset(rows, 20, seed=1)) != _order(subset(rows, 20, seed=2))
