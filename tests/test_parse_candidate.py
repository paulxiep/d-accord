"""`_parse_candidate` robustness — reasoning-model scaffolding must not fail a
valid answer. Qwen3 emits `<think>...</think>` before the JSON; without stripping
it, every fine-tune row scores as a parse error at M4 eval. See
`daccord.eval.clients`.
"""

from __future__ import annotations

import json

from daccord.eval.clients import _parse_candidate

_OBJ = {
    "citation_id": "5",
    "target_mechanism": "Security of processing.",
    "mapping_justification": "Both require appropriate technical measures.",
}
_JSON = json.dumps(_OBJ)


class TestParseCandidate:
    def test_plain_json(self) -> None:
        cand, err = _parse_candidate(_JSON)
        assert err is None
        assert cand is not None and cand.citation_id == "5"

    def test_empty_think_block_prefix(self) -> None:
        # the exact shape Qwen3 produced in the M3 sanity run
        cand, err = _parse_candidate(f"<think>\n\n</think>\n\n{_JSON}")
        assert err is None
        assert cand is not None and cand.citation_id == "5"

    def test_nonempty_think_block_prefix(self) -> None:
        raw = f"<think>Map by purpose-limitation.</think>\n{_JSON}"
        cand, err = _parse_candidate(raw)
        assert err is None
        assert cand is not None and cand.target_mechanism == "Security of processing."

    def test_code_fenced_json(self) -> None:
        cand, err = _parse_candidate(f"```json\n{_JSON}\n```")
        assert err is None
        assert cand is not None and cand.citation_id == "5"

    def test_think_block_plus_fence(self) -> None:
        cand, err = _parse_candidate(f"<think></think>\n```json\n{_JSON}\n```")
        assert err is None
        assert cand is not None

    def test_garbage_still_errors(self) -> None:
        cand, err = _parse_candidate("<think>no answer</think> sorry, no idea")
        assert cand is None
        assert err is not None
