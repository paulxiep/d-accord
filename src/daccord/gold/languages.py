"""Framework → jurisdiction + primary citation-text language maps.

Single source of truth shared by tier-2A toy-gold drafting
(`envs/eval/scripts/draft_toy_gold.py`) and the tier-9 gold freeze
(`daccord.gold.freeze`). Kept here (in `daccord.gold`, not the eval env)
so the freeze can fill `GoldPair.{source,target}_language` without importing
a script.

`FRAMEWORK_TO_JURISDICTION` codes match the split partitions in
`daccord.ensemble.splits` (test={th,ph}, val={my}). `FRAMEWORK_TO_LANGUAGE`
are ISO 639-1 codes, used for the per-language metric breakdown that
quantifies the SEA/FR/DE differentiation (development_plan.md §5).
"""

from __future__ import annotations

FRAMEWORK_TO_JURISDICTION: dict[str, str] = {
    "gdpr": "eu",
    "uk_gdpr": "uk",
    "dpa_2018": "uk",
    "bdsg": "de",
    "loi_il": "fr",
    "pdpa_sg": "sg",
    "pdpa_th": "th",
    "dpa_2012_ph": "ph",
    "pdpa_my": "my",
}

# Per-framework primary citation-text language (ISO 639-1).
# pdpa_th/bdsg/loi_il can also be cited in English translations — drafter picks per-pair.
FRAMEWORK_TO_LANGUAGE: dict[str, str] = {
    "gdpr": "en",
    "uk_gdpr": "en",
    "dpa_2018": "en",
    "bdsg": "de",
    "loi_il": "fr",
    "pdpa_sg": "en",
    "pdpa_th": "th",
    "dpa_2012_ph": "en",
    "pdpa_my": "en",
}
