"""Trainer-config assertions — completion-only loss, MLflow callback, all-linear
LoRA. Constructs LoraConfig/SFTConfig objects (no GPU, no model load).

Skipped where peft/trl aren't installed (e.g. the lightweight root env); runs
in the `training` docker service.
"""

from __future__ import annotations

import pytest

pytest.importorskip("peft")
pytest.importorskip("trl")

from config import SmallRunConfig  # noqa: E402
from train import build_lora_config, build_sft_config  # noqa: E402


def test_lora_targets_all_linear_layers() -> None:
    lc = build_lora_config(SmallRunConfig())
    assert set(lc.target_modules) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    assert lc.r == 16
    assert lc.lora_alpha == 32
    assert lc.task_type == "CAUSAL_LM"


def test_sft_config_is_completion_only_with_mlflow_callback() -> None:
    sc = build_sft_config(SmallRunConfig())
    assert sc.assistant_only_loss is True
    assert sc.report_to == ["mlflow"]
    assert sc.gradient_checkpointing is True
    assert sc.optim == "paged_adamw_8bit"
    assert sc.lr_scheduler_type == "cosine"


def test_oom_retry_flows_into_sft_config() -> None:
    sc = build_sft_config(SmallRunConfig().for_oom_retry(2))
    assert sc.max_length == 2048
    assert sc.per_device_train_batch_size == 1
    assert sc.gradient_accumulation_steps == 16


def test_precision_flags_follow_bf16() -> None:
    # exactly one mixed-precision mode, coherent with the model-load dtype
    on = build_sft_config(SmallRunConfig())  # bf16=True default
    assert on.bf16 is True and on.fp16 is False
    off = build_sft_config(SmallRunConfig(bf16=False))
    assert off.bf16 is False and off.fp16 is True
