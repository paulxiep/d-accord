"""Tier-10B small-run hyperparameter config — one explicit object, no YAML.

Every knob the M3 small-run touches lives here, so the deliberate engineering
choices are reviewable in one place rather than hidden in a config-wrapper
([development_plan.md §1]). Defaults are the frozen M3 config from
[docs/m3_gate.md]; `for_oom_retry` encodes the OOM ladder.

Deliberately torch-free (only stdlib + `daccord.validation`) so config + data
logic is unit-testable without the GPU stack.
"""

from __future__ import annotations

from pathlib import Path

from daccord.validation import ValidatedModel

# Repo root, resolved from this file (training/config.py → parents[1]). Lets
# defaults point at repo-root paths regardless of the process CWD (the docker
# `training` service runs from /workspace/training).
REPO_ROOT = Path(__file__).resolve().parents[1]

# All linear projections on Qwen3-8B — the QLoRA-paper default (decision #6).
# Strongest adaptation for a specialist; also the representative worst-case
# OOM test for the tier-12A full train.
ALL_LINEAR_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class SmallRunConfig(ValidatedModel):
    """Frozen M3 small-run config (docs/m3_gate.md tier-10B table)."""

    run_name: str = "qlora-small-run-200"
    base_model: str = "Qwen/Qwen3-8B"
    seed: int = 42

    # 4-bit NF4 — identical to LocalAdapterClient's BitsAndBytesConfig so the
    # adapter reloads under the same quantization it trained under.
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ALL_LINEAR_MODULES

    max_seq_len: int = 4096
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # effective batch 16
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    optim: str = "paged_adamw_8bit"  # bnb paged optimizer guards optim-state OOM
    num_train_epochs: int = 1
    gradient_checkpointing: bool = True
    bf16: bool = True

    train_subset: int = 200  # seeded subsample of train split
    eval_subset: int = 50  # head of val split, for the loss curve

    train_path: Path = REPO_ROOT / "data" / "training" / "train.jsonl"
    eval_path: Path = REPO_ROOT / "data" / "training" / "val.jsonl"
    output_dir: Path = REPO_ROOT / "training" / "runs"

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name

    @property
    def adapter_dir(self) -> Path:
        return self.run_dir / "adapter"

    def for_oom_retry(self, step: int) -> SmallRunConfig:
        """Next rung of the OOM ladder (docs/m3_gate.md, tier 11).

        step 1: max_seq_len 4096 → 2048 (gradient_checkpointing already on)
        step 2: + micro-batch 1 / grad-accum 16
        step 3: max_seq_len 1024 — last stop before swapping to Unsloth
        """
        if step <= 1:
            return self.model_copy(update={"max_seq_len": 2048})
        seq_len = 2048 if step == 2 else 1024
        return self.model_copy(
            update={
                "max_seq_len": seq_len,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 16,
            }
        )
