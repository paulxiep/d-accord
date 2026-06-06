"""SmallRunConfig tests — the frozen M3 hyperparameters + the OOM ladder.

Torch-free: runs anywhere the daccord package is importable.
"""

from __future__ import annotations

from config import ALL_LINEAR_MODULES, SmallRunConfig


class TestDefaults:
    def test_defaults_match_m3_gate_table(self) -> None:
        cfg = SmallRunConfig()
        assert cfg.base_model == "Qwen/Qwen3-8B"
        assert cfg.seed == 42
        assert cfg.bnb_4bit_quant_type == "nf4"
        assert (cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout) == (16, 32, 0.05)
        assert cfg.max_seq_len == 4096
        assert (cfg.per_device_train_batch_size, cfg.gradient_accumulation_steps) == (2, 8)
        assert cfg.optim == "paged_adamw_8bit"
        assert cfg.num_train_epochs == 1
        assert cfg.gradient_checkpointing is True
        assert cfg.train_subset == 200
        assert cfg.eval_subset == 50

    def test_targets_all_seven_linear_modules(self) -> None:
        cfg = SmallRunConfig()
        assert cfg.lora_target_modules == ALL_LINEAR_MODULES
        assert len(cfg.lora_target_modules) == 7

    def test_run_and_adapter_dirs_derive_from_run_name(self) -> None:
        cfg = SmallRunConfig(run_name="rn-1")
        assert cfg.run_dir.name == "rn-1"
        assert cfg.adapter_dir == cfg.run_dir / "adapter"


class TestOomLadder:
    def test_step1_drops_seq_len_only(self) -> None:
        cfg = SmallRunConfig()
        s1 = cfg.for_oom_retry(1)
        assert s1.max_seq_len == 2048
        assert s1.per_device_train_batch_size == cfg.per_device_train_batch_size

    def test_step2_micro_batches(self) -> None:
        s2 = SmallRunConfig().for_oom_retry(2)
        assert (s2.max_seq_len, s2.per_device_train_batch_size, s2.gradient_accumulation_steps) == (
            2048,
            1,
            16,
        )

    def test_step3_drops_to_1024(self) -> None:
        s3 = SmallRunConfig().for_oom_retry(3)
        assert s3.max_seq_len == 1024
        assert s3.per_device_train_batch_size == 1

    def test_retry_does_not_mutate_original(self) -> None:
        cfg = SmallRunConfig()
        cfg.for_oom_retry(3)
        assert cfg.max_seq_len == 4096
