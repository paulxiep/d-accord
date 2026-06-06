"""Tier-10A — QLoRA small-run trainer for Qwen3-8B (the M3 gate).

Wires the shipped daccord pieces (the `daccord.tracking` reproducibility
contract, the eval prompt baked into the training data, and the
`LocalAdapterClient` reload check) around a `trl` `SFTTrainer`. This is the
M3 plumbing gate, not a performance run — see [docs/m3_gate.md].

The heavy ML imports (torch / transformers / peft / trl / datasets) are
function-local so `config.py` and `data.py` stay torch-free and unit-testable.
`build_lora_config` / `build_sft_config` are importable in the training env
without a GPU so their settings can be asserted in tests.

Run (after `scripts/build_training_data.py`):
    docker compose run --rm training uv run python train.py
OOM ladder (docs/m3_gate.md):
    ... train.py --max-seq-len 2048
    ... train.py --max-seq-len 2048 --per-device-batch-size 1 --grad-accum 16
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

import mlflow
from config import SmallRunConfig

from daccord.tracking import (
    compute_file_sha256,
    log_adapter_sha256,
    log_standard_params,
    set_all_seeds,
    setup_mlflow,
)
from data import load_chat_examples, subset

log = logging.getLogger("train")

# Hyperparameters mirrored into MLflow params alongside the 1B reproducibility
# contract. The HF callback (report_to=["mlflow"]) also logs TrainingArguments,
# but logging these explicitly keeps the run self-describing if the callback
# surface changes across trl versions.
HYPERPARAM_KEYS = (
    "base_model",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "max_seq_len",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "lr_scheduler_type",
    "warmup_ratio",
    "optim",
    "num_train_epochs",
    "gradient_checkpointing",
    "bf16",
    "train_subset",
)


def build_lora_config(cfg: SmallRunConfig) -> Any:
    """All-linear QLoRA `LoraConfig` (decision #6)."""
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_sft_config(cfg: SmallRunConfig) -> Any:
    """`SFTConfig` with completion-only loss + MLflow callback (decisions #2, #5).

    `max_length` is the current trl name for the truncation length (older trl
    called it `max_seq_length`); pin tracks the env's trl version.
    """
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(cfg.run_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        # Eval batch defaults to 8 in TrainingArguments — at these long sequences
        # that alone spills VRAM at the eval step. Tie it to the train micro-batch.
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        optim=cfg.optim,
        bf16=cfg.bf16,
        fp16=not cfg.bf16,  # exactly one mixed-precision mode; matches the load dtype
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=cfg.max_seq_len,
        packing=False,
        assistant_only_loss=True,  # completion-only loss via chat-template masking
        report_to=["mlflow"],  # HF callback logs params + metrics; NOT global autolog
        run_name=cfg.run_name,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="no",  # adapter saved explicitly after train()
        seed=cfg.seed,
    )


def _load_model_and_tokenizer(cfg: SmallRunConfig) -> tuple[Any, Any]:
    """Load Qwen3-8B in 4-bit NF4, kbit-prepare, and attach the LoRA adapter."""
    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Compute/storage dtype follows cfg.bf16 so the SFTConfig precision flag and
    # the model load can't diverge: bf16 on Blackwell (RTX 5080), fp16 fallback
    # on GPUs without bf16. (`dtype` is the transformers-5.x name for the
    # deprecated `torch_dtype`.)
    compute_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=quant,
        device_map="auto",
        dtype=compute_dtype,
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.gradient_checkpointing
    )
    model = get_peft_model(model, build_lora_config(cfg))
    model.print_trainable_parameters()
    return model, tokenizer


def _sanity_check(cfg: SmallRunConfig, examples: list[dict[str, object]]) -> None:
    """Reload the saved adapter via LocalAdapterClient and run a few prompts.

    This *is* the M3 "adapter saves+reloads cleanly" + "sanity-check inference
    output" check — it reuses the production reload path, no bespoke code.
    """
    from daccord.eval.schema import PromptMessages
    from daccord.serving.clients import LocalAdapterClient

    client = LocalAdapterClient(cfg.adapter_dir, base_model=cfg.base_model)
    out_path = cfg.run_dir / "sanity.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(examples):
            messages = ex["messages"]
            assert isinstance(messages, list)
            prompt = PromptMessages(system=messages[0]["content"], user=messages[1]["content"])
            resp = client.generate(prompt, run_id=cfg.run_name, batch_id=f"sanity-{i}")
            f.write(
                json.dumps(
                    {
                        "user": messages[1]["content"],
                        "gold": messages[2]["content"],
                        "predicted": resp.raw_text,
                        "parse_error": resp.parse_error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    log.info("sanity inference → %s", out_path)


def main(argv: list[str] | None = None) -> int:
    cfg = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not cfg.train_path.exists():
        raise FileNotFoundError(
            f"{cfg.train_path} missing — run scripts/build_training_data.py first"
        )

    setup_mlflow()
    set_all_seeds(cfg.seed)

    from datasets import Dataset

    train_rows = subset(load_chat_examples(cfg.train_path), cfg.train_subset, cfg.seed)
    eval_rows = load_chat_examples(cfg.eval_path)[: cfg.eval_subset]
    log.info(
        "train=%d (subset target %d)  eval=%d", len(train_rows), cfg.train_subset, len(eval_rows)
    )
    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows)
    dataset_hash = compute_file_sha256(cfg.train_path)

    model, tokenizer = _load_model_and_tokenizer(cfg)

    from trl import SFTTrainer

    with mlflow.start_run(run_name=cfg.run_name):
        log_standard_params(
            cfg.run_name,
            seed=cfg.seed,
            dataset_hash=dataset_hash,
            extra={k: str(getattr(cfg, k)) for k in HYPERPARAM_KEYS},
        )
        trainer = SFTTrainer(
            model=model,
            args=build_sft_config(cfg),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
        )
        trainer.train()
        trainer.save_model(str(cfg.adapter_dir))
        log_adapter_sha256(cfg.adapter_dir / "adapter_model.safetensors")
        log.info("adapter saved → %s", cfg.adapter_dir)

    _sanity_check(cfg, eval_rows[:3])
    return 0


def _parse_args(argv: list[str] | None) -> SmallRunConfig:
    parser = argparse.ArgumentParser(description="QLoRA small-run trainer (M3 gate).")
    parser.add_argument("--run-name")
    parser.add_argument("--max-seq-len", type=int, help="OOM ladder: 4096→2048→1024")
    parser.add_argument("--per-device-batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--train-subset", type=int)
    a = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if a.run_name:
        overrides["run_name"] = a.run_name
    if a.max_seq_len:
        overrides["max_seq_len"] = a.max_seq_len
    if a.per_device_batch_size:
        overrides["per_device_train_batch_size"] = a.per_device_batch_size
    if a.grad_accum:
        overrides["gradient_accumulation_steps"] = a.grad_accum
    if a.train_subset is not None:
        overrides["train_subset"] = a.train_subset
    return SmallRunConfig(**overrides)


if __name__ == "__main__":
    sys.exit(main())
