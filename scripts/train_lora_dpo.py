"""TRL DPOTrainer + LoRA persona run on the provided Ramsay-style subset.

This is the only track that uses TRL and PEFT. The reference policy is the
frozen base model with the LoRA adapters disabled, which is TRL's default
when ``ref_model`` is omitted and a ``peft_config`` is supplied.
"""

from __future__ import annotations

import os

# TRL's DPOTrainer rejects DataParallel. This run fits on one GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

from alignment.runtime import append_jsonl, load_yaml, prepare_causal_tokenizer, set_seed


def truncate_to_tokens(tokenizer, text, max_tokens):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)


def conversational_rows(raw_rows, tokenizer, max_prompt_length):
    rows = []
    for row in raw_rows:
        prompt = row["prompt"]
        if max_prompt_length:
            prompt = truncate_to_tokens(tokenizer, prompt, int(max_prompt_length))
        rows.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "chosen": [{"role": "assistant", "content": row["chosen"]}],
                "rejected": [{"role": "assistant", "content": row["rejected"]}],
            }
        )
    return rows


class MetricsCallback(TrainerCallback):
    def __init__(self, path):
        self.path = path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {"step": int(state.global_step)}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                record[key] = float(value)
        append_jsonl(self.path, record)


def build_dpo_config(config, output_dir):
    from trl import DPOConfig

    common = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(config["per_device_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=float(config["num_epochs"]),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(config.get("warmup_ratio", 0.03)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        fp16=bool(config.get("fp16", True)) and torch.cuda.is_available(),
        bf16=False,
        logging_steps=int(config.get("logging_steps", 5)),
        save_strategy="no",
        report_to=[],
        seed=int(config["seed"]),
        remove_unused_columns=False,
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        beta=float(config["beta"]),
        max_length=int(config["max_length"]),
    )
    # TRL <= 1.15 still takes max_prompt_length on DPOConfig. TRL 1.16 dropped it
    # and truncates the whole sequence with max_length.
    if "max_prompt_length" in config:
        try:
            return DPOConfig(max_prompt_length=int(config["max_prompt_length"]), **common)
        except TypeError:
            pass
    return DPOConfig(**common)


def build_trainer(model, tokenizer, train_dataset, config, output_dir):
    from trl import DPOTrainer

    lora_config = LoraConfig(
        r=int(config["lora_rank"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config["target_modules"]),
    )
    trainer_kwargs = dict(
        model=model,
        ref_model=None,
        args=build_dpo_config(config, output_dir),
        train_dataset=train_dataset,
        peft_config=lora_config,
        callbacks=[MetricsCallback(output_dir / "metrics.jsonl")],
    )
    try:
        return DPOTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError as exc:
        if "processing_class" not in str(exc):
            raise
        return DPOTrainer(tokenizer=tokenizer, **trainer_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    set_seed(int(config["seed"]))

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    with open(config["dataset_path"], encoding="utf-8") as handle:
        raw_rows = json.load(handle)

    tokenizer = prepare_causal_tokenizer(AutoTokenizer.from_pretrained(config["model_name"]))
    tokenizer.padding_side = "left"
    train_dataset = Dataset.from_list(
        conversational_rows(raw_rows, tokenizer, config.get("max_prompt_length"))
    )
    dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], dtype=dtype)
    if torch.cuda.is_available():
        model.to("cuda:0")

    trainer = build_trainer(model, tokenizer, train_dataset, config, output_dir)
    trainable = sum(parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in trainer.model.parameters())
    summary = {
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "trainable_fraction": float(trainable) / float(total),
        "examples": len(raw_rows),
        "lora_rank": int(config["lora_rank"]),
        "lora_alpha": int(config["lora_alpha"]),
        "lora_dropout": float(config["lora_dropout"]),
        "target_modules": list(config["target_modules"]),
    }
    with open(output_dir / "trainable_params.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary))

    trainer.train()
    trainer.model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    print(f"saved LoRA adapter to {output_dir / 'final'}")


if __name__ == "__main__":
    main()
