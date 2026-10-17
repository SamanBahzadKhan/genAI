"""Supervised fine-tuning with response-only loss.

The guide configuration is Qwen2.5-0.5B, 30k/1k examples, context 512,
per-device batch 2, gradient accumulation 16, learning rate 2e-5, 3% warmup,
cosine decay, weight decay 0.1, gradient clipping 1.0, and one epoch.
Trainable weights stay in fp32. The forward runs under fp16 autocast on CUDA.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from alignment.batching import build_lm_batch
from alignment.data import read_jsonl
from alignment.formatting import format_prompt
from alignment.losses import compute_sft_loss, response_logprobs_from_batch
from alignment.runtime import append_jsonl, enable_training_mode, load_yaml, prepare_causal_tokenizer, set_seed


class InstructionDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            "prompt": format_prompt(row["instruction"]),
            "response": row["response"],
        }


def collate(rows):
    return {
        "prompt": [row["prompt"] for row in rows],
        "response": [row["response"] for row in rows],
    }


def move_batch_loss(model, tokenizer, batch, max_length, max_prompt_length):
    return compute_sft_loss(
        model,
        tokenizer,
        batch["prompt"],
        batch["response"],
        max_length,
        max_prompt_length=max_prompt_length,
    )


@torch.no_grad()
def validation_loss(model, loader, tokenizer, max_length, max_prompt_length):
    """Token-mean response NLL over the full validation loader."""

    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for batch in loader:
        lm_batch = build_lm_batch(
            tokenizer,
            batch["prompt"],
            batch["response"],
            max_length,
            max_prompt_length=max_prompt_length,
        )
        summed, mask = response_logprobs_from_batch(model, lm_batch)
        total_nll += float((-summed).sum().cpu())
        total_tokens += int(mask.sum().cpu())
    model.train()
    if total_tokens == 0:
        return float("nan")
    return total_nll / total_tokens


def save_checkpoint(model, tokenizer, path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    previous_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    model.save_pretrained(path)
    model.config.use_cache = previous_cache
    tokenizer.save_pretrained(path)


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

    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print("CUDA is not available. This run is a functional CPU fallback, not the Kaggle recipe.")
    device = torch.device("cuda:0" if use_cuda else "cpu")

    tokenizer = prepare_causal_tokenizer(AutoTokenizer.from_pretrained(config["model_name"]))
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.float32)
    model.to(device)
    enable_training_mode(model, bool(config.get("gradient_checkpointing", False)))
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    train_rows = read_jsonl(config["train_path"])
    val_rows = read_jsonl(config["val_path"])
    train_loader = DataLoader(
        InstructionDataset(train_rows),
        batch_size=int(config["per_device_batch_size"]),
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        InstructionDataset(val_rows),
        batch_size=int(config["per_device_batch_size"]),
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    microbatches = math.ceil(len(train_rows) / int(config["per_device_batch_size"]))
    accum = int(config["gradient_accumulation_steps"])
    updates_per_epoch = math.ceil(microbatches / accum)
    total_updates = updates_per_epoch * int(config["num_epochs"])
    warmup_steps = int(float(config["warmup_ratio"]) * total_updates)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_updates, 1),
    )
    use_amp = bool(config.get("fp16", True)) and use_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    max_length = int(config["max_length"])
    max_prompt_length = config.get("max_prompt_length")
    if max_prompt_length is not None:
        max_prompt_length = int(max_prompt_length)
    grad_norm = float(config["max_grad_norm"])
    logging_steps = int(config.get("logging_steps", 10))
    eval_steps = int(config.get("eval_steps", 100))

    print(
        f"SFT examples train={len(train_rows)} val={len(val_rows)} "
        f"updates={total_updates} warmup={warmup_steps} amp={use_amp}"
    )

    best_val = float("inf")
    initial_val = validation_loss(model, val_loader, tokenizer, max_length, max_prompt_length)
    append_jsonl(metrics_path, {"step": 0, "epoch": 0, "split": "val", "loss": initial_val})
    print(f"step 0 val_loss {initial_val:.4f}")
    optimizer.zero_grad(set_to_none=True)
    global_update = 0
    window_loss = 0.0
    window_count = 0

    for epoch in range(int(config["num_epochs"])):
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = move_batch_loss(model, tokenizer, batch, max_length, max_prompt_length)
            scaled = loss / accum
            scaler.scale(scaled).backward()
            window_loss += float(loss.detach().cpu())
            window_count += 1

            reached_accum = step % accum == 0
            end_of_epoch = step == len(train_loader)
            if not reached_accum and not end_of_epoch:
                continue

            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_update += 1
            mean_loss = window_loss / max(window_count, 1)
            window_loss = 0.0
            window_count = 0

            if global_update % logging_steps == 0 or global_update == total_updates:
                record = {
                    "step": global_update,
                    "epoch": epoch,
                    "split": "train",
                    "loss": mean_loss,
                    "lr": scheduler.get_last_lr()[0],
                }
                append_jsonl(metrics_path, record)
                print(f"step {global_update} train_loss {mean_loss:.4f} lr {record['lr']:.3e}")

            if global_update % eval_steps == 0 or global_update == total_updates:
                val_loss = validation_loss(
                    model, val_loader, tokenizer, max_length, max_prompt_length
                )
                append_jsonl(
                    metrics_path,
                    {"step": global_update, "epoch": epoch, "split": "val", "loss": val_loss},
                )
                print(f"step {global_update} val_loss {val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(model, tokenizer, output_dir / "best")
                    append_jsonl(
                        output_dir / "best_metric.jsonl",
                        {"step": global_update, "val_loss": val_loss},
                    )
                    print(f"saved best checkpoint val_loss {val_loss:.4f}")

    save_checkpoint(model, tokenizer, output_dir / "final")
    print(f"saved final checkpoint to {output_dir / 'final'}")


if __name__ == "__main__":
    main()
