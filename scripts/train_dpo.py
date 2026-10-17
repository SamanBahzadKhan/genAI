"""From-scratch DPO from a frozen SFT reference.

Policy weights stay in fp32 on cuda:0. The reference is a frozen fp16 copy on
cuda:1 when a second GPU is visible. Both runs use the same loss: beta 0.1,
learning rate 1e-6 with no decay schedule, per-device batch 1, and gradient
accumulation 32. The best checkpoint is the one with the highest validation
reward accuracy.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.batching import build_dpo_batch
from alignment.data import read_jsonl
from alignment.formatting import format_prompt
from alignment.losses import (
    compute_dpo_loss_from_logps,
    preference_rewards,
    response_logprobs_from_batch,
)
from alignment.runtime import append_jsonl, enable_training_mode, load_yaml, prepare_causal_tokenizer, set_seed


class PreferenceDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            "prompt": format_prompt(row["instruction"]),
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        }


def collate(rows):
    return {
        "prompt": [row["prompt"] for row in rows],
        "chosen": [row["chosen"] for row in rows],
        "rejected": [row["rejected"] for row in rows],
    }


def pair_logprobs(model, tokenizer, batch, max_length, max_prompt_length):
    encoded = build_dpo_batch(
        tokenizer,
        batch["prompt"],
        batch["chosen"],
        batch["rejected"],
        max_length,
        max_prompt_length=max_prompt_length,
    )
    chosen, _ = response_logprobs_from_batch(model, encoded["chosen"])
    rejected, _ = response_logprobs_from_batch(model, encoded["rejected"])
    return chosen, rejected


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


@torch.no_grad()
def evaluate(policy, reference, loader, tokenizer, max_length, max_prompt_length, beta):
    was_training = policy.training
    policy.eval()
    losses = []
    accuracies = []
    margins = []
    counts = []
    for batch in loader:
        policy_chosen, policy_rejected = pair_logprobs(
            policy, tokenizer, batch, max_length, max_prompt_length
        )
        reference_chosen, reference_rejected = pair_logprobs(
            reference, tokenizer, batch, max_length, max_prompt_length
        )
        loss = compute_dpo_loss_from_logps(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            beta,
        )
        chosen_reward, rejected_reward = preference_rewards(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            beta,
        )
        count = policy_chosen.shape[0]
        losses.append(float(loss.cpu()) * count)
        accuracies.append(float((chosen_reward > rejected_reward).float().sum().cpu()))
        margins.append(float((chosen_reward - rejected_reward).sum().cpu()))
        counts.append(count)
    policy.train(was_training)
    total = max(sum(counts), 1)
    return {
        "loss": sum(losses) / total,
        "reward_accuracy": sum(accuracies) / total,
        "reward_margin": sum(margins) / total,
    }


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
    policy_device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda and torch.cuda.device_count() > 1:
        reference_device = torch.device("cuda:1")
    else:
        reference_device = policy_device
        if use_cuda:
            print("Only one GPU is visible. The reference model shares cuda:0.")
    if not use_cuda:
        print("CUDA is not available. This run is a functional CPU fallback, not the Kaggle recipe.")

    tokenizer = prepare_causal_tokenizer(AutoTokenizer.from_pretrained(config["policy_path"]))
    policy = AutoModelForCausalLM.from_pretrained(config["policy_path"], torch_dtype=torch.float32)
    reference = AutoModelForCausalLM.from_pretrained(
        config["policy_path"],
        torch_dtype=torch.float16 if use_cuda else torch.float32,
    )
    policy.to(policy_device)
    reference.to(reference_device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    enable_training_mode(policy, bool(config.get("gradient_checkpointing", False)))
    reference.config.use_cache = True
    if tokenizer.pad_token_id is not None:
        policy.config.pad_token_id = tokenizer.pad_token_id
        reference.config.pad_token_id = tokenizer.pad_token_id

    train_rows = read_jsonl(config["train_path"])
    val_rows = read_jsonl(config["val_path"])
    batch_size = int(config["per_device_batch_size"])
    train_loader = DataLoader(
        PreferenceDataset(train_rows),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        PreferenceDataset(val_rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    if optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            policy.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    else:
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )

    accum = int(config["gradient_accumulation_steps"])
    microbatches = math.ceil(len(train_rows) / batch_size)
    updates_per_epoch = math.ceil(microbatches / accum)
    total_updates = updates_per_epoch * int(config["num_epochs"])
    use_amp = bool(config.get("fp16", True)) and use_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    beta = float(config["beta"])
    max_length = int(config["max_length"])
    max_prompt_length = int(config["max_prompt_length"])
    grad_norm = float(config["max_grad_norm"])
    logging_steps = int(config.get("logging_steps", 10))
    eval_steps = int(config.get("eval_steps", 100))

    print(
        f"DPO examples train={len(train_rows)} val={len(val_rows)} "
        f"updates={total_updates} beta={beta} policy={policy_device} reference={reference_device}"
    )

    best_accuracy = -1.0
    initial_metrics = evaluate(
        policy, reference, val_loader, tokenizer, max_length, max_prompt_length, beta
    )
    append_jsonl(metrics_path, {"step": 0, "epoch": 0, "split": "val", **initial_metrics})
    print(
        "step 0 val_loss {loss:.4f} reward_accuracy {reward_accuracy:.4f} "
        "reward_margin {reward_margin:.4f}".format(**initial_metrics)
    )
    optimizer.zero_grad(set_to_none=True)
    global_update = 0
    window_loss = 0.0
    window_count = 0

    for epoch in range(int(config["num_epochs"])):
        policy.train()
        for step, batch in enumerate(train_loader, start=1):
            with torch.autocast(device_type=policy_device.type, dtype=torch.float16, enabled=use_amp):
                policy_chosen, policy_rejected = pair_logprobs(
                    policy, tokenizer, batch, max_length, max_prompt_length
                )
            with torch.no_grad():
                reference_chosen, reference_rejected = pair_logprobs(
                    reference, tokenizer, batch, max_length, max_prompt_length
                )
            loss = compute_dpo_loss_from_logps(
                policy_chosen,
                policy_rejected,
                reference_chosen,
                reference_rejected,
                beta,
            )
            scaler.scale(loss / accum).backward()
            window_loss += float(loss.detach().cpu())
            window_count += 1

            reached_accum = step % accum == 0
            end_of_epoch = step == len(train_loader)
            if not reached_accum and not end_of_epoch:
                continue

            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
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
                    "lr": float(config["learning_rate"]),
                }
                append_jsonl(metrics_path, record)
                print(f"step {global_update} dpo_loss {mean_loss:.4f}")

            if global_update % eval_steps == 0 or global_update == total_updates:
                metrics = evaluate(
                    policy, reference, val_loader, tokenizer, max_length, max_prompt_length, beta
                )
                append_jsonl(
                    metrics_path,
                    {"step": global_update, "epoch": epoch, "split": "val", **metrics},
                )
                print(
                    "step {step} val_loss {loss:.4f} reward_accuracy {reward_accuracy:.4f} "
                    "reward_margin {reward_margin:.4f}".format(step=global_update, **metrics)
                )
                if metrics["reward_accuracy"] > best_accuracy:
                    best_accuracy = metrics["reward_accuracy"]
                    save_checkpoint(policy, tokenizer, output_dir / "best")
                    append_jsonl(
                        output_dir / "best_metric.jsonl",
                        {"step": global_update, **metrics},
                    )
                    print(f"saved best checkpoint reward_accuracy {best_accuracy:.4f}")

    save_checkpoint(policy, tokenizer, output_dir / "final")
    print(f"saved final checkpoint to {output_dir / 'final'}")


if __name__ == "__main__":
    main()
