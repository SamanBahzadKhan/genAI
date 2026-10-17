"""Small helpers shared by the training and evaluation scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch


def load_yaml(path):
    import yaml

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()


def prepare_causal_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def enable_training_mode(model, gradient_checkpointing):
    model.config.use_cache = not gradient_checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model
