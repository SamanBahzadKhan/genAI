"""Dataset conversion for the three training tracks.

SFT uses the first user/assistant exchange of each UltraChat dialogue.
From-scratch DPO keeps only single-turn preference pairs.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def first_ultrachat_turn(messages):
    """Convert a dialogue into one instruction and the first assistant reply."""

    instruction = None
    for message in messages:
        content = (message.get("content") or "").strip()
        role = message.get("role")
        if instruction is None and role == "user" and content:
            instruction = content
            continue
        if instruction is not None and role == "assistant" and content:
            return instruction, content
    return None


def single_turn_messages(messages):
    users = []
    assistants = []
    for message in messages:
        content = (message.get("content") or "").strip()
        if not content:
            continue
        if message.get("role") == "user":
            users.append(content)
        elif message.get("role") == "assistant":
            assistants.append(content)
    if len(users) != 1 or len(assistants) != 1:
        return None
    return users[0], assistants[0]


def parse_hh_dialogue(text):
    """Return (user, assistant) only when the transcript is a single turn."""

    turns = []
    for block in text.strip().split("\n\n"):
        block = block.strip()
        if block.startswith("Human:"):
            turns.append(("user", block[len("Human:") :].strip()))
        elif block.startswith("Assistant:"):
            turns.append(("assistant", block[len("Assistant:") :].strip()))
    users = [content for role, content in turns if role == "user" and content]
    assistants = [content for role, content in turns if role == "assistant" and content]
    if len(users) != 1 or len(assistants) != 1:
        return None
    return users[0], assistants[0]


def take_split(rows, train_count, val_count, seed):
    """Shuffle a filtered pool and cut disjoint train and validation slices."""

    pool = list(rows)
    random.Random(seed).shuffle(pool)
    need = train_count + val_count
    if len(pool) < need:
        raise RuntimeError(f"Need {need} examples after filtering, found {len(pool)}")
    return pool[:train_count], pool[train_count:need]


def prepare_sft_rows(dataset, train_count, val_count, seed):
    pool = []
    for index, row in enumerate(dataset):
        converted = first_ultrachat_turn(row["messages"])
        if converted is None:
            continue
        instruction, response = converted
        pool.append(
            {
                "id": row.get("prompt_id") or f"ultrachat-{index}",
                "instruction": instruction,
                "response": response,
                "source": "HuggingFaceH4/ultrachat_200k",
            }
        )
    return take_split(pool, train_count, val_count, seed)


def prepare_ultrafeedback_rows(dataset, train_count, val_count, seed):
    pool = []
    for index, row in enumerate(dataset):
        chosen = single_turn_messages(row["chosen"])
        rejected = single_turn_messages(row["rejected"])
        if chosen is None or rejected is None:
            continue
        if chosen[0] != rejected[0] or chosen[1] == rejected[1]:
            continue
        pool.append(
            {
                "id": row.get("prompt_id") or f"ultrafeedback-{index}",
                "instruction": chosen[0],
                "chosen": chosen[1],
                "rejected": rejected[1],
                "source": "HuggingFaceH4/ultrafeedback_binarized",
            }
        )
    return take_split(pool, train_count, val_count, seed)


def prepare_hh_rows(dataset, train_count, val_count, seed):
    pool = []
    for index, row in enumerate(dataset):
        chosen = parse_hh_dialogue(row["chosen"])
        rejected = parse_hh_dialogue(row["rejected"])
        if chosen is None or rejected is None:
            continue
        if chosen[0] != rejected[0] or chosen[1] == rejected[1]:
            continue
        pool.append(
            {
                "id": f"hh-rlhf-{index}",
                "instruction": chosen[0],
                "chosen": chosen[1],
                "rejected": rejected[1],
                "source": "Anthropic/hh-rlhf",
            }
        )
    return take_split(pool, train_count, val_count, seed)
