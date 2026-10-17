"""Download and freeze the training subsets used by the assignment.

SFT: first user/assistant turn from UltraChat 200k, 30,000 train and 1,000 val.
HH-RLHF DPO: single-turn helpful/harmless pairs, 15,000 train and 1,000 val.
UltraFeedback DPO: single-turn preference pairs, 15,000 train and 1,000 val.
A shared seed shuffles each filtered pool before the split, so the slices are disjoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from alignment.data import (
    prepare_hh_rows,
    prepare_sft_rows,
    prepare_ultrafeedback_rows,
    write_json,
    write_jsonl,
)
from alignment.runtime import load_yaml


def maybe_config(args):
    if not args.config:
        return {}
    return load_yaml(args.config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft_train", type=int, default=30000)
    parser.add_argument("--sft_val", type=int, default=1000)
    parser.add_argument("--dpo_train", type=int, default=15000)
    parser.add_argument("--dpo_val", type=int, default=1000)
    parser.add_argument("--splits", default="sft,hh,ultrafeedback")
    args = parser.parse_args()
    config = maybe_config(args)
    out_dir = Path(config.get("out_dir", args.out_dir))
    seed = int(config.get("seed", args.seed))
    requested = set(args.splits.split(","))
    manifest = {"seed": seed, "out_dir": str(out_dir)}

    if "sft" in requested:
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        train_rows, val_rows = prepare_sft_rows(
            dataset,
            int(config.get("sft_train", args.sft_train)),
            int(config.get("sft_val", args.sft_val)),
            seed,
        )
        write_jsonl(out_dir / "sft_train.jsonl", train_rows)
        write_jsonl(out_dir / "sft_val.jsonl", val_rows)
        manifest["sft_train"] = len(train_rows)
        manifest["sft_val"] = len(val_rows)
        print(f"SFT train {len(train_rows)} val {len(val_rows)}")

    if "hh" in requested:
        dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        train_rows, val_rows = prepare_hh_rows(
            dataset,
            int(config.get("dpo_train", args.dpo_train)),
            int(config.get("dpo_val", args.dpo_val)),
            seed,
        )
        write_jsonl(out_dir / "hh_train.jsonl", train_rows)
        write_jsonl(out_dir / "hh_val.jsonl", val_rows)
        manifest["hh_train"] = len(train_rows)
        manifest["hh_val"] = len(val_rows)
        print(f"HH-RLHF train {len(train_rows)} val {len(val_rows)}")

    if "ultrafeedback" in requested:
        dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
        train_rows, val_rows = prepare_ultrafeedback_rows(
            dataset,
            int(config.get("dpo_train", args.dpo_train)),
            int(config.get("dpo_val", args.dpo_val)),
            seed,
        )
        write_jsonl(out_dir / "ultrafeedback_train.jsonl", train_rows)
        write_jsonl(out_dir / "ultrafeedback_val.jsonl", val_rows)
        manifest["ultrafeedback_train"] = len(train_rows)
        manifest["ultrafeedback_val"] = len(val_rows)
        print(f"UltraFeedback train {len(train_rows)} val {len(val_rows)}")

    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
