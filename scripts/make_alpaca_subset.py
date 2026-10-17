"""Build the fixed 300-prompt AlpacaEval subset.

The full instruction list is sorted, shuffled with ``random.Random(seed)``,
and then truncated to 300 rows. Sorting first keeps the subset stable if the
upstream dataset order changes.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

def load_alpaca_rows():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        filename="alpaca_eval.json",
        repo_type="dataset",
    )
    with open(path, encoding="utf-8") as handle:
        dataset = json.load(handle)
    rows = []
    for index, row in enumerate(dataset):
        instruction = (row.get("instruction") or "").strip()
        if not instruction:
            continue
        rows.append(
            {
                "source_index": index,
                "instruction": instruction,
                "origin": row.get("dataset") or "alpaca_eval",
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="eval_sets/alpaca_eval_300_seed42.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=300)
    args = parser.parse_args()

    rows = load_alpaca_rows()
    rows.sort(key=lambda row: row["instruction"])
    random.Random(args.seed).shuffle(rows)
    chosen = rows[: args.count]
    payload = []
    for index, row in enumerate(chosen):
        payload.append(
            {
                "id": f"alpaca-{args.seed}-{index:04d}",
                "instruction": row["instruction"],
                "dataset": "alpaca_eval",
                "category": row["origin"],
                "source_index": row["source_index"],
            }
        )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"wrote {len(payload)} prompts to {output_path}")


if __name__ == "__main__":
    main()
