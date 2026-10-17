"""Build blinded DPO-versus-SFT pairwise files.

Response order is randomized. Model identity is stored only in ``_model_map``
and is not included in the text sent to the judge.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_generations(path):
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    return {row["id"]: row for row in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_path", required=True)
    parser.add_argument("--dpo_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate_name", default="dpo")
    parser.add_argument("--baseline_name", default="sft")
    args = parser.parse_args()

    baseline = load_generations(args.sft_path)
    candidate = load_generations(args.dpo_path)
    shared_ids = [row_id for row_id in baseline if row_id in candidate]
    if not shared_ids:
        raise RuntimeError("The two generation files have no shared ids")

    rng = random.Random(args.seed)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for row_id in shared_ids:
            left = baseline[row_id]
            right = candidate[row_id]
            if rng.random() < 0.5:
                response_a, response_b = left["output"], right["output"]
                map_a, map_b = args.baseline_name, args.candidate_name
            else:
                response_a, response_b = right["output"], left["output"]
                map_a, map_b = args.candidate_name, args.baseline_name
            record = {
                "id": row_id,
                "instruction": left["instruction"],
                "response_a": response_a,
                "response_b": response_b,
                "dataset": left.get("dataset"),
                "category": left.get("category"),
                "_model_map": {"A": map_a, "B": map_b},
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(shared_ids)} pairs to {output_path}")


if __name__ == "__main__":
    main()
