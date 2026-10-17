"""Summarize blinded pairwise judgments from the candidate model's point of view."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rate(count, total):
    if total == 0:
        return 0.0
    return count / total


def summarize(rows, failures, candidate):
    wins = sum(1 for row in rows if row.get("winner_model") == candidate)
    ties = sum(1 for row in rows if row.get("judgment") == "tie")
    unknown = sum(1 for row in rows if row.get("judgment") == "unknown")
    losses = sum(
        1
        for row in rows
        if row.get("judgment") in {"A", "B"} and row.get("winner_model") not in {candidate, "tie", "unknown"}
    )
    judged = wins + losses + ties + unknown
    decided = wins + losses
    return {
        "candidate": candidate,
        "sample_count": judged,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "unknown": unknown,
        "failures": len(failures),
        "win_rate_including_ties": rate(wins, judged),
        "win_rate_excluding_ties": rate(wins, decided),
        "judge_model": rows[0].get("judge_model") if rows else None,
    }


def by_category(rows, candidate):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("category") or "all"].append(row)
    return {name: summarize(group, [], candidate) for name, group in sorted(grouped.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged_path", required=True)
    parser.add_argument("--failure_path", default=None)
    parser.add_argument("--candidate", default="dpo")
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    judged_path = Path(args.judged_path)
    failure_path = Path(args.failure_path) if args.failure_path else judged_path.with_suffix(".failures.jsonl")
    rows = read_jsonl(judged_path)
    failures = read_jsonl(failure_path)
    summary = summarize(rows, failures, args.candidate)
    summary["by_category"] = by_category(rows, args.candidate)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
