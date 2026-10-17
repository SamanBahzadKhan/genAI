"""Plot training and validation curves from a run's metrics.jsonl file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_metrics(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def series(rows, split, key):
    points = [row for row in rows if row.get("split") == split and key in row]
    return [row["step"] for row in points], [row[key] for row in points]


def rolling(values, window):
    if window <= 1:
        return values
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def plot_line(steps, values, title, ylabel, output_path, window):
    if not steps:
        print(f"skip {output_path.name}: no points")
        return
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(steps, values, label="raw", alpha=0.35)
    axis.plot(steps, rolling(values, window), label=f"rolling mean ({window})")
    axis.set_xlabel("optimizer step")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"wrote {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    rows = read_metrics(args.metrics_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, key, ylabel in (
        ("train", "loss", "loss"),
        ("val", "loss", "loss"),
        ("val", "reward_accuracy", "reward accuracy"),
        ("val", "reward_margin", "reward margin"),
    ):
        steps, values = series(rows, split, key)
        plot_line(
            steps,
            values,
            f"{args.run_name} {split} {key.replace('_', ' ')}",
            ylabel,
            output_dir / f"{split}_{key}.png",
            args.window,
        )


if __name__ == "__main__":
    main()
