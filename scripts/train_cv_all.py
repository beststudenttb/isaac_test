"""Train or validate the four CV model variants one by one.

Examples:

    python scripts/train_cv_all.py --updates 1000 --batch-size 64 --device cuda
    python scripts/train_cv_all.py --val --size 100 --seed 0 --device cuda
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODELS = ("resnet", "mobile", "old", "old-mobile")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/validate all CV model variants in order.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--val", action="store_true")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("./data_isaac"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def run_model(cmd: list[str]) -> dict[str, float]:
    result = {}
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    for line in process.stdout.splitlines():
        line = line + "\n"
        print(line, end="")
        if line.startswith("mean_err "):
            parts = dict(part.split("=", 1) for part in line.strip().split()[1:] if "=" in part)
            result = {key: float(value) for key, value in parts.items()}
    return result


def main():
    args = parse_args()
    mode = "--val" if args.val else "--train"
    results = []
    for model in MODELS:
        cmd = [
            sys.executable,
            "train/cv.py",
            mode,
            "--model",
            model,
            "--updates",
            str(args.updates),
            "--batch-size",
            str(args.batch_size),
            "--size",
            str(args.size),
            "--seed",
            str(args.seed),
            "--data-dir",
            str(args.data_dir),
            "--device",
            args.device,
        ]
        print(f"\n===== {mode[2:]} {model} =====", flush=True)
        result = run_model(cmd)
        if args.val:
            results.append((model, result))

    if args.val:
        print("\n===== mean err summary =====")
        for model, result in results:
            print(f"{model}: x={result['x']:.2f} dist={result['dist']:.2f} n={int(result['n'])}")


if __name__ == "__main__":
    main()
