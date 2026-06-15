"""Train the four CV model variants one by one."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODELS = ("resnet", "mobile", "old", "old-mobile")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all CV model variants in order.")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--size", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("./data_isaac"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    for model in MODELS:
        cmd = [
            sys.executable,
            "train/train_cv.py",
            "--train",
            "--model",
            model,
            "--updates",
            str(args.updates),
            "--batch-size",
            str(args.batch_size),
            "--size",
            str(args.size),
            "--data-dir",
            str(args.data_dir),
            "--device",
            args.device,
        ]
        print(f"\n===== train {model} =====", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
