from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse

from src.visualize import make_syndrome_animation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a GIF animation of syndrome defects and correction chains.")
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--physical-error-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--error-type", choices=["z", "x"], default="z")
    parser.add_argument("--decoder", choices=["mwpm", "union_find"], default="mwpm")
    parser.add_argument("--output", type=str, default="results/syndrome_correction_animation.gif")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_syndrome_animation(
        distance=args.distance,
        physical_error_rate=args.physical_error_rate,
        seed=args.seed,
        output_path=args.output,
        error_type=args.error_type,
        decoder_name=args.decoder,
    )
    print(f"Saved animation: {args.output}")


if __name__ == "__main__":
    main()
