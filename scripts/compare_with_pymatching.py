from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from src.pymatching_compare import compare_from_scratch_with_pymatching


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the from-scratch decoder with PyMatching.")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--physical-error-rate", type=float, default=0.02)
    parser.add_argument("--error-type", choices=["z", "x"], default="z")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    try:
        df = compare_from_scratch_with_pymatching(
            distance=args.distance,
            physical_error_rate=args.physical_error_rate,
            shots=args.shots,
            seed=args.seed,
            error_type=args.error_type,
        )
    except RuntimeError as exc:
        print(exc)
        return

    output_path = results_dir / "pymatching_comparison.csv"
    df.to_csv(output_path, index=False)

    print("Comparison finished.")
    print(f"Saved CSV: {output_path}")
    print(df.describe(include="all"))


if __name__ == "__main__":
    main()
