from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from src.evaluate import run_css_experiment, run_experiment
from src.plot_results import plot_logical_failure_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run from-scratch surface-code decoder experiments.")
    parser.add_argument("--distances", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--shots", type=int, default=300)
    parser.add_argument(
        "--error-rates",
        nargs="+",
        type=float,
        default=[0.005, 0.01, 0.02, 0.03, 0.05],
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--decoder", choices=["mwpm", "union_find"], default="mwpm")
    parser.add_argument("--error-type", choices=["z", "x"], default="z")
    parser.add_argument("--css", action="store_true", help="Run CSS-style separated X/Z decoding.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    if args.css:
        df = run_css_experiment(
            distances=args.distances,
            physical_error_rates=args.error_rates,
            shots=args.shots,
            seed=args.seed,
            decoder_name=args.decoder,
        )
        csv_path = results_dir / f"css_{args.decoder}_logical_failure_rate.csv"
        plot_path = results_dir / f"css_{args.decoder}_logical_failure_rate_plot.png"
    else:
        df = run_experiment(
            distances=args.distances,
            physical_error_rates=args.error_rates,
            shots=args.shots,
            seed=args.seed,
            error_type=args.error_type,
            decoder_name=args.decoder,
        )
        # Keep the original default filenames for the first GitHub result.
        if args.decoder == "mwpm" and args.error_type == "z":
            csv_path = results_dir / "logical_failure_rate.csv"
            plot_path = results_dir / "logical_failure_rate_plot.png"
        else:
            csv_path = results_dir / f"{args.error_type}_{args.decoder}_logical_failure_rate.csv"
            plot_path = results_dir / f"{args.error_type}_{args.decoder}_logical_failure_rate_plot.png"

    df.to_csv(csv_path, index=False)
    plot_logical_failure_rate(df, str(plot_path))

    print("Experiment finished.")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")
    print(df)


if __name__ == "__main__":
    main()
