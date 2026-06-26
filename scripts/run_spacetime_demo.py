from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt

from src.spacetime import SpaceTimeDecodingGraph, SpaceTimeMWPMDecoder, defects_to_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small 3D space-time decoding graph demo.")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--data-error-rate", type=float, default=0.03)
    parser.add_argument("--measurement-error-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    rng = random.Random(args.seed)
    graph = SpaceTimeDecodingGraph(distance=args.distance, rounds=args.rounds)
    sample = graph.sample_noise(args.data_error_rate, args.measurement_error_rate, rng)
    decoded = SpaceTimeMWPMDecoder(graph).decode(sample.defects)

    defects_df = defects_to_dataframe(sample.defects)
    defects_path = results_dir / "spacetime_defects.csv"
    defects_df.to_csv(defects_path, index=False)

    summary_path = results_dir / "spacetime_matching_summary.csv"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("distance,rounds,data_error_rate,measurement_error_rate,num_defects,total_matching_weight,num_matches\n")
        f.write(
            f"{args.distance},{args.rounds},{args.data_error_rate},{args.measurement_error_rate},"
            f"{len(sample.defects)},{decoded.total_weight},{len(decoded.matches)}\n"
        )

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    if not defects_df.empty:
        ax.scatter(defects_df["col"], defects_df["row"], defects_df["time"], s=60)
    ax.set_title("3D space-time syndrome defects")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_zlabel("time")
    ax.set_xlim(-0.5, args.distance - 0.5)
    ax.set_ylim(args.distance - 0.5, -0.5)
    ax.set_zlim(-0.5, args.rounds - 0.5)
    plt.tight_layout()
    plot_path = results_dir / "spacetime_defects.png"
    plt.savefig(plot_path, dpi=180)
    plt.close(fig)

    print("Space-time demo finished.")
    print(f"Saved defects CSV: {defects_path}")
    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Number of defects: {len(sample.defects)}")
    print(f"Total matching weight: {decoded.total_weight}")


if __name__ == "__main__":
    main()
