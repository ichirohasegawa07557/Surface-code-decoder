from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_logical_failure_rate(df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot logical failure rate against physical error rate."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for distance in sorted(df["distance"].unique()):
        subset = df[df["distance"] == distance]
        plt.plot(
            subset["physical_error_rate"],
            subset["logical_failure_rate"],
            marker="o",
            label=f"distance={distance}",
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Physical error rate")
    plt.ylabel("Logical failure rate")
    plt.title("From-Scratch Surface-Code Decoder")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
