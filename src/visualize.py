from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.decoder import make_decoder
from src.lattice import Edge, Node, PlanarSurfaceCodeLattice, xor_edges
from src.noise import sample_iid_edge_noise


def _edge_xy(edge: Edge, distance: int) -> tuple[list[float], list[float]]:
    if edge.kind == "h":
        return [edge.col, edge.col + 1], [edge.row, edge.row]
    if edge.kind == "v":
        return [edge.col, edge.col], [edge.row, edge.row + 1]
    if edge.kind == "bt":
        return [edge.col, edge.col], [-0.7, 0]
    if edge.kind == "bb":
        return [edge.col, edge.col], [distance - 1, distance - 0.3]
    if edge.kind == "bl":
        return [-0.7, 0], [edge.row, edge.row]
    if edge.kind == "br":
        return [distance - 1, distance - 0.3], [edge.row, edge.row]
    raise ValueError(f"unknown edge kind: {edge.kind}")


def draw_decoding_state(
    lattice: PlanarSurfaceCodeLattice,
    syndrome: set[Node],
    correction_edges: set[Edge] | None = None,
    residual_edges: set[Edge] | None = None,
    title: str = "",
    output_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    for node in lattice.nodes():
        ax.scatter(node.col, node.row, s=25, c="lightgray")

    for edge in lattice.data_edges():
        xs, ys = _edge_xy(edge, lattice.distance)
        ax.plot(xs, ys, linewidth=0.8, c="lightgray")

    for node in syndrome:
        ax.scatter(node.col, node.row, s=120, c="red", marker="o", label="syndrome defect")

    if correction_edges:
        for edge in correction_edges:
            xs, ys = _edge_xy(edge, lattice.distance)
            ax.plot(xs, ys, linewidth=3.0, c="blue", alpha=0.85)

    if residual_edges:
        for edge in residual_edges:
            xs, ys = _edge_xy(edge, lattice.distance)
            ax.plot(xs, ys, linewidth=3.0, c="black", alpha=0.85)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-1, lattice.distance)
    ax.set_ylim(lattice.distance, -1)
    ax.set_xticks(range(lattice.distance))
    ax.set_yticks(range(lattice.distance))
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=180)
        plt.close(fig)
    else:
        plt.show()


def make_syndrome_animation(
    distance: int,
    physical_error_rate: float,
    seed: int,
    output_path: str,
    error_type: str = "z",
    decoder_name: str = "mwpm",
) -> None:
    rng = random.Random(seed)
    lattice = PlanarSurfaceCodeLattice(distance=distance, error_type=error_type)  # type: ignore[arg-type]
    sample = sample_iid_edge_noise(lattice, physical_error_rate, rng)
    decoder = make_decoder(decoder_name, lattice)
    decoded = decoder.decode(sample.syndrome)
    residual = xor_edges(sample.error_edges, decoded.correction_edges)

    frames = [
        (set(), set(), set(), "1. Empty decoding graph"),
        (sample.syndrome, set(), set(), "2. Syndrome defects"),
        (sample.syndrome, decoded.correction_edges, set(), "3. Correction chain"),
        (sample.syndrome, decoded.correction_edges, residual, "4. Residual chain after correction"),
    ]

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame_index: int):
        ax.clear()
        syndrome, correction_edges, residual_edges, title = frames[frame_index]
        for node in lattice.nodes():
            ax.scatter(node.col, node.row, s=25, c="lightgray")
        for edge in lattice.data_edges():
            xs, ys = _edge_xy(edge, lattice.distance)
            ax.plot(xs, ys, linewidth=0.8, c="lightgray")
        for node in syndrome:
            ax.scatter(node.col, node.row, s=120, c="red", marker="o")
        for edge in correction_edges:
            xs, ys = _edge_xy(edge, lattice.distance)
            ax.plot(xs, ys, linewidth=3.0, c="blue", alpha=0.85)
        for edge in residual_edges:
            xs, ys = _edge_xy(edge, lattice.distance)
            ax.plot(xs, ys, linewidth=3.0, c="black", alpha=0.85)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-1, lattice.distance)
        ax.set_ylim(lattice.distance, -1)
        ax.set_xticks(range(lattice.distance))
        ax.set_yticks(range(lattice.distance))
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("column")
        ax.set_ylabel("row")

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1100, repeat=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=1))
    plt.close(fig)
