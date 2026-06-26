from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, NamedTuple

Boundary = Literal["top", "bottom", "left", "right"]
ErrorType = Literal["z", "x"]


class Node(NamedTuple):
    """A syndrome-check node on a 2D planar decoding grid."""

    row: int
    col: int


@dataclass(frozen=True)
class Edge:
    """An elementary error/correction edge in the decoding graph.

    kind:
        "h"  : horizontal edge between two neighboring checks
        "v"  : vertical edge between two neighboring checks
        "bt" : boundary edge between top boundary and top-row check
        "bb" : boundary edge between bottom-row check and bottom boundary
        "bl" : boundary edge between left boundary and left-column check
        "br" : boundary edge between right-column check and right boundary

    row, col:
        Integer coordinates. For boundary edges, row/col identify the
        boundary contact.
    """

    kind: str
    row: int
    col: int


class PlanarSurfaceCodeLattice:
    """Small planar surface-code decoding graph.

    This is an educational decoding graph for a phenomenological surface-code
    memory experiment. Syndrome checks are laid out on a distance x distance
    grid. Elementary errors toggle either two neighboring checks or one check
    and a boundary.

    error_type="z":
        Decodes Z-type data errors using top/bottom rough boundaries.

    error_type="x":
        Decodes X-type data errors on the dual orientation using left/right
        smooth boundaries.

    This gives a simple CSS-style separation: X and Z errors are sampled and
    decoded on two independent decoding graphs.
    """

    def __init__(self, distance: int, error_type: ErrorType = "z"):
        if distance < 3:
            raise ValueError("distance must be at least 3")
        if distance % 2 == 0:
            raise ValueError("use an odd code distance such as 3, 5, or 7")
        if error_type not in ("z", "x"):
            raise ValueError("error_type must be 'z' or 'x'")
        self.distance = distance
        self.error_type: ErrorType = error_type

    @property
    def boundaries(self) -> tuple[Boundary, Boundary]:
        if self.error_type == "z":
            return "top", "bottom"
        return "left", "right"

    def nodes(self) -> list[Node]:
        return [Node(r, c) for r in range(self.distance) for c in range(self.distance)]

    def data_edges(self) -> list[Edge]:
        """Return all elementary error edges in the decoding graph."""

        d = self.distance
        edges: list[Edge] = []

        # Horizontal edges between neighboring checks.
        for r in range(d):
            for c in range(d - 1):
                edges.append(Edge("h", r, c))

        # Vertical edges between neighboring checks.
        for r in range(d - 1):
            for c in range(d):
                edges.append(Edge("v", r, c))

        # Boundary edges depend on the CSS component being decoded.
        if self.error_type == "z":
            for c in range(d):
                edges.append(Edge("bt", -1, c))
                edges.append(Edge("bb", d, c))
        else:
            for r in range(d):
                edges.append(Edge("bl", r, -1))
                edges.append(Edge("br", r, d))

        return edges

    def edge_endpoints(self, edge: Edge) -> tuple[Node | None, Node | None]:
        """Return endpoints of an edge.

        Boundary endpoints are represented as None because they do not appear
        as syndrome defects.
        """

        d = self.distance

        if edge.kind == "h":
            return Node(edge.row, edge.col), Node(edge.row, edge.col + 1)
        if edge.kind == "v":
            return Node(edge.row, edge.col), Node(edge.row + 1, edge.col)
        if edge.kind == "bt":
            return None, Node(0, edge.col)
        if edge.kind == "bb":
            return Node(d - 1, edge.col), None
        if edge.kind == "bl":
            return None, Node(edge.row, 0)
        if edge.kind == "br":
            return Node(edge.row, d - 1), None

        raise ValueError(f"unknown edge kind: {edge.kind}")

    def toggle_syndrome_from_edges(self, edges: Iterable[Edge]) -> set[Node]:
        """Compute syndrome defects created by a set of error edges."""

        syndrome: set[Node] = set()
        for edge in edges:
            for endpoint in self.edge_endpoints(edge):
                if endpoint is None:
                    continue
                if endpoint in syndrome:
                    syndrome.remove(endpoint)
                else:
                    syndrome.add(endpoint)
        return syndrome

    def distance_between(self, a: Node, b: Node) -> int:
        """Manhattan distance between two syndrome defects."""

        return abs(a.row - b.row) + abs(a.col - b.col)

    def boundary_distance(self, node: Node, boundary: Boundary) -> int:
        """Distance from a syndrome defect to a decoding boundary."""

        if boundary == "top":
            return node.row + 1
        if boundary == "bottom":
            return self.distance - node.row
        if boundary == "left":
            return node.col + 1
        if boundary == "right":
            return self.distance - node.col
        raise ValueError(f"unknown boundary: {boundary}")

    def path_between(self, a: Node, b: Node) -> set[Edge]:
        """Return a deterministic shortest path between two defects."""

        edges: set[Edge] = set()
        r, c = a.row, a.col

        # Move vertically first.
        while r < b.row:
            edges.add(Edge("v", r, c))
            r += 1
        while r > b.row:
            edges.add(Edge("v", r - 1, c))
            r -= 1

        # Then move horizontally.
        while c < b.col:
            edges.add(Edge("h", r, c))
            c += 1
        while c > b.col:
            edges.add(Edge("h", r, c - 1))
            c -= 1

        return edges

    def path_to_boundary(self, node: Node, boundary: Boundary) -> set[Edge]:
        """Return a deterministic shortest path from a defect to a boundary."""

        edges: set[Edge] = set()
        r, c = node.row, node.col

        if boundary == "top":
            while r > 0:
                edges.add(Edge("v", r - 1, c))
                r -= 1
            edges.add(Edge("bt", -1, c))
            return edges

        if boundary == "bottom":
            while r < self.distance - 1:
                edges.add(Edge("v", r, c))
                r += 1
            edges.add(Edge("bb", self.distance, c))
            return edges

        if boundary == "left":
            while c > 0:
                edges.add(Edge("h", r, c - 1))
                c -= 1
            edges.add(Edge("bl", r, -1))
            return edges

        if boundary == "right":
            while c < self.distance - 1:
                edges.add(Edge("h", r, c))
                c += 1
            edges.add(Edge("br", r, self.distance))
            return edges

        raise ValueError(f"unknown boundary: {boundary}")

    def logical_parity(self, chain: Iterable[Edge]) -> int:
        """Compute logical parity of a residual chain.

        For Z-error decoding, a top-to-bottom chain is logical and is detected
        by crossing the middle horizontal cut with a vertical edge.

        For X-error decoding, a left-to-right chain is logical and is detected
        by crossing the middle vertical cut with a horizontal edge.
        """

        cut = self.distance // 2
        parity = 0
        for edge in chain:
            if self.error_type == "z":
                if edge.kind == "v" and edge.row == cut - 1:
                    parity ^= 1
            else:
                if edge.kind == "h" and edge.col == cut - 1:
                    parity ^= 1
        return parity


def xor_edges(a: Iterable[Edge], b: Iterable[Edge]) -> set[Edge]:
    """Symmetric difference of two edge chains."""

    result = set(a)
    for edge in b:
        if edge in result:
            result.remove(edge)
        else:
            result.add(edge)
    return result
