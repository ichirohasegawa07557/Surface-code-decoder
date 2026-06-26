from src.decoder import BruteForceMWPMDecoder, UnionFindDecoder
from src.lattice import Edge, Node, PlanarSurfaceCodeLattice, xor_edges
from src.spacetime import SpaceTimeDecodingGraph, SpaceTimeMWPMDecoder


def test_single_boundary_error_is_corrected():
    lattice = PlanarSurfaceCodeLattice(distance=3)
    decoder = BruteForceMWPMDecoder(lattice)

    error_edges = {Edge("bt", -1, 1)}
    syndrome = lattice.toggle_syndrome_from_edges(error_edges)

    decoded = decoder.decode(syndrome)
    residual = xor_edges(error_edges, decoded.correction_edges)

    assert syndrome == {Node(0, 1)}
    assert lattice.logical_parity(residual) == 0


def test_pair_of_neighboring_defects_is_corrected():
    lattice = PlanarSurfaceCodeLattice(distance=5)
    decoder = BruteForceMWPMDecoder(lattice)

    error_edges = {Edge("h", 2, 1)}
    syndrome = lattice.toggle_syndrome_from_edges(error_edges)

    decoded = decoder.decode(syndrome)
    residual = xor_edges(error_edges, decoded.correction_edges)

    assert syndrome == {Node(2, 1), Node(2, 2)}
    assert lattice.logical_parity(residual) == 0


def test_logical_chain_has_odd_parity():
    lattice = PlanarSurfaceCodeLattice(distance=5)
    chain = {
        Edge("bt", -1, 2),
        Edge("v", 0, 2),
        Edge("v", 1, 2),
        Edge("v", 2, 2),
        Edge("v", 3, 2),
        Edge("bb", 5, 2),
    }

    assert lattice.logical_parity(chain) == 1


def test_x_error_type_uses_left_right_boundaries():
    lattice = PlanarSurfaceCodeLattice(distance=5, error_type="x")
    decoder = BruteForceMWPMDecoder(lattice)

    error_edges = {Edge("bl", 2, -1)}
    syndrome = lattice.toggle_syndrome_from_edges(error_edges)

    decoded = decoder.decode(syndrome)
    residual = xor_edges(error_edges, decoded.correction_edges)

    assert syndrome == {Node(2, 0)}
    assert lattice.logical_parity(residual) == 0


def test_union_find_decoder_runs_on_simple_pair():
    lattice = PlanarSurfaceCodeLattice(distance=5)
    decoder = UnionFindDecoder(lattice)

    error_edges = {Edge("v", 1, 2)}
    syndrome = lattice.toggle_syndrome_from_edges(error_edges)
    decoded = decoder.decode(syndrome)

    assert len(decoded.correction_edges) > 0
    assert decoded.total_weight >= 1


def test_spacetime_graph_decoder_runs():
    graph = SpaceTimeDecodingGraph(distance=3, rounds=3)
    sample = graph.sample_noise(0.0, 0.0, __import__("random").Random(1))
    assert sample.defects == set()

    decoder = SpaceTimeMWPMDecoder(graph)
    decoded = decoder.decode(set())
    assert decoded.total_weight == 0
