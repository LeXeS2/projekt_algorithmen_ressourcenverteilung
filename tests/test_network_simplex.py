import copy

from generator import generate_random_directed_graph
from node import Node
from edge import Edge

from network_simplex_algorithm import network_simplex
from successive_shortest_path import successive_shortest_path


def test_simple_split():
    # simple deterministic example: node0 supplies 5, node1 consumes 3, node2 consumes 2
    nodes = [Node(id=0, supply=5), Node(id=1, supply=-3), Node(id=2, supply=-2)]
    edges = [Edge(0, 1), Edge(0, 2)]
    costs = {(0, 1): 1, (0, 2): 2}

    res = network_simplex(copy.deepcopy(nodes), copy.deepcopy(edges), costs=costs)

    # verify flows were assigned on the original call (function writes into passed edges),
    # so we call again to inspect transported values from returned edges instance
    nodes2 = copy.deepcopy(nodes)
    edges2 = copy.deepcopy(edges)
    res2 = network_simplex(nodes2, edges2, costs=costs)

    assert res2["flow"] == 5
    # expected transported values: 3 units to node1 (cost 1) and 2 units to node2 (cost 2)
    transported = [e.transported for e in edges2]
    assert transported == [3, 2]
    assert res2["cost"] == 1 * 3 + 2 * 2