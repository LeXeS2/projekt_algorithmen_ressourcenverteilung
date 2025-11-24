from network_simplex_algorithm import network_simplex
from node import Node
from edge import Edge


def test_simple_split():
    # Node 0 supply +5, Node1 -3, Node2 -2
    nodes = [Node(id=0, supply=5), Node(id=1, supply=-3), Node(id=2, supply=-2)]
    edges = [Edge(0,1,transported=-1), Edge(0,2,transported=-1)]
    costs = {(0,1): 1, (0,2): 2}
    res = network_simplex(nodes, edges, costs=costs)
    print('test_simple_split res:', res)
    for e in edges:
        print(e)


if __name__ == '__main__':
    test_simple_split()
