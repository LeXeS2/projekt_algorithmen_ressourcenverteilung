"""Successive Shortest Path (Min-Cost Flow) Implementation

Dieses Modul implementiert einen einfachen Successive Shortest Path (SSP)
Algorithmus zur Lösung des Transportproblems (Weizenfelder -> Mühlen).

Funktionen:
- `successive_shortest_path(nodes, edges, costs=None, capacities=None)`:
    Füllt die `transported`-Felder der übergebenen `Edge`-Objekte mit der
    berechneten Flowmenge (int). `costs` ist optional ein Dict[(u,v)]->cost.

Hinweis:
- Die vorhandenen `Node` und `Edge` Dataclasses aus dem Repo werden verwendet.
- Standardkosten für Kanten ohne Angabe: 1. Standardkapazität: sehr groß.
"""
from typing import List, Dict, Tuple, Optional
import heapq

from node import Node
from edge import Edge

INF = 10 ** 12


class _ResEdge:
    __slots__ = ("to", "rev", "cap", "cost")

    def __init__(self, to: int, rev: int, cap: int, cost: int):
        self.to = to
        self.rev = rev
        self.cap = cap
        self.cost = cost


def _add_edge(graph: List[List[_ResEdge]], u: int, v: int, cap: int, cost: int):
    graph[u].append(_ResEdge(v, len(graph[v]), cap, cost))
    graph[v].append(_ResEdge(u, len(graph[u]) - 1, 0, -cost))


def _min_cost_flow(graph: List[List[_ResEdge]], s: int, t: int, maxf: int) -> Tuple[int, int]:
    n = len(graph)
    prevv = [0] * n
    preve = [0] * n
    dist = [0] * n
    potential = [0] * n  # Johnson potentials to handle negative costs

    flow = 0
    cost = 0
    while flow < maxf:
        # Dijkstra on reduced costs
        for i in range(n):
            dist[i] = INF
        dist[s] = 0
        pq = [(0, s)]
        while pq:
            d, v = heapq.heappop(pq)
            if dist[v] < d:
                continue
            for ei, e in enumerate(graph[v]):
                if e.cap > 0:
                    nd = dist[v] + e.cost + potential[v] - potential[e.to]
                    if dist[e.to] > nd:
                        dist[e.to] = nd
                        prevv[e.to] = v
                        preve[e.to] = ei
                        heapq.heappush(pq, (nd, e.to))

        if dist[t] == INF:
            break  # cannot send more flow

        for v in range(n):
            if dist[v] < INF:
                potential[v] += dist[v]

        # augment as much as possible on found shortest path
        addf = maxf - flow
        v = t
        while v != s:
            e = graph[prevv[v]][preve[v]]
            if addf > e.cap:
                addf = e.cap
            v = prevv[v]

        v = t
        while v != s:
            e = graph[prevv[v]][preve[v]]
            e.cap -= addf
            graph[v][e.rev].cap += addf
            v = prevv[v]

        flow += addf
        cost += addf * potential[t]

    return flow, cost


def successive_shortest_path(nodes: List[Node], edges: List[Edge], costs: Optional[Dict[Tuple[int, int], int]] = None, capacities: Optional[Dict[Tuple[int, int], int]] = None) -> Dict[str, int]:
    """Berechnet einen (minimum-cost) Fluss, der alle Supplies erfüllt.

    Args:
        nodes: Liste von Node-Objekten (id: 0..n-1)
        edges: Liste von Edge-Objekten (source, target). `transported` wird aktualisiert.
        costs: Optionales Mapping (u,v) -> cost (int). Default: 1.
        capacities: Optionales Mapping (u,v) -> capacity (int). Default: sehr groß.

    Returns:
        Dict mit Zusammenfassung: {'flow': int, 'cost': int}
    """
    if costs is None:
        costs = {}
    if capacities is None:
        capacities = {}

    # determine number of nodes (assume ids are 0..n-1 but be robust)
    n = max((nd.id for nd in nodes), default=-1) + 1
    ss = n
    tt = n + 1
    graph: List[List[_ResEdge]] = [[] for _ in range(n + 2)]

    # add original edges and remember forward edge references
    original_edge_refs: List[Tuple[int, int, int]] = []  # (u, index_in_graph[u], initial_capacity)
    for e in edges:
        u, v = e.source, e.target
        cap = capacities.get((u, v), INF)
        cost = costs.get((u, v), 1)
        idx = len(graph[u])
        _add_edge(graph, u, v, cap, cost)
        original_edge_refs.append((u, idx, cap))

    # connect super-source and super-sink according to node supplies
    total_supply = 0
    for nd in nodes:
        if nd.supply > 0:
            _add_edge(graph, ss, nd.id, nd.supply, 0)
            total_supply += nd.supply
        elif nd.supply < 0:
            _add_edge(graph, nd.id, tt, -nd.supply, 0)

    # run min-cost flow from ss to tt with desired flow = total_supply
    flow, cost = _min_cost_flow(graph, ss, tt, total_supply)

    # assign transported values back to original Edge objects
    for i, e in enumerate(edges):
        u, idx, init_cap = original_edge_refs[i]
        # forward edge in graph[u][idx]
        forward_edge = graph[u][idx]
        # flow sent = initial_capacity - remaining capacity
        sent = init_cap - forward_edge.cap
        e.transported = int(sent)

    return {"flow": int(flow), "cost": int(cost)}


if __name__ == "__main__":
    # Kleines Demo: generiere einen ausgeglichenen Graphen und löse ihn
    from generator import generate_random_directed_graph

    # Demo-Parameter (kann angepasst werden)
    nodes, edges = generate_random_directed_graph(num_nodes=6, num_edges=12, seed=42, supply_range=5, balance_demand=True)

    # Beispiel: setze Kosten = 1 für alle Kanten (kann angepasst werden)
    costs = {(e.source, e.target): 1 for e in edges}

    result = successive_shortest_path(nodes, edges, costs=costs)

    print("Result:", result)
    print("Assigned edges:")
    for e in edges:
        assigned = e.transported if e.transported >= 0 else "unassigned"
        print(f"  {e.source} -> {e.target} transported={assigned}")
