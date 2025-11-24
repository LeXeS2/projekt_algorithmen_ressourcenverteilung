"""Pure-Python Network Simplex Implementation.

Diese Implementierung vermeidet Abhängigkeiten von externen Paketen.
Sie ist eine vereinfachte, aber praktische Version des Network Simplex für
das Transport-/Min-Cost-Flow-Problem, angepasst an die `Node`/`Edge`
Dataclasses des Projekts.

Vorgehen (Kurzfassung):
 - Erzeuge vollständiges gerichtetes Graph-Modell: vorhandene Kanten mit
   gegebenen Kosten und Kapazitäten; fehlende Kanten werden als "künstliche"
   Kanten mit sehr hohen Kosten hinzugefügt, um stets eine initiale
   zulässige Basis zu konstruieren.
 - Baue eine Anfangs-Baumlösung (Spanning Tree) und weise Flüsse auf den
   Baumkanten zu, sodass die Knotensupplies erfüllt sind.
 - Wiederhole: berechne Baum-Potenziale, finde eine Nicht-Baum-Kante mit
   negativem reduzierten Kosten; führe Pivot durch (Finde Zyklus, augmentiere
   Fluss, ersetze knotenbildende Baumkante). Beende, wenn keine negative
   reduzierte Kosten mehr existiert.

Einschränkungen / Hinweise:
 - Diese Implementierung ist absichtlich lesbar und kommentiert; sie ist nicht
   auf maximale Performance optimiert (für sehr große Graphen kann sie langsam
   sein).
 - Künstliche Kanten werden in den Ergebnissen ignoriert (nur gegebene
   `edges` werden zurückgeschrieben).
"""

from typing import List, Dict, Tuple, Optional, Any
from collections import deque

from node import Node
from edge import Edge

INF = 10 ** 18
ARTIFICIAL_COST = 10 ** 9


class _Arc:
    __slots__ = ("u", "v", "cost", "cap", "flow", "is_artificial")

    def __init__(self, u: int, v: int, cost: int, cap: int, is_artificial: bool = False):
        self.u = u
        self.v = v
        self.cost = cost
        self.cap = cap
        self.flow = 0
        self.is_artificial = is_artificial


def network_simplex(nodes: List[Node], edges: List[Edge], costs: Optional[Dict[Tuple[int, int], int]] = None, capacities: Optional[Dict[Tuple[int, int], int]] = None) -> Dict[str, Any]:
    """Network Simplex (pure Python).

    Args:
        nodes: Liste von Node-Objekten (ids können 0..n-1 sein)
        edges: Liste von Edge-Objekten; `transported` wird aktualisiert.
        costs: Optionales Mapping (u,v)->cost (default 1)
        capacities: Optionales Mapping (u,v)->capacity (default INF)

    Returns:
        Dict mit {'flow': total_positive_supply, 'cost': total_cost}
    """
    if costs is None:
        costs = {}
    if capacities is None:
        capacities = {}

    n = max((nd.id for nd in nodes), default=-1) + 1

    # compute total positive supply to bound capacities (prevents unbounded augmentation)
    total_supply = sum(max(0, int(nd.supply)) for nd in nodes)

    # Build complete set of arcs: use given edges, and add artificial arcs for missing pairs
    arc_list: List[_Arc] = []
    arc_index: Dict[Tuple[int, int], int] = {}

    # add real edges
    for e in edges:
        u, v = e.source, e.target
        c = int(costs.get((u, v), 1))
        # bound capacity by total supply to keep augmentation finite
        cap = int(capacities.get((u, v), total_supply if total_supply > 0 else INF))
        ai = len(arc_list)
        arc_list.append(_Arc(u, v, c, cap, is_artificial=False))
        arc_index[(u, v)] = ai

    # add artificial edges for any missing directed pair
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            if (u, v) not in arc_index:
                ai = len(arc_list)
                # artificial arcs: give them only necessary capacity (total_supply)
                art_cap = total_supply if total_supply > 0 else int(1e12)
                arc_list.append(_Arc(u, v, ARTIFICIAL_COST, art_cap, is_artificial=True))
                arc_index[(u, v)] = ai

    m = len(arc_list)

    # Build adjacency of tree representation (we will maintain a spanning tree by parent pointers)
    # Start with a simple spanning tree along nodes: edges (0->1, 1->2, ...)
    parent = [-1] * n
    parent_edge = [-1] * n  # index in arc_list of the tree edge connecting node to parent
    root = 0
    # Build an initial spanning tree preferring real (non-artificial) arcs.
    parent = [-1] * n
    parent_edge = [-1] * n
    root = 0
    visited = [False] * n
    visited[root] = True
    from collections import deque as _deque
    q = _deque([root])
    while q:
        u = q.popleft()
        # try neighbors v reachable via a real arc (u->v) or (v->u)
        for v in range(n):
            if visited[v] or v == u:
                continue
            chosen_idx = -1
            # prefer existing real arc u->v
            if (u, v) in arc_index and not arc_list[arc_index[(u, v)]].is_artificial:
                chosen_idx = arc_index[(u, v)]
            # else prefer real arc v->u
            elif (v, u) in arc_index and not arc_list[arc_index[(v, u)]].is_artificial:
                chosen_idx = arc_index[(v, u)]
            if chosen_idx >= 0:
                parent[v] = u
                parent_edge[v] = chosen_idx
                visited[v] = True
                q.append(v)

    # connect any still-unvisited nodes using artificial arcs (there will always be an artificial arc available)
    for v in range(n):
        if not visited[v]:
            # connect via an artificial arc from root to v (should exist)
            if (root, v) in arc_index:
                parent[v] = root
                parent_edge[v] = arc_index[(root, v)]
            else:
                # fallback: use any arc that touches v
                for u in range(n):
                    if (u, v) in arc_index:
                        parent[v] = u
                        parent_edge[v] = arc_index[(u, v)]
                        break

    # maintain set of tree edge indices for easy updates
    tree_edges = set(ei for ei in parent_edge if ei >= 0)

    # Assign initial flows on tree to satisfy supplies: do a post-order accumulation
    supply = [int(nd.supply) for nd in nodes]

    children = [[] for _ in range(n)]
    for v in range(n):
        p = parent[v]
        if p != -1:
            children[p].append(v)

    order = []
    # produce a DFS order; we need post-order so use stack with visited marker
    stack = [(root, False)]
    while stack:
        v, done = stack.pop()
        if done:
            order.append(v)
            continue
        stack.append((v, True))
        for w in children[v]:
            stack.append((w, False))

    # flows on tree edges: push subtree net supply up to parent along parent_edge
    for v in order:
        if v == root:
            continue
        eidx = parent_edge[v]
        if eidx < 0:
            continue
        amt = supply[v]
        # direct arc from parent to v or v to parent?
        arc = arc_list[eidx]
        if arc.u == parent[v] and arc.v == v:
            arc.flow += amt
        else:
            # arc opposite direction: represent as negative flow on that arc
            arc.flow -= amt
        supply[parent[v]] += amt

    # Now we have a feasible tree solution (flows on tree arcs satisfy supplies)

    # Helper: compute node potentials based on tree
    def compute_potentials() -> List[int]:
        pot = [0] * n
        # BFS from root
        q = deque([root])
        seen = [False] * n
        seen[root] = True
        while q:
            u = q.popleft()
            for v in range(n):
                # check if tree edge connects u<->v
                if parent[v] == u:
                    eidx = parent_edge[v]
                    arc = arc_list[eidx]
                    # potential[v] - potential[u] = cost(u->v) if arc.u==u and arc.v==v
                    if arc.u == u and arc.v == v:
                        pot[v] = pot[u] + arc.cost
                    else:
                        # edge in reverse direction
                        pot[v] = pot[u] - arc.cost
                    seen[v] = True
                    q.append(v)
        return pot

    # compute reduced cost for an arc given potentials: rc = cost + pot[u] - pot[v]
    def reduced_cost(arc: _Arc, pot: List[int]) -> int:
        return arc.cost + pot[arc.u] - pot[arc.v]

    # helper to find path between two nodes in tree and return sequence of (node,edgeIndex)
    def path_in_tree(u: int, v: int) -> List[int]:
        # produce path of nodes u->...->v via ancestors using parent pointers
        anc_u = set()
        x = u
        while x != -1:
            anc_u.add(x)
            x = parent[x]
        path = []
        y = v
        # walk v->... until hitting ancestor or root
        while y not in anc_u and y != -1:
            path.append(y)
            y = parent[y]
        if y == -1:
            # no common ancestor (disconnected tree) -- fallback: return simple path
            return [u, v]
        # y is LCA
        lca = y
        up = []
        x = u
        while x != lca:
            up.append(x)
            x = parent[x]
        up.append(lca)
        full = up + list(reversed(path))
        return full

    # main pivot loop
    max_iters = max(10000, m * 10)
    iter_count = 0
    banned_enter = set()
    while True:
        iter_count += 1
        if iter_count > max_iters:
            raise RuntimeError("Network simplex did not converge within iteration limit")

        pot = compute_potentials()

        # find an entering arc with negative reduced cost
        enter_idx = -1
        min_rc = 0
        for idx, arc in enumerate(arc_list):
            # skip arcs that are currently in the tree or temporarily banned
            if idx in tree_edges or idx in banned_enter:
                continue
            rc = reduced_cost(arc, pot)
            if rc < min_rc:
                min_rc = rc
                enter_idx = idx

        if enter_idx == -1:
            break  # optimal

        # entering arc
        ent = arc_list[enter_idx]

        # find path nodes sequence from ent.u -> ent.v
        path_nodes = path_in_tree(ent.u, ent.v)


        # determine orientation and compute max allowable augmentation (theta)
        # Build path as list of nodes from ent.u -> ... -> ent.v
        path_nodes = path_in_tree(ent.u, ent.v)
        # ensure the cycle is path u->...->v plus entering arc v->u? entering arc is u->v
        # cycle edges are consecutive pairs on path_nodes
        theta = INF
        leaving_idx = -1

        # examine each tree edge along path and compute available augmentation
        for i in range(len(path_nodes) - 1):
            a = path_nodes[i]
            b = path_nodes[i + 1]
            # find the tree edge index connecting a and b
            eidx = -1
            # if parent[b] == a, the edge is parent_edge[b]
            if parent[b] == a:
                eidx = parent_edge[b]
            elif parent[a] == b:
                eidx = parent_edge[a]
            else:
                # disconnected or unexpected; skip
                continue

            if eidx < 0:
                continue
            arc = arc_list[eidx]
            # cycle direction along this link is a->b
            # if the arc's stored direction matches a->b (arc.u==a and arc.v==b),
            # then increasing along cycle uses capacity - flow; otherwise limited by flow.
            if arc.u == a and arc.v == b:
                avail = arc.cap - arc.flow
            else:
                # arc oriented opposite to cycle direction; can decrease up to current flow
                avail = arc.flow

            if avail < theta:
                theta = avail
                leaving_idx = eidx

        # also consider entering arc capacity (we will increase ent)
        ent_avail = ent.cap - ent.flow
        if ent_avail < theta:
            theta = ent_avail
            leaving_idx = enter_idx

        if theta <= 0:
            # cannot augment; mark this entering arc as temporarily non-improving
            banned_enter.add(enter_idx)
            continue

        # apply augmentation: add theta along entering arc direction, then update tree edges along path
        # update entering arc
        ent.flow += theta

        # update flows on path from u -> v (consecutive node pairs)
        for i in range(len(path_nodes) - 1):
            a = path_nodes[i]
            b = path_nodes[i + 1]
            # find the tree edge index connecting a and b
            if parent[b] == a:
                eidx = parent_edge[b]
                arc = arc_list[eidx]
                # if arc direction aligns with a->b, increase; else decrease
                if arc.u == a and arc.v == b:
                    arc.flow += theta
                else:
                    arc.flow -= theta
            elif parent[a] == b:
                eidx = parent_edge[a]
                arc = arc_list[eidx]
                # a->b is opposite of stored direction b->a
                if arc.u == a and arc.v == b:
                    arc.flow += theta
                else:
                    arc.flow -= theta
            else:
                # unexpected, skip
                continue

        # after successful augmentation, clear banned enters (they may become improving later)
        banned_enter.clear()

        # update tree: if leaving arc is not entering, replace it in the tree_edges set
        if leaving_idx != enter_idx:
            if leaving_idx in tree_edges:
                tree_edges.remove(leaving_idx)
            tree_edges.add(enter_idx)

            # rebuild parent and parent_edge arrays from the tree_edges (BFS)
            new_parent = [-1] * n
            new_parent_edge = [-1] * n
            q = deque([root])
            visited = [False] * n
            visited[root] = True
            while q:
                u = q.popleft()
                for eidx in list(tree_edges):
                    arc = arc_list[eidx]
                    # treat tree edges as undirected for parent assignment
                    if arc.u == u and not visited[arc.v]:
                        new_parent[arc.v] = u
                        new_parent_edge[arc.v] = eidx
                        visited[arc.v] = True
                        q.append(arc.v)
                    elif arc.v == u and not visited[arc.u]:
                        new_parent[arc.u] = u
                        new_parent_edge[arc.u] = eidx
                        visited[arc.u] = True
                        q.append(arc.u)

            parent = new_parent
            parent_edge = new_parent_edge
            # continue; this ensures path queries use the updated tree

        # loop continue until optimal

    # If any artificial arc still carries flow, the original problem was infeasible
    for arc in arc_list:
        if arc.is_artificial and arc.flow != 0:
            raise ValueError("Infeasible: artificial arc carries flow -> no feasible flow satisfies supplies/demands")

    # write back flows into original Edge objects
    for e in edges:
        idx = arc_index.get((e.source, e.target))
        if idx is not None:
            arc = arc_list[idx]
            # if arc.flow negative, it means flow goes opposite direction
            if arc.flow >= 0:
                transported = int(arc.flow)
            else:
                transported = int(-arc.flow)
            e.transported = max(0, transported)
        else:
            e.transported = 0

    # compute total cost using transported values on the original edges and provided costs
    total_cost = 0
    for e in edges:
        c = int(costs.get((e.source, e.target), 1))
        transported = int(e.transported) if e.transported >= 0 else 0
        total_cost += transported * c

    total_supply = sum(max(0, int(nd.supply)) for nd in nodes)
    return {"flow": int(total_supply), "cost": int(total_cost)}


if __name__ == "__main__":
    # Demo: use generator to create a complete digraph (feasible)
    from generator import generate_random_directed_graph

    num_nodes = 6
    max_edges = num_nodes * (num_nodes - 1)
    nodes, edges = generate_random_directed_graph(num_nodes=num_nodes, num_edges=max_edges, seed=42, supply_range=5, balance_demand=True)
    costs = {(e.source, e.target): 1 for e in edges}

    res = network_simplex(nodes, edges, costs=costs)
    print("Result:", res)
    for e in edges:
        print(e)
