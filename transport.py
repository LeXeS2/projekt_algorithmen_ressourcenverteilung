"""
Transportzuweisung für das Transportproblem.

Funktionen:
- assign_transport_flows(nodes, edges, method="network_simplex", allow_unbalanced=False)
    Versucht, für jede Kante eine Transportmenge zu bestimmen, sodass die
    Knotensupplies (Produzenten/ Konsumenten) erfüllt werden.

Implementiert:
- NetworkX Network Simplex (sofern verfügbar) als präferierte, exakte Methode.
- Ein einfacher Greedy-Fallback, falls NetworkX nicht verfügbar oder das Problem
  unlösbar ist (z.B. unfeasible für network_simplex).

Die Funktion verändert die übergebene Liste `edges` und setzt `Edge.transported`
auf die berechneten Werte (int oder float). Ursprünglich -1 bedeutet unassigned.
"""
from typing import List, Dict, Tuple, Optional
from collections import deque
import math

from node import Node
from edge import Edge


def _reset_transported(edges: List[Edge]) -> None:
    """Initialisiert alle Edge.transported Felder: -1 -> 0, andere Werte belassen."""
    for e in edges:
        if e.transported < 0:
            e.transported = 0


def _build_edge_map(edges: List[Edge]) -> Dict[Tuple[int, int], Edge]:
    """Schneller Lookup von (source,target) -> Edge-Objekt."""
    return {(e.source, e.target): e for e in edges}


def _is_integer_like(x: float, eps: float = 1e-9) -> bool:
    return abs(x - round(x)) < eps


def _apply_flow_dict_to_edges(flow: Dict[int, Dict[int, float]], edges_map: Dict[Tuple[int, int], Edge]) -> None:
    """Wendet ein flow-dict von networkx (u -> {v: value}) auf die Edge-Objekte an."""
    for u, targets in flow.items():
        for v, val in targets.items():
            if (u, v) in edges_map:
                # Konvertiere zu int, wenn integral
                if isinstance(val, (int,)) or _is_integer_like(val):
                    edges_map[(u, v)].transported = int(round(val))
                else:
                    edges_map[(u, v)].transported = float(val)
            else:
                # Wenn die Kante nicht im originalen Edge-Set ist, ignoriere (kann passieren
                # wenn networkx intern Hilfskanten erzeugt - hier aber unwahrscheinlich).
                pass


def assign_transport_flows(nodes: List[Node], edges: List[Edge], method: str = "network_simplex", allow_unbalanced: bool = False) -> List[Edge]:
    """
    Bestimmt die Transportmenge für jede Kante.

    Args:
        nodes: Liste von Node-Objekten (node.supply: int, >0 = Produzent, <0 = Konsument)
        edges: Liste von Edge-Objekten (Edge.transported wird gesetzt)
        method: "network_simplex" (default) oder "greedy"
        allow_unbalanced: Wenn False, wird bei Sum(supply) != 0 eine Exception geworfen.
                         Wenn True, verteilt der Algorithmus nur die min(gesamt_prod, -gesamt_cons).

    Returns:
        Die veränderte Liste `edges` (gleiche Objekte, transportiert-Werte gesetzt).
    """
    total_supply = sum(n.supply for n in nodes)
    if total_supply != 0 and not allow_unbalanced:
        raise ValueError(f"Gesamtsumme der supplies ist {total_supply}, muss 0 sein. Setze allow_unbalanced=True um tolerant zu sein.")

    # Initialisiere transported Felder
    _reset_transported(edges)
    edges_map = _build_edge_map(edges)

    if method == "greedy":
        return greedy_assign(nodes, edges)

    # Versuche NetworkX network_simplex
    try:
        import networkx as nx
    except ImportError:
        # Fallback auf greedy, falls networkx nicht installiert ist
        print("Warnung: networkx nicht installiert, verwende Greedy-Fallback.")
        return greedy_assign(nodes, edges)

    # Baue DiGraph für networkx
    G = nx.DiGraph()
    # networkx: demand attribute: Sum of demands must be zero. demand > 0 = Nachfrage.
    # Wir setzen demand = -supply  (producer supply>0 -> demand negative = supply)
    for n in nodes:
        G.add_node(n.id, demand=-n.supply)

    # Wähle eine großzügige Kapazität (Standard: None/inf nicht erlaubt im network_simplex),
    # daher setzen wir Kapazität auf absolute Summe der positiven supplies oder ein großes int.
    cap = sum(max(0, n.supply) for n in nodes)
    if cap <= 0:
        cap = sum(max(0, -n.supply) for n in nodes)  # fallback

    if cap <= 0:
        cap = 10 ** 9  # arbitrary large if all supplies are zero (degenerate)

    for e in edges:
        # weight=0 (keine Kosten). Bei Bedarf kann dies erweitert werden.
        # capacity als cap (gilt für alle Kanten gleich).
        G.add_edge(e.source, e.target, capacity=cap, weight=0)

    try:
        cost, flow_dict = nx.network_simplex(G)
        # Apply results to edges
        _apply_flow_dict_to_edges(flow_dict, edges_map)
        return edges
    except nx.exception.NetworkXUnfeasible:
        # Falls unfeasible (z.B. nicht erreichbare Nachfrage), falle zurück auf greedy
        return greedy_assign(nodes, edges)
    except Exception:
        # Sonstige Fehler -> greedy fallback
        return greedy_assign(nodes, edges)


def greedy_assign(nodes: List[Node], edges: List[Edge]) -> List[Edge]:
    """
    Ein einfacher, nicht-optimaler Greedy-Algorithmus:
    - Arbeite mit Kopien der verbleibenden supplies/demands.
    - Priorisiere kürzeste (ungewichtete) Pfade von Produzenten zu Konsumenten.
    - Sende jeweils min(producer_rest, consumer_rest) entlang des Pfads und akkumuliert
      die Mengen auf den Kanten.

    Dieser Algorithmus liefert eine zulässige Verteilung, sofern die Nachfrage erreichbar ist
    und die Gesamtversorgung mindestens die Gesamtnachfrage abdecken kann (oder allow_unbalanced=True).
    """
    # Mapping und Initialisierung
    _reset_transported(edges)
    edges_map = _build_edge_map(edges)

    # Adjazenzliste (nur vorhandene gerichtete Kanten)
    adj: Dict[int, List[int]] = {}
    for e in edges:
        adj.setdefault(e.source, []).append(e.target)

    # Mutable Kopien der supplies (producer positive, consumer negative)
    node_supply: Dict[int, int] = {n.id: n.supply for n in nodes}

    # Helper: finde per BFS einen Pfad von src zu einem Ziel in targets_set
    def _bfs_find_path(src: int, targets_set: set, supply_condition=lambda nid: True) -> Optional[List[int]]:
        q = deque([src])
        prev: Dict[int, Optional[int]] = {src: None}
        while q:
            u = q.popleft()
            if u in targets_set and supply_condition(u):
                # Reconstruct path
                path = []
                cur = u
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                path.reverse()
                return path
            for v in adj.get(u, []):
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        return None

    # Listen der Produzenten und Konsumenten (IDs)
    producers = [n.id for n in nodes if n.supply > 0]
    consumers = [n.id for n in nodes if n.supply < 0]

    # Continue until no producer with remainder or no consumer with remainder
    producer_idx = 0
    # We'll loop over producers and try to satisfy consumers
    while True:
        # Find next producer with remaining supply
        prod = next((p for p in producers if node_supply[p] > 0), None)
        if prod is None:
            break
        # Build set of consumers with remaining demand
        consumers_remaining = {c for c in consumers if node_supply[c] < 0}
        if not consumers_remaining:
            break

        # Find nearest reachable consumer from this producer
        path = _bfs_find_path(prod, consumers_remaining, supply_condition=lambda nid: node_supply[nid] < 0)
        if path is None:
            # No reachable consumers from this producer -> mark producer as exhausted to avoid infinite loop
            # (it cannot send to any consumer)
            # To avoid skipping useful producers for other consumers, temporarily remove this producer.
            producers.remove(prod)
            continue

        # Determine amount to send: min(prod_rest, -consumer_rest)
        consumer = path[-1]
        amount = min(node_supply[prod], -node_supply[consumer])
        if amount <= 0:
            # Nothing to send (shouldn't happen), just continue
            producers.remove(prod)
            continue

        # Apply amount along path edges
        for u, v in zip(path, path[1:]):
            edge = edges_map.get((u, v))
            if edge is None:
                # Shouldn't happen because path built from adj, but guard anyway
                continue
            # Sum auf der Kante (bereits vorhandene Menge addieren)
            prev_val = edge.transported if edge.transported >= 0 else 0
            edge.transported = prev_val + amount

        # Update node supplies
        node_supply[prod] -= amount
        node_supply[consumer] += amount  # consumer is negative, so adding reduces magnitude

    # End: return edges (some producers/consumers may remain unsatisfied)
    return edges


if __name__ == "__main__":
    # Kleines CLI-Demo: erzeugt einen zufälligen Graphen und weist Flüsse zu.
    import argparse
    from generator import generate_random_directed_graph

    parser = argparse.ArgumentParser(description="Demo: Berechne Transportmengen für einen zufälligen Graphen")
    parser.add_argument("num_nodes", type=int, nargs="?", default=6, help="Anzahl Knoten")
    parser.add_argument("num_edges", type=int, nargs="?", default=10, help="Anzahl Kanten")
    parser.add_argument("seed", type=int, nargs="?", default=42, help="Seed für RNG")
    parser.add_argument("--method", choices=["network_simplex", "greedy"], default="network_simplex", help="Verwendeter Algorithmus")
    parser.add_argument("--balance-demand", action="store_true", help="Letzten Knoten so setzen, dass supplies sum = 0")
    args = parser.parse_args()

    nodes, edges = generate_random_directed_graph(args.num_nodes, args.num_edges, args.seed, balance_demand=args.balance_demand)

    try:
        edges = assign_transport_flows(nodes, edges, method=args.method)
    except ValueError as ve:
        print(f"Fehler: {ve}")
        # Wenn unbalanced ist und Methode fehlschlägt, versuche greedy tolerant aufzuteilen
        print("Versuche Greedy-Fallback mit partialer Verteilung...")
        edges = greedy_assign(nodes, edges)

    # Ausgabe
    print("Nodes:")
    for n in nodes:
        kind = "Producer" if n.is_producer() else ("Consumer" if n.is_consumer() else "Neutral")
        print(f"  id={n.id:2d} supply={n.supply:4d} ({kind})")
    print("\\nEdges:")
    for e in edges:
        assigned = e.transported if e.is_assigned() else "unassigned"
        print(f"  {e.source} -> {e.target} transported={assigned}")
