"""
DAG utilities: topological sort of edges, edge-to-source-agent mapping.

For the linear 3-agent pipeline (user → agent_1 → agent_2 → agent_3) this
simply yields [e_0, e_1, e_2], but the implementation handles arbitrary DAGs.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

from data.schema import Edge, ExecutionGraph


def topological_sort(graph: ExecutionGraph) -> List[Edge]:
    """
    Return edges in topological order (causal dependency order).

    An edge e_out produced by agent A depends on all edges in A.input_edges.
    We build an edge-level dependency graph and run Kahn's algorithm.

    Returns:
        Ordered list of Edge objects, e.g. [e_0, e_1, e_2].
    """
    edges_by_id: Dict[str, Edge] = {e.edge_id: e for e in graph.edges}

    consumed_by: Dict[str, List[str]] = {}
    produced_by: Dict[str, Optional[str]] = {}

    for node in graph.nodes:
        if node.id == "user":
            continue
        consumed_by[node.id] = []
        produced_by[node.id] = None

    for edge in graph.edges:
        receiver = edge.to_node
        if receiver in consumed_by:
            consumed_by[receiver].append(edge.edge_id)
        sender = edge.from_node
        if sender in produced_by:
            produced_by[sender] = edge.edge_id

    # Edge-level dependency graph:
    #   deps[e_out] = list of e_in that must come before e_out
    # e_out is produced by some agent A; e_out depends on all edges consumed by A.
    deps: Dict[str, List[str]] = {e.edge_id: [] for e in graph.edges}
    for agent_id, out_edge_id in produced_by.items():
        if out_edge_id is None:
            continue
        for in_edge_id in consumed_by.get(agent_id, []):
            deps[out_edge_id].append(in_edge_id)

    # Kahn's algorithm on edges
    in_degree: Dict[str, int] = {eid: len(prereqs) for eid, prereqs in deps.items()}
    successors: Dict[str, List[str]] = {eid: [] for eid in deps}
    for eid, prereqs in deps.items():
        for pre in prereqs:
            successors[pre].append(eid)

    queue = deque(eid for eid, deg in in_degree.items() if deg == 0)
    sorted_edge_ids: List[str] = []
    while queue:
        eid = queue.popleft()
        sorted_edge_ids.append(eid)
        for succ in successors[eid]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(sorted_edge_ids) != len(graph.edges):
        raise ValueError("Cycle detected in execution graph — not a DAG.")

    return [edges_by_id[eid] for eid in sorted_edge_ids]


def edge_source_agent_id(edge: Edge) -> Optional[int]:
    """
    Return the numeric agent_id of the edge's sender, or None if sender is 'user'.

    e.g. "agent_2" → 2, "user" → None
    """
    if edge.from_node == "user":
        return None
    try:
        return int(edge.from_node.split("_")[1])
    except (IndexError, ValueError):
        raise ValueError(f"Cannot parse agent id from node name: '{edge.from_node}'")


def edge_receiver_agent_id(edge: Edge) -> int:
    """Return the numeric agent_id of the edge's receiver."""
    try:
        return int(edge.to_node.split("_")[1])
    except (IndexError, ValueError):
        raise ValueError(f"Cannot parse agent id from node name: '{edge.to_node}'")
