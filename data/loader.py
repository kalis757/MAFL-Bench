"""
Log loader: parse a v0.2 JSON log file into an ExecutionLog dataclass.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .schema import (
    AgentRecord, AttackConfig, BasicPrompt, Edge, ExecutionGraph, ExecutionLog,
    ExecutionStep, GraphNode, GroundTruth, Metadata, ModuleLabels, Plan,
    PlanRecord, PlanStep, PotExample, PotPrompt, RetrievedMemory, RetrievedPair,
    SystemPrompt,
)

logger = logging.getLogger(__name__)


def _parse_attack_config(d: dict) -> AttackConfig:
    return AttackConfig(
        attack_type=d.get("attack_type", "CLEAN"),
        variant=d.get("variant", ""),
        targeted_agent=d.get("targeted_agent"),
        description=d.get("description", ""),
        folder_name=d.get("folder_name", ""),
    )


def _infer_topology_from_scenario_name(scenario_name: str) -> Optional[str]:
    """Return topology type if the scenario name contains a known topology keyword."""
    name = scenario_name.lower()
    for ttype in ("random", "tree", "complete"):
        if ttype in name:
            return ttype
    return None


def _parse_metadata(d: dict) -> Metadata:
    topology_type = d.get("topology_type") or _infer_topology_from_scenario_name(
        d.get("scenario_name", "")
    ) or "chain"
    return Metadata(
        run_id=d["run_id"],
        query_id=d.get("query_id", ""),
        scenario_name=d.get("scenario_name", ""),
        model_name=d.get("model_name", ""),
        executor_mode=d.get("executor_mode", ""),
        attack_config=_parse_attack_config(d.get("attack_config", {})),
        topology_type=topology_type,
        validation_warning=d.get("validation_warning", False),
        payload=d.get("payload"),
        random_seed=d.get("random_seed"),
        discussion_order=d.get("discussion_order", []),
        random_rounds=d.get("random_rounds", 0),
    )


def _parse_graph_node(d: dict) -> GraphNode:
    return GraphNode(id=d["id"], role=d.get("role"))


def _parse_edge(d: dict) -> Edge:
    return Edge(
        edge_id=d["edge_id"],
        from_node=d["from"],
        to_node=d["to"],
        content=d.get("content", ""),
        is_poisoned=d.get("is_poisoned", False),
        poison_type=d.get("poison_type"),
        edge_type=d.get("edge_type"),
    )


def _parse_execution_graph(d: dict) -> ExecutionGraph:
    nodes = [_parse_graph_node(n) for n in d.get("nodes", [])]
    edges = [_parse_edge(e) for e in d.get("edges", [])]
    return ExecutionGraph(nodes=nodes, edges=edges)


def _parse_module_labels(d: dict) -> ModuleLabels:
    return ModuleLabels(
        POT=d.get("POT", -1),
        MEM=d.get("MEM", -1),
        INP=d.get("INP", -1),
        OBS=d.get("OBS", -1),
    )


def _parse_pot_prompt(d: dict) -> PotPrompt:
    examples = [
        PotExample(
            query=ex.get("query", ""),
            plan=ex.get("plan", ""),
            label=ex.get("label", 0),
        )
        for ex in d.get("examples", [])
    ]
    return PotPrompt(
        content=d.get("content", ""),
        examples=examples,
        label=d.get("label", -1),
    )


def _parse_system_prompt(d: dict) -> SystemPrompt:
    basic_d = d.get("basic", {})
    basic = BasicPrompt(
        content=basic_d.get("content", ""),
        label=basic_d.get("label", -1),
    )
    pot = _parse_pot_prompt(d.get("pot_prompt", {"content": "", "examples": [], "label": -1}))
    return SystemPrompt(basic=basic, pot_prompt=pot)


def _parse_retrieved_memory(d: dict) -> RetrievedMemory:
    pairs = []
    for p in d.get("retrieved_pairs", []):
        pairs.append(RetrievedPair(
            memory_id=p.get("memory_id", ""),
            query=p.get("query", ""),
            plan=p.get("plan", ""),
            is_poisoned=p.get("is_poisoned", False),
            similarity_score=p.get("similarity_score", 0.0),
            label=p.get("label", 0),
            poison_targeted_agent=p.get("poison_targeted_agent"),
        ))
    return RetrievedMemory(
        query_used_for_retrieval=d.get("query_used_for_retrieval", ""),
        top_k=d.get("top_k", 0),
        retrieved_pairs=pairs,
        label=d.get("label", 0),
    )


def _parse_execution_history(steps: list) -> List[ExecutionStep]:
    result = []
    for s in steps:
        result.append(ExecutionStep(
            step=s.get("step"),
            plan_step_action=s.get("plan_step_action", ""),
            tool_called=s.get("tool_called", ""),
            params=s.get("params", {}),
            observation=s.get("observation", ""),
            label=s.get("label", 0),
            is_malicious_tool=s.get("is_malicious_tool", False),
        ))
    return result


def _parse_plan_record(d: dict) -> PlanRecord:
    raw_plan = d.get("plan")
    # plan_record.plan may be stored as a flat list of step dicts (the raw steps array)
    # rather than the full Plan object dict {"steps": [...], "raw_output": ..., "label": ...}.
    if isinstance(raw_plan, list):
        raw_plan = {"steps": raw_plan, "raw_output": "", "label": -1}
    return PlanRecord(
        phase_index=d.get("phase_index", 0),
        input=d.get("input", ""),
        plan=_parse_plan(raw_plan),
    )


def _parse_plan(d: Optional[dict]) -> Optional[Plan]:
    if d is None:
        return None
    steps = [
        PlanStep(
            step=s.get("step", 0),
            action=s.get("action", ""),
            tool=s.get("tool", ""),
        )
        for s in d.get("steps", [])
    ]
    return Plan(
        steps=steps,
        raw_output=d.get("raw_output", ""),
        label=d.get("label", -1),
    )


def _parse_agent(d: dict) -> AgentRecord:
    plan_records = [_parse_plan_record(r) for r in d.get("plan_records", [])]
    output_edge = d.get("output_edge")
    output_edges = list(d.get("output_edges", []))
    if output_edge and output_edge not in output_edges:
        output_edges.insert(0, output_edge)
    if output_edge is None and output_edges:
        output_edge = output_edges[0]
    return AgentRecord(
        agent_id=d["agent_id"],
        role=d.get("role", ""),
        input_edges=d.get("input_edges", []),
        system_prompt=_parse_system_prompt(d.get("system_prompt", {})),
        retrieved_memory=_parse_retrieved_memory(d.get("retrieved_memory", {})),
        raw_model_output=d.get("raw_model_output", ""),
        output_edge=output_edge,
        module_labels=_parse_module_labels(d.get("module_labels", {})),
        output_edges=output_edges,
        execution_history=_parse_execution_history(d.get("execution_history", [])),
        plan=_parse_plan(d.get("plan")),
        output_message=d.get("output_message"),
        final_response=d.get("final_response"),
        parse_mode=d.get("parse_mode", "strict_json"),
        plan_records=plan_records,
    )


def _parse_ground_truth(d: dict) -> GroundTruth:
    return GroundTruth(
        attack_succeeded=d.get("attack_succeeded", False),
        malicious_tool_name=d.get("malicious_tool_name"),
        invoked_by_agent_id=d.get("invoked_by_agent_id"),
        earliest_poisoned_agent_id=d.get("earliest_poisoned_agent_id"),
        poisoned_module=d.get("poisoned_module"),
        poisoned_edge_ids=d.get("poisoned_edge_ids", []),
        malicious_tool_invoked=d.get("malicious_tool_invoked", False),
        plan_contains_malicious_step=d.get("plan_contains_malicious_step", False),
        malicious_instruction_in_text=d.get("malicious_instruction_in_text", False),
    )


def load_log(path: str | Path) -> ExecutionLog:
    """
    Parse a single JSON log file (v0.2) into an ExecutionLog.

    Raises:
        ValueError: if schema_version != "0.2".
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    version = raw.get("schema_version", "unknown")
    if version != "0.2":
        raise ValueError(f"Unsupported schema version '{version}' in {path}. Expected '0.2'.")

    metadata = _parse_metadata(raw["metadata"])
    original_user_query = raw.get("original_user_query", "")
    execution_graph = _parse_execution_graph(raw.get("execution_graph", {}))
    agents = [_parse_agent(a) for a in raw.get("agents", [])]
    ground_truth = _parse_ground_truth(raw.get("ground_truth", {}))

    # Build lookup maps
    agent_map: Dict[int, AgentRecord] = {a.agent_id: a for a in agents}
    edge_map: Dict[str, Edge] = {e.edge_id: e for e in execution_graph.edges}

    log = ExecutionLog(
        schema_version=version,
        metadata=metadata,
        original_user_query=original_user_query,
        execution_graph=execution_graph,
        agents=agents,
        ground_truth=ground_truth,
        agent_map=agent_map,
        edge_map=edge_map,
    )

    _validate_ground_truth_vs_module_labels(log)
    return log


def _validate_ground_truth_vs_module_labels(log: ExecutionLog) -> None:
    """
    Cross-validate ground_truth.earliest_poisoned_agent_id against module_labels scan.
    Logs a warning if they disagree.
    """
    gt_agent = log.ground_truth.earliest_poisoned_agent_id
    if gt_agent is None:
        return

    # Scan all agents for any module_labels == 1
    label1_agents = [
        a.agent_id for a in log.agents
        if any(getattr(a.module_labels, m) == 1 for m in ("POT", "MEM", "INP", "OBS"))
    ]

    if not label1_agents:
        return

    min_label1_agent = min(label1_agents)
    if min_label1_agent != gt_agent:
        logger.warning(
            "Ground truth earliest_poisoned_agent_id=%d but module_labels scan found "
            "agent %d with label=1. Using ground_truth value for evaluation.",
            gt_agent, min_label1_agent,
        )
