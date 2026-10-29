"""Context builders for model inputs."""

from __future__ import annotations

import logging
from typing import Optional

from data.schema import AgentRecord, Edge, ExecutionLog, RetrievedPair

logger = logging.getLogger(__name__)


def _format_edge(edge: Edge) -> str:
    return (
        f"\n=== Message from {edge.from_node} to {edge.to_node} ===\n"
        f"{edge.content}\n"
    )


def _format_memory_pair(pair: RetrievedPair) -> str:
    return (
        f"- Memory ID: {pair.memory_id}\n"
        f"  Query: {pair.query}\n"
        f"  Plan: {pair.plan}\n"
    )


def format_level2_pot(agent: AgentRecord) -> str:
    return agent.system_prompt.pot_prompt.content.strip()


def format_level2_memory(agent: AgentRecord) -> str:
    pairs = agent.retrieved_memory.retrieved_pairs
    if not pairs:
        return ""
    parts = ["\n=== Retrieved Memory ===\n"]
    for pair in pairs:
        parts.append(_format_memory_pair(pair))
    return "\n".join(parts).rstrip() + "\n"


def format_level2_observations(agent: AgentRecord) -> str:
    if not agent.execution_history:
        return ""
    parts = ["\n=== Execution Observations ===\n"]
    for step in agent.execution_history:
        parts.append(
            f"Step {step.step}: {step.tool_called}\n"
            f"Observation: {step.observation}\n"
        )
    return "\n".join(parts).rstrip() + "\n"


def build_level2_base_input(
    log: ExecutionLog,
    agent: AgentRecord,
    prompt_to_act: str,
    input_override: Optional[str] = None,
) -> str:
    parts = [agent.system_prompt.basic.content]

    if input_override is not None:
        if input_override:
            parts.append(f"\n=== Causal Input ===\n{input_override}\n")
    else:
        for edge_id in agent.input_edges:
            edge = log.edge_map.get(edge_id)
            if edge is None:
                logger.warning(
                    "Agent %d references missing input edge %s in log %s",
                    agent.agent_id, edge_id, log.metadata.run_id,
                )
                continue
            parts.append(_format_edge(edge))

    parts.append(f"\n{prompt_to_act}")
    return "".join(parts)


def build_level2_module_input(
    log: ExecutionLog,
    agent: AgentRecord,
    module: str,
    prompt_to_act: str,
    input_override: Optional[str] = None,
) -> str:
    parts = [agent.system_prompt.basic.content]

    if module == "POT":
        pot_text = format_level2_pot(agent)
        if pot_text:
            parts.append("\n" + pot_text)

    if module == "MEM":
        parts.append(format_level2_memory(agent))

    if input_override is not None:
        if input_override:
            parts.append(f"\n=== Causal Input ===\n{input_override}\n")
    else:
        for edge_id in agent.input_edges:
            edge = log.edge_map.get(edge_id)
            if edge is not None:
                parts.append(_format_edge(edge))

    if module == "OBS":
        parts.append(format_level2_observations(agent))

    parts.append(f"\n{prompt_to_act}")
    return "".join(parts)


def build_level3_base_input(
    log: ExecutionLog,
    agent: AgentRecord,
    predicted_module: str,
    prompt_to_act: str,
    input_override: Optional[str] = None,
) -> str:
    """
    Build C_clean for Level 3: basic prompt + all non-winning modules + input edges.

    The *predicted_module* (m*) is excluded so that each element can be scored
    in isolation against this clean baseline.

    Module order: basic → POT → MEM → INP(edges) → OBS → prompt_to_act
    """
    parts = [agent.system_prompt.basic.content]

    if predicted_module != "POT":
        pot_text = format_level2_pot(agent)
        if pot_text:
            parts.append("\n" + pot_text)

    if predicted_module != "MEM":
        mem_text = format_level2_memory(agent)
        if mem_text.strip():
            parts.append(mem_text)

    if input_override is not None:
        if input_override:
            parts.append(f"\n=== Causal Input ===\n{input_override}\n")
    else:
        for edge_id in agent.input_edges:
            edge = log.edge_map.get(edge_id)
            if edge is not None:
                parts.append(_format_edge(edge))

    if predicted_module != "OBS":
        obs_text = format_level2_observations(agent)
        if obs_text.strip():
            parts.append(obs_text)

    parts.append(f"\n{prompt_to_act}")
    return "".join(parts)


def build_level3_element_input(
    log: ExecutionLog,
    agent: AgentRecord,
    predicted_module: str,
    element_text: str,
    prompt_to_act: str,
    input_override: Optional[str] = None,
) -> str:
    """
    Build context for Level 3 element scoring: C_clean with *element_text*
    inserted at the correct position for *predicted_module*.

    Module order: basic → POT → MEM → INP(edges) → OBS → prompt_to_act

    For POT elements *element_text* is a single serialized example (no header);
    a framing header is prepended here.  For MEM and OBS the caller supplies
    a pre-formatted block including its section header.
    """
    parts = [agent.system_prompt.basic.content]

    if predicted_module == "POT":
        if element_text:
            parts.append(f"\nHere are some examples of how to create plans:\n{element_text}\n")
    else:
        pot_text = format_level2_pot(agent)
        if pot_text:
            parts.append("\n" + pot_text)

    if predicted_module == "MEM":
        if element_text:
            parts.append(f"\n=== Retrieved Memory ===\n{element_text}\n")
    else:
        mem_text = format_level2_memory(agent)
        if mem_text.strip():
            parts.append(mem_text)

    if input_override is not None:
        if input_override:
            parts.append(f"\n=== Causal Input ===\n{input_override}\n")
    else:
        for edge_id in agent.input_edges:
            edge = log.edge_map.get(edge_id)
            if edge is not None:
                parts.append(_format_edge(edge))

    if predicted_module == "OBS":
        if element_text:
            parts.append(element_text)
    else:
        obs_text = format_level2_observations(agent)
        if obs_text.strip():
            parts.append(obs_text)

    parts.append(f"\n{prompt_to_act}")
    return "".join(parts)


def build_level1_minimal_input(
    system_prompt: str,
    edge: Edge,
    prompt_to_act: str,
) -> str:
    """
    Build a minimal single-edge context for Level 1 scoring.

    Contains only:
    - the invoking agent's basic system prompt
    - the content of the single edge being scored
    - the role-specific prompt_to_act
    No preceding edges are concatenated.
    """
    return "".join([system_prompt, _format_edge(edge), f"\n{prompt_to_act}"])


def build_level1_baseline_input(
    system_prompt: str,
    user_query: str,
    prompt_to_act: str,
) -> str:
    """Build the Level 1 baseline context: system prompt and the original user query only."""
    parts = [system_prompt]
    if user_query:
        parts.append(f"\n=== User Query ===\n{user_query}\n")
    parts.append(f"\n{prompt_to_act}")
    return "".join(parts)
