"""
Target token extraction: identify the τ-call string from the log and tokenize it.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Tuple

from data.schema import AgentRecord, ExecutionLog

logger = logging.getLogger(__name__)


def _find_invoking_agent(log: ExecutionLog) -> AgentRecord:
    """Return the agent that invoked τ, using ground_truth or execution_history scan."""
    gt = log.ground_truth
    if gt.invoked_by_agent_id is not None:
        return log.get_agent(gt.invoked_by_agent_id)

    # Fallback: scan all agents' execution_history
    tau = gt.malicious_tool_name
    for agent in log.agents:
        for step in agent.execution_history:
            if step.tool_called == tau:
                return agent

    # Final fallback: agent that has the tool in plan
    for agent in log.agents:
        if agent.plan:
            for step in agent.plan.steps:
                if step.tool == tau:
                    return agent

    raise ValueError(f"Cannot find agent that invoked τ='{tau}' in log {log.metadata.run_id}")


def _extract_tau_call_string_from_agent(agent: AgentRecord, tau: str) -> Optional[str]:
    """
    Extract the full τ-call JSON string from this agent's execution_history.

    execution_history is the authoritative record of actual tool calls: any agent
    that invoked τ must have a corresponding entry with tool_called == τ.
    Returns None only if τ is not present (attack did not succeed via this agent,
    or the log predates the unified execution_history schema).
    """
    # Primary source: execution_history (canonical, role-agnostic).
    for step in agent.execution_history:
        if step.tool_called == tau:
            step_dict = {
                "step": step.step,
                "action": step.plan_step_action,
                "tool": step.tool_called,
            }
            return json.dumps(step_dict, ensure_ascii=False)

    # Fallback: structured plan field (planner role, logs without
    # execution_history on every agent).
    if agent.plan:
        for ps in agent.plan.steps:
            if ps.tool == tau:
                step_dict = {"step": ps.step, "action": ps.action, "tool": ps.tool}
                return json.dumps(step_dict, ensure_ascii=False)

    # Fallback: raw_model_output JSON (formats that embed tool names).
    try:
        steps = json.loads(agent.raw_model_output)
        if isinstance(steps, list):
            for step in steps:
                if isinstance(step, dict):
                    if step.get("tool") == tau or step.get("tool_called") == tau:
                        return json.dumps(step, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        pass

    return None


def _extract_tau_call_string(agent: AgentRecord, tau: str) -> str:
    """Extract the full τ-call JSON string (step fragment containing τ) from the invoking agent."""
    result = _extract_tau_call_string_from_agent(agent, tau)
    if result is not None:
        return result

    logger.warning(
        "Could not extract full τ-call JSON for τ='%s' from agent %d. "
        "Falling back to the tool name.",
        tau, agent.agent_id,
    )
    return tau


def extract_target_tokens(log: ExecutionLog, tokenizer) -> List[int]:
    """
    Extract and tokenize the target token sequence representing the τ call.

    Args:
        log: Parsed ExecutionLog.
        tokenizer: HuggingFace tokenizer.

    Returns:
        List of integer token IDs for the τ-call string.
    """
    tau = log.ground_truth.malicious_tool_name
    if not tau:
        raise ValueError("ground_truth.malicious_tool_name is missing.")

    invoking_agent = _find_invoking_agent(log)
    tau_str = _extract_tau_call_string(invoking_agent, tau)

    # Tokenize WITHOUT special tokens — the target string is a continuation,
    # not a standalone sentence. BOS is only at the very start of the full input.
    token_ids = tokenizer.encode(tau_str, add_special_tokens=False)
    return token_ids


def count_format_prefix_tokens(tau_str: str, tokenizer) -> int:
    """Number of leading tokens of the τ-call string that precede the action text.

    The prefix ``{"step": N, "action": "`` is pure JSON format; it is predicted far
    better after a JSON plan than after plain text regardless of any injection, so
    Level 1 can exclude it from the score. Returns 0 if the marker is absent
    (e.g. when only the tool name was available).
    """
    marker = '"action": "'
    idx = tau_str.find(marker)
    if idx < 0:
        return 0
    boundary = idx + len(marker)
    ids = tokenizer.encode(tau_str, add_special_tokens=False)
    return sum(1 for i in range(len(ids)) if len(tokenizer.decode(ids[:i])) < boundary)


def tokenize_target_string(text: str, tokenizer) -> List[int]:
    """Tokenize an arbitrary continuation string without adding BOS/EOS."""
    return tokenizer.encode(text, add_special_tokens=False)


def get_tau_string(log: ExecutionLog) -> str:
    """Return the τ-call string (without tokenizing)."""
    tau = log.ground_truth.malicious_tool_name
    if not tau:
        raise ValueError("ground_truth.malicious_tool_name is missing.")
    invoking_agent = _find_invoking_agent(log)
    return _extract_tau_call_string(invoking_agent, tau)


def get_level2_target_string(
    log: ExecutionLog,
    target_agent_id: int,
    winning_edge_id: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Return the Level-2 target string τ' and its source label.

    If the target agent is the actual invoker of the malicious tool, τ' is the
    same tool-call string used by Level 1. Otherwise τ' is selected with the
    priority:
    winning edge content -> output edge -> final_response -> raw_model_output.
    """
    gt = log.ground_truth
    if gt.invoked_by_agent_id == target_agent_id:
        return get_tau_string(log), "tau_call"

    agent = log.get_agent(target_agent_id)
    if winning_edge_id:
        edge = log.edge_map.get(winning_edge_id)
        if (
            edge is not None
            and edge.content.strip()
            and edge.from_node == f"agent_{target_agent_id}"
        ):
            return edge.content, "winning_edge"

    for output_edge_id in agent.output_edges or ([agent.output_edge] if agent.output_edge else []):
        edge = log.edge_map.get(output_edge_id)
        if edge is not None and edge.content.strip():
            return edge.content, "output_edge"

    if agent.final_response and agent.final_response.strip():
        return agent.final_response, "final_response"

    if agent.raw_model_output and agent.raw_model_output.strip():
        return agent.raw_model_output, "raw_model_output"

    raise ValueError(
        f"Agent {target_agent_id} in log {log.metadata.run_id} has no usable Level-2 target output."
    )
