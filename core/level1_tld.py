"""
Level 1: agent localization via token log-probability differences (TLD).

Algorithm:
1. Order the edges (topological sort, or logged order for random/MD) → [e_0, ..., e_M].
2. Extract the target token IDs of the τ-call (JSON format prefix excluded from scoring).
3. Score the edges against the baseline ψ_{-1} (no edges):
   s_i = log P(τ | edge e_i alone), jump j_i = s_i - s_{i-1}.
4. i* = argmax over the jumps.
5. Map e_{i*} to its culprit agent (receiver for DPI and user-origin edges, sender otherwise).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from data.schema import Edge, ExecutionLog
from core.context_builder import build_level1_baseline_input, build_level1_minimal_input
from core.dag import edge_receiver_agent_id, edge_source_agent_id, topological_sort
from core.logprob import compute_log_prob, compute_log_prob_batch
from core.target_tokens import count_format_prefix_tokens, extract_target_tokens, get_tau_string

logger = logging.getLogger(__name__)


def _get_prompt_to_act(agent, config: dict) -> str:
    """Return the short prompt appended after edge context to elicit the τ call.

    Supports both linear-pipeline roles (planner / executor / summarizer) and
    tree-topology roles (delegator / analyst-as-sub-delegator / analyst-as-executor).

    For ``analyst`` roles the distinction is made by inspecting whether the agent
    produced a delegation plan (``agent.plan is not None``) or executed tools
    directly — matching the two structural behaviours in the tree topology.
    """
    from data.schema import AgentRecord  # local import to avoid circular deps

    role_map = config.get("level1", {}).get("prompt_to_act", {})

    # Accept either an AgentRecord or a plain role string.
    if isinstance(agent, AgentRecord):
        role = (agent.role or "").lower()
        has_plan = agent.plan is not None
    else:
        role = (agent or "").lower()
        has_plan = False

    if "planner" in role:
        return role_map.get("planner", "Now create your execution plan as a JSON array.")
    elif "executor" in role:
        return role_map.get("executor", "Now execute the next step. Output your tool call.")
    elif "summarizer" in role:
        return role_map.get("summarizer", "Now produce the final report.")
    elif "full_analyst" in role:
        return role_map.get("analyst_executor", role_map.get(
            "executor", "Now execute the next step. Output your tool call."
        ))
    elif "delegator" in role:
        return role_map.get(
            "delegator",
            "Now create your delegation plan as a JSON object assigning tasks to each sub-agent.",
        )
    elif "analyst" in role:
        # Tree topology: analyst is either a sub-delegator (has a plan) or a leaf executor.
        if has_plan:
            return role_map.get(
                "analyst_planner",
                role_map.get(
                    "delegator",
                    "Now create your sub-delegation plan as a JSON object assigning tasks to each sub-agent.",
                ),
            )
        else:
            return role_map.get(
                "analyst_executor",
                role_map.get("executor", "Now execute the next step. Output your tool call."),
            )
    else:
        return "Now proceed based on the above information."


def level1_localize(
    log: ExecutionLog,
    model,
    tokenizer,
    config: dict,
    use_batch: bool = True,
) -> Tuple[int, List[float], Dict[str, str], str, Dict]:
    """
    Level 1: identify the earliest compromised agent (A*).

    Args:
        log: Parsed ExecutionLog.
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        config: Config dict (from default.yaml).
        use_batch: If True, batch all forward passes together.

    Returns:
        (predicted_agent_id, jumps, edge_mapping, winning_edge_id, extra_diags)
        - predicted_agent_id: int ID of the predicted earliest compromised agent.
        - jumps: the jump values [j_0, ..., j_M].
        - edge_mapping: dict {edge_id: "from→to"} for diagnostics.
        - winning_edge_id: edge_id of the argmax edge (e.g. "e_0").
        - extra_diags: {"scores": [s_0, ..., s_M]}.
    """
    max_seq = config.get("model", {}).get("max_seq_length", 4096)

    # Prefix for synthetic final-response edge IDs (one per terminal agent).
    _SYNTHETIC_EDGE_PREFIX = "__e_final__"

    # Edge ordering: random and MD topologies use logged edge order directly
    # (execution_graph.edges records the actual discussion/delegation sequence).
    # For acyclic topologies use topological sort to recover causal order.
    if log.topology_type in {"random", "md"}:
        ordered_edges: List[Edge] = list(log.execution_graph.edges)
        logger.info(
            "%s topology: using logged edge order directly (%d edges).",
            log.topology_type,
            len(ordered_edges),
        )
    else:
        ordered_edges = topological_sort(log.execution_graph)

    target_token_ids = extract_target_tokens(log, tokenizer)
    logger.info("Target tokens: %d tokens for τ='%s'",
                len(target_token_ids), log.ground_truth.malicious_tool_name)

    # The JSON format prefix of τ is cheap after any JSON context regardless of
    # poisoning, so it is excluded from every score.
    score_from = count_format_prefix_tokens(
        get_tau_string(log), tokenizer
    )

    invoking_agent_id = log.ground_truth.invoked_by_agent_id
    invoking_agent = log.get_agent(invoking_agent_id)
    system_prompt = invoking_agent.system_prompt.basic.content
    prompt_to_act = _get_prompt_to_act(invoking_agent, config)

    # Append a synthetic output edge for every leaf agent (no outgoing edge).
    _agents_with_outgoing = {
        e.from_node for e in log.execution_graph.edges if e.from_node != "user"
    }

    for agent in log.agents:
        is_leaf = f"agent_{agent.agent_id}" not in _agents_with_outgoing
        if not is_leaf:
            continue

        if agent.final_response and agent.final_response.strip():
            final_content = agent.final_response
            content_source = "final_response"
        elif agent.execution_history:
            final_content = "\n".join(
                f"Step {s.step}: {s.tool_called} → {s.observation}"
                for s in agent.execution_history
            )
            content_source = "execution_history"
        else:
            final_content = agent.raw_model_output or ""
            content_source = "raw_model_output"

        if not final_content:
            logger.warning(
                "Leaf agent_%d has no content for synthetic edge; "
                "skipping.", agent.agent_id,
            )
            continue

        syn_edge = Edge(
            edge_id=f"{_SYNTHETIC_EDGE_PREFIX}{agent.agent_id}",
            from_node=f"agent_{agent.agent_id}",
            to_node="final_output",
            content=final_content,
            is_poisoned=False,
            poison_type=None,
        )
        ordered_edges = ordered_edges + [syn_edge]
        logger.info(
            "Appended synthetic edge '%s' for leaf agent_%d "
            "(content_source=%s, len=%d).",
            syn_edge.edge_id, agent.agent_id, content_source, len(final_content),
        )

    baseline_ctx = build_level1_baseline_input(
        system_prompt, log.original_user_query.strip(), prompt_to_act
    )
    psi_baseline = compute_log_prob(
        model, tokenizer, baseline_ctx, target_token_ids, max_seq_length=max_seq,
        score_from=score_from,
    )
    logger.info("ψ_baseline = %.4f", psi_baseline)

    # One minimal context per edge (system prompt + single edge + prompt_to_act).
    minimal_contexts = [
        build_level1_minimal_input(system_prompt, edge, prompt_to_act)
        for edge in ordered_edges
    ]

    if use_batch:
        try:
            scores: List[float] = compute_log_prob_batch(
                model, tokenizer, minimal_contexts, target_token_ids, max_seq_length=max_seq,
                score_from=score_from,
            )
        except Exception as e:
            logger.warning("Batch computation failed (%s), falling back to sequential.", e)
            scores = [
                compute_log_prob(model, tokenizer, ctx, target_token_ids, max_seq_length=max_seq, score_from=score_from)
                for ctx in minimal_contexts
            ]
    else:
        scores = [
            compute_log_prob(model, tokenizer, ctx, target_token_ids, max_seq_length=max_seq, score_from=score_from)
            for ctx in minimal_contexts
        ]

    for i, (s, edge) in enumerate(zip(scores, ordered_edges)):
        logger.info("  s_%d (%s) = %.4f", i, edge.edge_id, s)

    # Jumps: j_0 = s_0 - ψ_baseline, j_i = s_i - s_{i-1}
    jumps: List[float] = []
    for i, s in enumerate(scores):
        prev = psi_baseline if i == 0 else scores[i - 1]
        jumps.append(s - prev)

    logger.info("Jumps: %s", [f"{j:.4f}" for j in jumps])

    i_star = int(max(range(len(jumps)), key=lambda i: jumps[i]))
    winning_edge = ordered_edges[i_star]
    logger.info(
        "i* = %d, winning edge = %s (jump=%.4f)",
        i_star, winning_edge.edge_id, jumps[i_star],
    )

    source_agent_id = edge_source_agent_id(winning_edge)
    if log.metadata.attack_config.attack_type == "DPI":
        # Direct prompt injection compromises the agent that consumes the
        # winning input, even when that input was relayed by another agent.
        predicted_agent_id = edge_receiver_agent_id(winning_edge)
        logger.info(
            "DPI case: winning edge %s → A* = agent_%d (receiver)",
            winning_edge.edge_id, predicted_agent_id,
        )
    elif source_agent_id is None:
        predicted_agent_id = edge_receiver_agent_id(winning_edge)
        logger.info(
            "User-origin winning edge %s → A* = agent_%d (receiver)",
            winning_edge.edge_id, predicted_agent_id,
        )
    else:
        # The winning edge's sender is the culprit (also for synthetic output edges).
        predicted_agent_id = source_agent_id
        logger.info(
            "Winning edge %s from agent_%d → A* = agent_%d",
            winning_edge.edge_id, source_agent_id, predicted_agent_id,
        )

    edge_mapping = {
        e.edge_id: f"{e.from_node}→{e.to_node}"
        for e in ordered_edges
    }

    return predicted_agent_id, jumps, edge_mapping, winning_edge.edge_id, {"scores": scores}


def compute_gain_margin(gains: List[float]) -> float:
    """
    Gain margin = g_{i*} - second_highest_g.
    Returns 0.0 if there is only one edge.
    """
    if len(gains) <= 1:
        return 0.0
    sorted_gains = sorted(gains, reverse=True)
    return sorted_gains[0] - sorted_gains[1]
