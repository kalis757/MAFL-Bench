"""
Level 2: module localization via independent TLD gains.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from core.context_builder import (
    build_level2_base_input,
    build_level2_module_input,
    format_level2_memory,
    format_level2_observations,
    format_level2_pot,
)
from core.level1_tld import _get_prompt_to_act
from core.logprob import compute_log_prob, compute_log_prob_batch
from core.target_tokens import get_level2_target_string, tokenize_target_string
from data.schema import AgentRecord, ExecutionLog, PlanRecord

logger = logging.getLogger(__name__)


def _find_plan_record_for_edge(
    log: ExecutionLog,
    agent: AgentRecord,
    winning_edge_id: str,
) -> Tuple[Optional[PlanRecord], Optional[int]]:
    """
    For a planner in random topology, find the plan_record corresponding to winning_edge_id.

    The k-th real planning edge from this agent (in execution_graph.edges order)
    maps one-to-one to the k-th plan_record sorted by phase_index.

    Returns (matched_plan_record, phase_index) or (None, None) if no match.
    """
    if not agent.plan_records:
        return None, None

    agent_node = f"agent_{agent.agent_id}"
    # Real planning edges produced by this agent, in logged order (no synthetic edges)
    real_edges = [
        e for e in log.execution_graph.edges
        if e.from_node == agent_node
    ]

    sorted_records = sorted(agent.plan_records, key=lambda r: r.phase_index)

    for k, edge in enumerate(real_edges):
        if edge.edge_id == winning_edge_id:
            if k < len(sorted_records):
                record = sorted_records[k]
                return record, record.phase_index
            break

    return None, None


def _module_has_content(agent: AgentRecord, module: str) -> bool:
    if module == "POT":
        return bool(format_level2_pot(agent))
    if module == "MEM":
        return bool(format_level2_memory(agent).strip())
    if module == "OBS":
        return bool(format_level2_observations(agent).strip())
    return False


def _candidate_modules(agent: AgentRecord) -> List[str]:
    candidates: List[str] = []
    for module in ("POT", "MEM", "OBS"):
        if agent.module_labels.get(module) == -1:
            continue
        if not _module_has_content(agent, module):
            continue
        candidates.append(module)
    return candidates


def _compute_margin(scores: Dict[str, float]) -> float:
    if len(scores) <= 1:
        return 0.0
    ordered = sorted(scores.values(), reverse=True)
    return ordered[0] - ordered[1]


def _pick_module(module_gains: Dict[str, float]) -> str:
    candidates = list(module_gains)
    if len(candidates) == 1:
        return candidates[0]
    if "OBS" not in module_gains:
        return max(module_gains, key=module_gains.get)
    pre_candidates = [m for m in ("POT", "MEM") if m in module_gains]
    if not pre_candidates:
        return "OBS"
    if len(pre_candidates) == 1:
        return max(module_gains, key=module_gains.get)

    m_pre = max(pre_candidates, key=lambda m: module_gains[m])
    gap_cross = abs(module_gains[m_pre] - module_gains["OBS"])
    gap_within = abs(module_gains["POT"] - module_gains["MEM"])
    return "OBS" if gap_cross > gap_within else m_pre


def level2_localize(
    log: ExecutionLog,
    target_agent_id: int,
    model,
    tokenizer,
    config: dict,
    winning_edge_id: str | None = None,
    use_batch: bool = True,
) -> dict:
    """
    Level 2: identify the poisoned module within the target agent.
    """
    attack_type = log.metadata.attack_config.attack_type
    if attack_type == "DPI":
        return {
            "predicted_module": "",
            "candidate_modules": [],
            "base_logprob": None,
            "module_gains": {},
            "gain_margin": None,
            "tau_prime": "",
            "tau_prime_source": "",
            "response_kind": "",
            "selected_phase_index": None,
            "skipped": True,
            "skip_reason": "DPI attack skips Level 2 by design.",
        }

    agent = log.get_agent(target_agent_id)
    candidates = _candidate_modules(agent)
    if not candidates:
        return {
            "predicted_module": "",
            "candidate_modules": [],
            "base_logprob": None,
            "module_gains": {},
            "gain_margin": None,
            "tau_prime": "",
            "tau_prime_source": "",
            "response_kind": "",
            "selected_phase_index": None,
            "skipped": True,
            "skip_reason": f"Agent {target_agent_id} has no Level-2 candidate modules.",
        }

    tau_prime, tau_prime_source = get_level2_target_string(
        log,
        target_agent_id,
        winning_edge_id=winning_edge_id,
    )

    # Determine response_kind and selected_phase_index for Level 3 consumption.
    # In random topology, when a planner's winning edge is matched to a plan_record,
    # expose which phase was selected so Level 3 can use the same response object.
    response_kind: str = tau_prime_source
    selected_phase_index: Optional[int] = None
    causal_input_override: Optional[str] = None
    if (
        log.topology_type == "random"
        and agent.plan_records
        and tau_prime_source == "winning_edge"
        and winning_edge_id is not None
    ):
        matched_record, phase_idx = _find_plan_record_for_edge(log, agent, winning_edge_id)
        if matched_record is not None:
            response_kind = "plan_record"
            selected_phase_index = phase_idx
            causal_input_override = matched_record.input or ""
            logger.info(
                "Random topology: matched plan_record phase_index=%d for agent %d "
                "(winning_edge=%s).",
                phase_idx, target_agent_id, winning_edge_id,
            )

    target_token_ids = tokenize_target_string(tau_prime, tokenizer)
    prompt_to_act = _get_prompt_to_act(agent, config)
    base_context = build_level2_base_input(
        log, agent, prompt_to_act, input_override=causal_input_override
    )
    module_contexts = {
        module: build_level2_module_input(
            log, agent, module, prompt_to_act, input_override=causal_input_override
        )
        for module in candidates
    }

    max_seq = config.get("model", {}).get("max_seq_length", 4096)
    if use_batch:
        try:
            all_contexts = [base_context] + [module_contexts[m] for m in candidates]
            all_scores = compute_log_prob_batch(
                model, tokenizer, all_contexts, target_token_ids, max_seq_length=max_seq
            )
            base_logprob = all_scores[0]
            module_scores = {
                module: score for module, score in zip(candidates, all_scores[1:])
            }
        except Exception as e:
            logger.warning("Level 2 batch computation failed (%s), falling back to sequential.", e)
            use_batch = False

    if not use_batch:
        base_logprob = compute_log_prob(
            model, tokenizer, base_context, target_token_ids, max_seq_length=max_seq
        )
        module_scores = {
            module: compute_log_prob(
                model, tokenizer, context, target_token_ids, max_seq_length=max_seq
            )
            for module, context in module_contexts.items()
        }

    module_gains = {
        module: score - base_logprob
        for module, score in module_scores.items()
    }
    predicted_module = _pick_module(module_gains)
    gain_margin = _compute_margin(module_gains)

    return {
        "predicted_module": predicted_module,
        "candidate_modules": candidates,
        "base_logprob": base_logprob,
        "module_gains": module_gains,
        "gain_margin": gain_margin,
        "tau_prime": tau_prime,
        "tau_prime_source": tau_prime_source,
        "response_kind": response_kind,
        "selected_phase_index": selected_phase_index,
        "skipped": False,
        "skip_reason": "",
    }
