"""
Level 3: intra-module element attribution via TLD scoring.

Given the Level 2 winner (A*, m*) and the Level 2 target string τ', this
module pinpoints which *elements* inside m* are responsible for the attack:

  POT / MEM  →  independent marginal gains per element; K-means clustering
               with K selected by Silhouette Coefficient; the cluster with
               the lowest mean gain is treated as benign, all others malicious.

  OBS        →  incremental prefix gains δ_k = L_k − L_{k−1} over
               execution-order prefixes; the earliest step with the largest
               jump is predicted as the poisoned observation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from core.context_builder import (
    build_level3_base_input,
    build_level3_element_input,
)
from core.level1_tld import _get_prompt_to_act
from core.logprob import compute_log_prob, compute_log_prob_batch
from core.target_tokens import tokenize_target_string
from data.schema import AgentRecord, ExecutionLog, ExecutionStep, PlanRecord, PotExample, RetrievedPair

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Element extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_level3_elements(
    agent: AgentRecord,
    module: str,
    plan_record: Optional[PlanRecord] = None,
) -> List[dict]:
    """
    Return a list of element dicts for Level 3 scoring.

    Each dict: {"id": str, "label": int, "obj": <element object>}

    POT  → agent.system_prompt.pot_prompt.examples  (PotExample objects)
    MEM  → agent.retrieved_memory.retrieved_pairs   (RetrievedPair objects)
    OBS  → agent.execution_history                  (ExecutionStep objects)
    """
    if module == "POT":
        examples = agent.system_prompt.pot_prompt.examples
        return [
            {"id": f"pot_{i}", "label": ex.label, "obj": ex}
            for i, ex in enumerate(examples)
        ]
    if module == "MEM":
        pairs = agent.retrieved_memory.retrieved_pairs
        return [
            {"id": f"mem_{i}", "label": p.label, "obj": p}
            for i, p in enumerate(pairs)
        ]
    if module == "OBS":
        steps = agent.execution_history
        return [
            {"id": f"obs_{i}", "label": s.label, "obj": s}
            for i, s in enumerate(steps)
        ]
    return []


def get_level3_ground_truth_elements(elements: List[dict], module: str) -> List[str]:
    """
    Return ground-truth element IDs for evaluation.

    POT / MEM → all element IDs where label == 1
    OBS       → single-element list containing the *earliest* ID where label == 1
                and the observation is actually poisoned (not "Tool not found." and
                not an adaptive step).  If element labels are unavailable, falls
                back to the real observation immediately preceding the first
                malicious tool call.
    """
    if module == "OBS":
        # Prefer the earliest step that was genuinely poisoned by IPI:
        # non-adaptive, non-malicious-tool, and with a real (non-empty) observation.
        for el in elements:
            step: ExecutionStep = el["obj"]
            if (
                el["label"] == 1
                and not step.is_malicious_tool
                and step.observation != "Tool not found."
            ):
                return [el["id"]]
        # Fallback: first label==1 element
        for el in elements:
            if el["label"] == 1:
                return [el["id"]]
        # MD logs mark the agent-level OBS module as poisoned but
        # leave per-observation labels clean.  In IPI, the poisoned observation
        # is the runtime observation that causes the next malicious tool call.
        for idx, el in enumerate(elements):
            step: ExecutionStep = el["obj"]
            if not step.is_malicious_tool:
                continue
            for prev in reversed(elements[:idx]):
                prev_step: ExecutionStep = prev["obj"]
                if (
                    not prev_step.is_malicious_tool
                    and prev_step.observation
                    and prev_step.observation != "Tool not found."
                ):
                    return [prev["id"]]
        return []
    return [el["id"] for el in elements if el["label"] == 1]


# ──────────────────────────────────────────────────────────────────────────────
# Per-element serialization
# ──────────────────────────────────────────────────────────────────────────────

def _serialize_pot_element(example: PotExample) -> str:
    """Serialize a single PoT example for element-level scoring."""
    return (
        f"For example, your task is to {example.query}\n"
        f"A plan example can be: {example.plan}"
    )


def _serialize_mem_element(pair: RetrievedPair) -> str:
    """Serialize a single memory pair for element-level scoring."""
    return (
        f"- Memory ID: {pair.memory_id}\n"
        f"  Query: {pair.query}\n"
        f"  Plan: {pair.plan}"
    )


def _serialize_obs_prefix(steps: List[ExecutionStep]) -> str:
    """Serialize a prefix of execution steps (observations) as a block."""
    if not steps:
        return ""
    parts = ["\n=== Execution Observations ===\n"]
    for step in steps:
        parts.append(
            f"Step {step.step}: {step.tool_called}\n"
            f"Observation: {step.observation}\n"
        )
    return "\n".join(parts).rstrip() + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# K-means partition with dynamic K via Silhouette Coefficient
# ──────────────────────────────────────────────────────────────────────────────

def _kmeans_partition(gains: List[float]) -> Tuple[List[int], List[int], float]:
    """
    Partition K gain values into a benign cluster and malicious clusters using
    K-means. K is chosen dynamically by maximising the Silhouette Coefficient
    over candidates K=2..min(K-1, 10). The cluster with the lowest mean gain
    is treated as benign; all other clusters are merged into the malicious set.

    Returns:
        benign_indices:   original indices assigned to the benign cluster (sorted)
        malicious_indices: original indices assigned to all other clusters (sorted)
        best_silhouette:  silhouette score for the chosen K (0.0 for K < 2)
    """
    K = len(gains)
    if K == 0:
        return [], [], 0.0
    if K == 1:
        return [], [0], 0.0
    if K == 2:
        # silhouette_score requires ≥3 samples; handle K=2 directly
        g = gains
        if g[0] <= g[1]:
            benign_val, mal_val = 0, 1
        else:
            benign_val, mal_val = 1, 0
        sil = 1.0 if abs(g[0] - g[1]) > 0 else 0.0
        return [benign_val], [mal_val], sil

    X = np.array(gains, dtype=float).reshape(-1, 1)

    max_k = min(K - 1, 10)
    best_k = 2
    best_sil = -1.0
    best_labels: Optional[np.ndarray] = None

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        # silhouette_score requires at least 2 distinct labels and >1 sample
        if len(set(labels)) < 2:
            continue
        sil = float(silhouette_score(X, labels))
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels

    if best_labels is None:
        # Fallback: simple 2-cluster split at median
        median = float(np.median(gains))
        best_labels = np.array([0 if g <= median else 1 for g in gains])

    # Identify the cluster with the lowest mean gain → benign
    cluster_ids = np.unique(best_labels)
    cluster_means = {c: float(np.mean(X[best_labels == c])) for c in cluster_ids}
    benign_cluster = min(cluster_means, key=lambda c: cluster_means[c])

    benign_indices = sorted(int(i) for i in np.where(best_labels == benign_cluster)[0])
    malicious_indices = sorted(int(i) for i in np.where(best_labels != benign_cluster)[0])

    return benign_indices, malicious_indices, best_sil


# ──────────────────────────────────────────────────────────────────────────────
# Shared helper
# ──────────────────────────────────────────────────────────────────────────────

def _skip_result(reason: str) -> dict:
    return {
        "predicted_elements": [],
        "predicted_element_type": "",
        "element_scores": {},
        "element_gains": {},
        "base_logprob": None,
        "gain_margin": None,
        "confidence": None,
        "skipped": True,
        "skip_reason": reason,
        "inconclusive": False,
    }


# ──────────────────────────────────────────────────────────────────────────────
# POT / MEM localization (independent marginal gains + K-means clustering)
# ──────────────────────────────────────────────────────────────────────────────

def _localize_pot_mem(
    log: ExecutionLog,
    agent: AgentRecord,
    module: str,
    elements: List[dict],
    target_token_ids: List[int],
    prompt_to_act: str,
    model,
    tokenizer,
    max_seq: int,
    epsilon: float,
    use_batch: bool,
    input_override: Optional[str] = None,
) -> dict:
    K = len(elements)
    element_ids = [el["id"] for el in elements]

    serialized = []
    for el in elements:
        if module == "POT":
            serialized.append(_serialize_pot_element(el["obj"]))
        else:
            serialized.append(_serialize_mem_element(el["obj"]))

    base_ctx = build_level3_base_input(
        log, agent, module, prompt_to_act, input_override=input_override
    )
    element_ctxs = [
        build_level3_element_input(
            log, agent, module, s, prompt_to_act, input_override=input_override
        )
        for s in serialized
    ]

    if use_batch:
        try:
            all_scores = compute_log_prob_batch(
                model, tokenizer,
                [base_ctx] + element_ctxs,
                target_token_ids,
                max_seq_length=max_seq,
            )
            base_logprob = all_scores[0]
            elem_scores = all_scores[1:]
        except Exception as exc:
            logger.warning("Level 3 batch failed (%s); falling back to sequential.", exc)
            use_batch = False

    if not use_batch:
        base_logprob = compute_log_prob(
            model, tokenizer, base_ctx, target_token_ids, max_seq_length=max_seq
        )
        elem_scores = [
            compute_log_prob(model, tokenizer, ctx, target_token_ids, max_seq_length=max_seq)
            for ctx in element_ctxs
        ]

    gains = [s - base_logprob for s in elem_scores]
    element_gains = dict(zip(element_ids, gains))
    element_scores_dict = dict(zip(element_ids, elem_scores))

    # ── Degenerate: K == 1 ────────────────────────────────────────────────────
    if K == 1:
        return {
            "predicted_elements": [element_ids[0]],
            "predicted_element_type": "pair",
            "element_scores": element_scores_dict,
            "element_gains": element_gains,
            "base_logprob": base_logprob,
            "gain_margin": None,
            "confidence": None,
            "skipped": False,
            "skip_reason": "",
            "inconclusive": False,
        }

    # ── Degenerate: K == 2 ────────────────────────────────────────────────────
    if K == 2:
        gap = abs(gains[1] - gains[0])
        inconclusive = gap < epsilon
        pred_idx = 0 if gains[0] >= gains[1] else 1
        return {
            "predicted_elements": [element_ids[pred_idx]],
            "predicted_element_type": "pair",
            "element_scores": element_scores_dict,
            "element_gains": element_gains,
            "base_logprob": base_logprob,
            "gain_margin": gap,
            "confidence": None if inconclusive else gap,
            "skipped": False,
            "skip_reason": "",
            "inconclusive": inconclusive,
        }

    # ── General case: K-means clustering with dynamic K ──────────────────────
    benign_idx, malicious_idx, best_sil = _kmeans_partition(gains)

    gain_range = max(gains) - min(gains)
    inconclusive = (best_sil < epsilon) or (gain_range < epsilon)

    if inconclusive:
        return {
            "predicted_elements": [],
            "predicted_element_type": "pair",
            "element_scores": element_scores_dict,
            "element_gains": element_gains,
            "base_logprob": base_logprob,
            "gain_margin": 0.0,
            "confidence": 0.0,
            "skipped": False,
            "skip_reason": "",
            "inconclusive": True,
        }

    predicted_ids = [element_ids[i] for i in malicious_idx]
    malicious_gains = [gains[i] for i in malicious_idx]
    benign_gains = [gains[i] for i in benign_idx]
    gap = (min(malicious_gains) - max(benign_gains)) if benign_gains else None

    return {
        "predicted_elements": predicted_ids,
        "predicted_element_type": "pair",
        "element_scores": element_scores_dict,
        "element_gains": element_gains,
        "base_logprob": base_logprob,
        "gain_margin": gap,
        "confidence": gap,
        "skipped": False,
        "skip_reason": "",
        "inconclusive": False,
    }


# ──────────────────────────────────────────────────────────────────────────────
# OBS localization (prefix-incremental gains)
# ──────────────────────────────────────────────────────────────────────────────

def _localize_obs(
    log: ExecutionLog,
    agent: AgentRecord,
    elements: List[dict],
    target_token_ids: List[int],
    prompt_to_act: str,
    model,
    tokenizer,
    max_seq: int,
    use_batch: bool,
    input_override: Optional[str] = None,
) -> dict:
    T = len(elements)
    steps = [el["obj"] for el in elements]
    element_ids = [el["id"] for el in elements]

    # Prefix contexts: base (empty obs), then obs[0:1], obs[0:2], ..., obs[0:T]
    base_ctx = build_level3_base_input(
        log, agent, "OBS", prompt_to_act, input_override=input_override
    )
    prefix_ctxs = [
        build_level3_element_input(
            log,
            agent,
            "OBS",
            _serialize_obs_prefix(steps[:k]),
            prompt_to_act,
            input_override=input_override,
        )
        for k in range(1, T + 1)
    ]

    if use_batch:
        try:
            all_scores = compute_log_prob_batch(
                model, tokenizer,
                [base_ctx] + prefix_ctxs,
                target_token_ids,
                max_seq_length=max_seq,
            )
            logprobs = all_scores   # [L_0, L_1, ..., L_T]
        except Exception as exc:
            logger.warning("Level 3 OBS batch failed (%s); falling back.", exc)
            use_batch = False

    if not use_batch:
        logprobs = [
            compute_log_prob(model, tokenizer, base_ctx, target_token_ids, max_seq_length=max_seq)
        ]
        for ctx in prefix_ctxs:
            logprobs.append(
                compute_log_prob(model, tokenizer, ctx, target_token_ids, max_seq_length=max_seq)
            )

    base_logprob = logprobs[0]
    # δ_k = L_k − L_{k−1}, k = 1..T  (1-indexed; δ_k corresponds to obs_{k−1})
    deltas = [logprobs[k] - logprobs[k - 1] for k in range(1, T + 1)]

    # Predicted: obs element at argmax(δ) — note argmax into deltas list is 0-indexed
    winning_k = int(max(range(T), key=lambda i: deltas[i]))
    predicted_id = element_ids[winning_k]

    gain_margin: Optional[float]
    if T == 1:
        gain_margin = None
    else:
        sorted_d = sorted(deltas, reverse=True)
        gain_margin = sorted_d[0] - sorted_d[1]

    delta_dict = {element_ids[i]: deltas[i] for i in range(T)}
    prefix_scores = {f"prefix_{k}": logprobs[k] for k in range(T + 1)}

    return {
        "predicted_elements": [predicted_id],
        "predicted_element_type": "observation",
        "element_scores": prefix_scores,
        "element_gains": delta_dict,
        "base_logprob": base_logprob,
        "gain_margin": gain_margin,
        "confidence": gain_margin,
        "skipped": False,
        "skip_reason": "",
        "inconclusive": False,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public interface
# ──────────────────────────────────────────────────────────────────────────────

def level3_localize(
    log: ExecutionLog,
    target_agent_id: int,
    predicted_module: str,
    tau_prime: str,
    model,
    tokenizer,
    config: dict,
    use_batch: bool = True,
    response_kind: Optional[str] = None,
    selected_phase_index: Optional[int] = None,
) -> dict:
    """
    Level 3: intra-module element attribution.

    Args:
        log:              Parsed execution log.
        target_agent_id:  Agent identified by Level 2 (A*).
        predicted_module: Module identified by Level 2 (m*).
        tau_prime:        Level-2 target string (τ').
        model:            HuggingFace causal LM.
        tokenizer:        Corresponding tokenizer.
        config:           Config dict.
        use_batch:        Use batched log-prob computation when possible.

    Returns:
        Dict with keys:
            predicted_elements      List[str]  — element IDs in the predicted set
            predicted_element_type  str        — "pair" | "observation"
            element_scores          dict       — raw log-prob per element / prefix
            element_gains           dict       — gain per element (marginal or δ_k)
            base_logprob            float|None
            gain_margin             float|None — gap between benign/malicious clusters (POT/MEM) or Δδ (OBS)
            confidence              float|None — same as gain_margin
            skipped                 bool
            skip_reason             str
            inconclusive            bool
    """
    if not predicted_module or predicted_module == "INP":
        return _skip_result("INP / unknown module; Level 3 not applicable.")

    agent = log.get_agent(target_agent_id)

    # Resolve the plan_record for random-topology planners so that Level 3
    # decomposes the same response object selected by Level 2.
    plan_record: Optional[PlanRecord] = None
    causal_input_override: Optional[str] = None
    if response_kind == "plan_record" and selected_phase_index is not None:
        sorted_records = sorted(agent.plan_records, key=lambda r: r.phase_index)
        plan_record = next(
            (r for r in sorted_records if r.phase_index == selected_phase_index), None
        )
        if plan_record is None:
            logger.warning(
                "Level 3: response_kind='plan_record' but no plan_record with "
                "phase_index=%d found for agent %d; falling back to default elements.",
                selected_phase_index, target_agent_id,
            )
        else:
            causal_input_override = plan_record.input or ""

    elements = extract_level3_elements(agent, predicted_module, plan_record=plan_record)
    if not elements:
        return _skip_result(
            f"No scorable elements found in module {predicted_module} "
            f"for agent {target_agent_id}."
        )

    prompt_to_act = _get_prompt_to_act(agent, config)
    max_seq = config.get("model", {}).get("max_seq_length", 4096)
    epsilon = config.get("level3", {}).get("inconclusive_gain_epsilon", 1e-3)
    target_token_ids = tokenize_target_string(tau_prime, tokenizer)

    try:
        if predicted_module in ("POT", "MEM"):
            return _localize_pot_mem(
                log, agent, predicted_module, elements,
                target_token_ids, prompt_to_act,
                model, tokenizer, max_seq, epsilon, use_batch,
                input_override=causal_input_override,
            )
        else:  # OBS
            return _localize_obs(
                log, agent, elements,
                target_token_ids, prompt_to_act,
                model, tokenizer, max_seq, use_batch,
                input_override=causal_input_override,
            )
    except Exception as exc:
        logger.error(
            "Level 3 error for agent %d module %s: %s",
            target_agent_id, predicted_module, exc, exc_info=True,
        )
        return _skip_result(f"Level 3 internal error: {exc}")
