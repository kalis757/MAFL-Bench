"""
Log-probability computation via teacher-forcing forward pass.

Core function: compute_log_prob(model, tokenizer, context_str, target_token_ids) -> float

Tokenization boundaries are handled as follows:
1. Tokenize full_string = context_str + target_str as a single unit.
2. Tokenize context_str alone.
3. target_positions = len(context_ids) ... len(full_ids)-1
4. Forward pass on full_ids; extract log-softmax at target positions.
"""

from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_log_prob(
    model,
    tokenizer,
    context_str: str,
    target_token_ids: List[int],
    max_seq_length: int = 4096,
    score_from: int = 0,
) -> float:
    """
    Compute the sum of log-probabilities of target_token_ids given context_str,
    using a single teacher-forcing forward pass (no generation).

    Args:
        model: HuggingFace causal LM (already on device).
        tokenizer: Corresponding tokenizer.
        context_str: The full context string (system prompt + accumulated edges, etc.).
        target_token_ids: Token IDs of the τ-call string (no BOS).
        max_seq_length: Max total tokens; context is truncated from the left if needed.
        score_from: Index of the first target token that is scored; earlier tokens are
            still teacher-forced as context but excluded from the sum.

    Returns:
        Sum of per-token log-probabilities (float, <= 0).
    """
    context_ids: List[int] = tokenizer.encode(context_str, add_special_tokens=True)

    full_ids: List[int] = context_ids + target_token_ids

    # --- Truncation: keep the end of context if too long ---
    if len(full_ids) > max_seq_length:
        overshoot = len(full_ids) - max_seq_length
        # Truncate from the LEFT side of context (keep target intact)
        new_context_ids = context_ids[overshoot:]
        if len(new_context_ids) < 1:
            logger.warning(
                "Context is too short after truncation (%d tokens). "
                "Target alone (%d tokens) may exceed max_seq_length.",
                len(new_context_ids), len(target_token_ids),
            )
            new_context_ids = context_ids[-1:]  # keep at least one token
        logger.warning(
            "Sequence length %d exceeds max_seq_length %d. "
            "Truncating context from left: %d → %d tokens.",
            len(full_ids), max_seq_length, len(context_ids), len(new_context_ids),
        )
        context_ids = new_context_ids
        full_ids = context_ids + target_token_ids

    device = next(model.parameters()).device
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    target_start = len(context_ids)
    target_len = len(target_token_ids)

    if target_len == 0:
        logger.warning("target_token_ids is empty, returning 0.0")
        return 0.0

    # --- Forward pass (teacher forcing, no grad) ---
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # For causal LM: logits[t] predicts token at position t+1.
    # So to get the log-prob of target[j] (at full_ids position target_start+j),
    # we look at logits at position (target_start + j - 1).
    log_prob = 0.0
    for j in range(score_from, target_len):
        logit_pos = target_start + j - 1  # position in input whose logits predict next
        if logit_pos < 0:
            # Target starts at position 0 — no causal context. Skip this token.
            logger.warning("target token at position 0 has no causal context; skipping.")
            continue
        token_id = target_token_ids[j]
        log_probs = F.log_softmax(logits[0, logit_pos, :], dim=-1)
        log_prob += log_probs[token_id].item()

    return log_prob


def compute_log_prob_batch(
    model,
    tokenizer,
    context_strings: List[str],
    target_token_ids: List[int],
    max_seq_length: int = 4096,
    score_from: int = 0,
) -> List[float]:
    """
    Compute log-probabilities for multiple context strings in a padded batch.
    More efficient than calling compute_log_prob in a loop when GPU memory allows.

    All contexts share the same target_token_ids.

    Returns:
        List of float log-probabilities, one per context string.
    """
    device = next(model.parameters()).device
    target_len = len(target_token_ids)

    if target_len == 0:
        return [0.0] * len(context_strings)

    all_context_ids: List[List[int]] = []
    for ctx in context_strings:
        cids = tokenizer.encode(ctx, add_special_tokens=True)
        full = cids + target_token_ids
        if len(full) > max_seq_length:
            overshoot = len(full) - max_seq_length
            cids = cids[overshoot:]
            logger.warning("Batch context truncated from %d to %d tokens.", len(cids) + overshoot, len(cids))
        all_context_ids.append(cids)

    max_len = max(len(c) + target_len for c in all_context_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    batch_input_ids = []
    batch_attention_mask = []
    context_lengths = []

    for cids in all_context_ids:
        full = cids + target_token_ids
        ctx_len = len(cids)
        pad_len = max_len - len(full)
        padded = [pad_id] * pad_len + full
        mask = [0] * pad_len + [1] * len(full)
        batch_input_ids.append(padded)
        batch_attention_mask.append(mask)
        context_lengths.append(ctx_len + pad_len)  # offset into the padded sequence

    input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch, seq_len, vocab_size]

    results = []
    for b, ctx_offset in enumerate(context_lengths):
        log_prob = 0.0
        # ctx_offset = position in padded sequence where context ends (= start of target)
        target_start = ctx_offset
        for j in range(score_from, target_len):
            logit_pos = target_start + j - 1
            if logit_pos < 0:
                continue
            token_id = target_token_ids[j]
            lp = F.log_softmax(logits[b, logit_pos, :], dim=-1)
            log_prob += lp[token_id].item()
        results.append(log_prob)

    return results
