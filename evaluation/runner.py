"""Batch evaluation over a directory of log files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Set, Tuple

from data.loader import load_log
from data.schema import ExecutionLog
from evaluation.metrics import (
    EvalReport, SingleResult, compute_metrics, extract_ground_truth,
)
from core.llm_judge import (
    dpi_query_units,
    extract_dpi_gold_span,
    flag_dpi_units,
    get_first_user_receiver,
    run_llm_judge,
    score_extracted_span,
)
from core.level1_tld import compute_gain_margin, level1_localize
from core.level2_tld import level2_localize
from core.level3_tld import (
    extract_level3_elements,
    get_level3_ground_truth_elements,
    level3_localize,
)

logger = logging.getLogger(__name__)


def _inp_unit_id() -> str:
    return "INP:user_query"


def _query_unit_id(element_id: str) -> str:
    return f"INP:{element_id}"


def _agent_unit_id(agent_id: int, module: str, element_id: str) -> str:
    return f"agent_{agent_id}:{module}:{element_id}"


def _all_and_malicious_unit_ids(log: ExecutionLog) -> Tuple[Set[str], Set[str]]:
    """Return all countable unit IDs and ground-truth malicious unit IDs."""
    malicious_units: Set[str] = set()

    if log.metadata.attack_config.attack_type == "DPI":
        benign_sentences, malicious_sentences = dpi_query_units(log)
        all_units: Set[str] = {_query_unit_id(f"benign_{i}") for i in range(len(benign_sentences))}
        for j in range(len(malicious_sentences)):
            all_units.add(_query_unit_id(f"malicious_{j}"))
            malicious_units.add(_query_unit_id(f"malicious_{j}"))
    else:
        all_units = {_inp_unit_id()}

    for agent in log.agents:
        for idx, ex in enumerate(agent.system_prompt.pot_prompt.examples):
            unit_id = _agent_unit_id(agent.agent_id, "POT", f"pot_{idx}")
            all_units.add(unit_id)
            if ex.label == 1:
                malicious_units.add(unit_id)

        for idx, pair in enumerate(agent.retrieved_memory.retrieved_pairs):
            unit_id = _agent_unit_id(agent.agent_id, "MEM", f"mem_{idx}")
            all_units.add(unit_id)
            if pair.label == 1:
                malicious_units.add(unit_id)

        obs_elements = extract_level3_elements(agent, "OBS")
        for el in obs_elements:
            all_units.add(_agent_unit_id(agent.agent_id, "OBS", el["id"]))
        if agent.module_labels.OBS == 1:
            for element_id in get_level3_ground_truth_elements(obs_elements, "OBS"):
                malicious_units.add(_agent_unit_id(agent.agent_id, "OBS", element_id))

    return all_units, malicious_units


def _predicted_unit_ids(result: SingleResult) -> Set[str]:
    """Map the hierarchical prediction for one log onto countable unit IDs."""
    if result.pred_agent < 0:
        return set()
    if result.pred_module == "INP":
        return {_query_unit_id(element_id) for element_id in result.pred_elements}
    if result.pred_module in {"POT", "MEM", "OBS"}:
        return {
            _agent_unit_id(result.pred_agent, result.pred_module, element_id)
            for element_id in result.pred_elements
        }
    return set()


def _attach_unit_metrics(result: SingleResult, log: ExecutionLog) -> SingleResult:
    all_units, gt_units = _all_and_malicious_unit_ids(log)
    pred_units = _predicted_unit_ids(result) & all_units
    benign_units = all_units - gt_units

    missed = gt_units - pred_units
    false_positive = pred_units & benign_units
    true_positive = pred_units & gt_units
    true_negative = benign_units - pred_units

    result.gt_unit_ids = sorted(gt_units)
    result.pred_unit_ids = sorted(pred_units)
    result.total_units = len(all_units)
    result.malicious_units = len(gt_units)
    result.benign_units = len(benign_units)
    result.missed_malicious_units = len(missed)
    result.false_positive_units = len(false_positive)
    result.correctly_classified_units = len(true_positive) + len(true_negative)
    result.unit_correct = pred_units == gt_units
    return result


def _should_apply_llm_judge(log: ExecutionLog, config: dict) -> bool:
    """DPI logs are always judged; IPI/MP/PoT only if llm_judge.judge_non_dpi is set."""
    if log.metadata.attack_config.attack_type == "DPI":
        return True
    return bool(config.get("llm_judge", {}).get("judge_non_dpi", False))


def _run_single(
    log: ExecutionLog,
    log_file: str,
    model,
    tokenizer,
    config: dict,
) -> SingleResult:
    """Run Level 1/2/3 on a single log. Returns a SingleResult."""
    gt_agent, gt_module = extract_ground_truth(log)
    attack_type = log.metadata.attack_config.attack_type
    llm_judge_enabled = _should_apply_llm_judge(log, config)
    llm_judge_answer = ""
    llm_judge_raw_output = ""
    llm_judge_extracted_sentences = ""
    llm_judge_gold_sentences = ""
    llm_judge_span_precision = None
    llm_judge_span_recall = None
    llm_judge_span_f1 = None
    llm_judge_span_hit = None
    llm_judge_short_circuit = False
    llm_judge_skip_reason = ""

    try:
        if llm_judge_enabled:
            judge_result = run_llm_judge(log, model, tokenizer, config)
            llm_judge_answer = str(judge_result.get("answer", ""))
            llm_judge_raw_output = str(judge_result.get("raw_output", ""))
            llm_judge_extracted_sentences = str(judge_result.get("extracted_sentences", ""))
            if attack_type == "DPI":
                llm_judge_gold_sentences = extract_dpi_gold_span(log)
                span_metrics = score_extracted_span(
                    llm_judge_extracted_sentences,
                    llm_judge_gold_sentences,
                    config=config,
                )
                llm_judge_span_precision = float(span_metrics["precision"])
                llm_judge_span_recall = float(span_metrics["recall"])
                llm_judge_span_f1 = float(span_metrics["f1"])
                llm_judge_span_hit = bool(span_metrics["hit"])

            if llm_judge_answer == "yes" and attack_type == "DPI":
                pred_agent = get_first_user_receiver(log)
                if pred_agent is None:
                    raise ValueError(f"Could not determine earliest user receiver in {log_file}")
                llm_judge_short_circuit = True
                llm_judge_skip_reason = "LLM judge=yes on DPI; short-circuit to earliest user-edge receiver."
                level1_correct = pred_agent == gt_agent
                benign_sentences, malicious_sentences = dpi_query_units(log)
                gt_query_elements = [f"malicious_{j}" for j in range(len(malicious_sentences))]
                pred_query_elements = (
                    flag_dpi_units(
                        llm_judge_extracted_sentences, benign_sentences, malicious_sentences, config
                    )
                    if level1_correct else []
                )
                return _attach_unit_metrics(SingleResult(
                    log_file=log_file,
                    attack_type=attack_type,
                    gt_agent=gt_agent,
                    gt_module=gt_module,
                    pred_agent=pred_agent,
                    pred_module="INP" if level1_correct else "",
                    level1_correct=level1_correct,
                    level2_correct=level1_correct,
                    full_correct=level1_correct,
                    level2_skipped=False,
                    level2_skip_reason=llm_judge_skip_reason,
                    gain_margin=0.0,
                    level2_gain_margin=None,
                    gt_elements=gt_query_elements,
                    pred_elements=pred_query_elements,
                    level3_correct=level1_correct and set(pred_query_elements) == set(gt_query_elements),
                    level3_skipped=False,
                    level3_skip_reason=llm_judge_skip_reason,
                    level3_mode="set",
                    llm_judge_enabled=True,
                    llm_judge_answer=llm_judge_answer,
                    llm_judge_raw_output=llm_judge_raw_output,
                    llm_judge_extracted_sentences=llm_judge_extracted_sentences,
                    llm_judge_gold_sentences=llm_judge_gold_sentences,
                    llm_judge_span_precision=llm_judge_span_precision,
                    llm_judge_span_recall=llm_judge_span_recall,
                    llm_judge_span_f1=llm_judge_span_f1,
                    llm_judge_span_hit=llm_judge_span_hit,
                    llm_judge_short_circuit=True,
                    llm_judge_skip_reason=llm_judge_skip_reason,
                ), log)

            if llm_judge_answer == "no" and attack_type == "DPI":
                llm_judge_skip_reason = "LLM judge=no on DPI; evaluation stopped and counted as failure."
                return _attach_unit_metrics(SingleResult(
                    log_file=log_file,
                    attack_type=attack_type,
                    gt_agent=gt_agent,
                    gt_module=gt_module,
                    pred_agent=-1,
                    pred_module="",
                    level1_correct=False,
                    level2_correct=None,
                    full_correct=False,
                    level2_skipped=True,
                    level2_skip_reason=llm_judge_skip_reason,
                    gain_margin=0.0,
                    level2_gain_margin=None,
                    level3_correct=None,
                    level3_skipped=True,
                    level3_skip_reason=llm_judge_skip_reason,
                    llm_judge_enabled=True,
                    llm_judge_answer=llm_judge_answer,
                    llm_judge_raw_output=llm_judge_raw_output,
                    llm_judge_extracted_sentences=llm_judge_extracted_sentences,
                    llm_judge_gold_sentences=llm_judge_gold_sentences,
                    llm_judge_span_precision=llm_judge_span_precision,
                    llm_judge_span_recall=llm_judge_span_recall,
                    llm_judge_span_f1=llm_judge_span_f1,
                    llm_judge_span_hit=llm_judge_span_hit,
                    llm_judge_short_circuit=False,
                    llm_judge_skip_reason=llm_judge_skip_reason,
                ), log)

            if llm_judge_answer == "yes" and attack_type != "DPI":
                llm_judge_skip_reason = "LLM judge=yes on non-DPI; evaluation stopped and counted as failure."
                return _attach_unit_metrics(SingleResult(
                    log_file=log_file,
                    attack_type=attack_type,
                    gt_agent=gt_agent,
                    gt_module=gt_module,
                    pred_agent=-1,
                    pred_module="",
                    level1_correct=False,
                    level2_correct=None,
                    full_correct=False,
                    level2_skipped=True,
                    level2_skip_reason=llm_judge_skip_reason,
                    gain_margin=0.0,
                    level2_gain_margin=None,
                    level3_correct=None,
                    level3_skipped=True,
                    level3_skip_reason=llm_judge_skip_reason,
                    llm_judge_enabled=True,
                    llm_judge_answer=llm_judge_answer,
                    llm_judge_raw_output=llm_judge_raw_output,
                    llm_judge_extracted_sentences=llm_judge_extracted_sentences,
                    llm_judge_gold_sentences=llm_judge_gold_sentences,
                    llm_judge_span_precision=llm_judge_span_precision,
                    llm_judge_span_recall=llm_judge_span_recall,
                    llm_judge_span_f1=llm_judge_span_f1,
                    llm_judge_span_hit=llm_judge_span_hit,
                    llm_judge_short_circuit=False,
                    llm_judge_skip_reason=llm_judge_skip_reason,
                ), log)

        pred_agent, gains, edge_mapping, winning_edge_id, l1_extra = level1_localize(
            log, model, tokenizer, config
        )
        gain_margin = compute_gain_margin(gains)

        if attack_type == "DPI":
            if log.metadata.topology_type == "md":
                level1_correct = pred_agent == gt_agent
            else:
                level1_correct = pred_agent == gt_agent and winning_edge_id == "e_0"
        elif gt_agent == 1:
            if log.metadata.topology_type == "chain":
                level1_correct = pred_agent == gt_agent and winning_edge_id == "e_1"
            else:
                winning_from = edge_mapping.get(winning_edge_id, "→").split("→")[0]
                level1_correct = pred_agent == gt_agent and winning_from == "agent_1"
        else:
            level1_correct = pred_agent == gt_agent

        pred_module = ""
        level2_correct = None
        full_correct = None
        level2_skipped = True
        level2_skip_reason = ""
        level2_gain_margin = None
        module_gains = {}
        tau_prime_for_l3 = ""
        response_kind_for_l3 = ""
        selected_phase_index_for_l3 = None

        if not level1_correct:
            level2_skip_reason = "Level 1 incorrect; skipping Level 2."
        else:
            l2_result = level2_localize(
                log, pred_agent, model, tokenizer, config, winning_edge_id=winning_edge_id
            )
            pred_module = l2_result["predicted_module"]
            level2_skipped = l2_result["skipped"]
            level2_skip_reason = l2_result["skip_reason"]
            level2_gain_margin = l2_result["gain_margin"]
            module_gains = l2_result["module_gains"]
            tau_prime_for_l3 = l2_result.get("tau_prime", "")
            response_kind_for_l3 = l2_result.get("response_kind", "")
            selected_phase_index_for_l3 = l2_result.get("selected_phase_index")
            if not level2_skipped:
                level2_correct = pred_module == gt_module
                full_correct = level1_correct and level2_correct

        gt_elements: List[str] = []
        pred_elements: List[str] = []
        level3_correct = None
        level3_skipped = True
        level3_skip_reason = ""
        level3_inconclusive = False
        level3_gain_margin = None
        level3_element_gains: dict = {}
        level3_mode = ""

        if not level1_correct:
            level3_skip_reason = "Level 1 incorrect; skipping Level 3."
        elif level2_skipped or not level2_correct:
            level3_skip_reason = "Level 2 incorrect or skipped; skipping Level 3."
        else:
            agent_obj = log.get_agent(pred_agent)

            plan_record_for_gt = None
            if response_kind_for_l3 == "plan_record" and selected_phase_index_for_l3 is not None:
                sorted_records = sorted(agent_obj.plan_records, key=lambda r: r.phase_index)
                plan_record_for_gt = next(
                    (r for r in sorted_records if r.phase_index == selected_phase_index_for_l3),
                    None,
                )

            elements = extract_level3_elements(agent_obj, pred_module, plan_record=plan_record_for_gt)
            gt_elements = get_level3_ground_truth_elements(elements, pred_module)

            l3_result = level3_localize(
                log, pred_agent, pred_module, tau_prime_for_l3,
                model, tokenizer, config,
                response_kind=response_kind_for_l3,
                selected_phase_index=selected_phase_index_for_l3,
            )
            pred_elements = l3_result["predicted_elements"]
            level3_skipped = l3_result["skipped"]
            level3_skip_reason = l3_result["skip_reason"]
            level3_inconclusive = l3_result["inconclusive"]
            level3_gain_margin = l3_result["gain_margin"]
            level3_element_gains = l3_result["element_gains"]
            level3_mode = (
                "point" if l3_result["predicted_element_type"] == "observation" else "set"
            )

            if not level3_skipped and not level3_inconclusive:
                if level3_mode == "point":
                    level3_correct = (
                        bool(pred_elements) and bool(gt_elements)
                        and pred_elements[0] == gt_elements[0]
                    )
                else:
                    level3_correct = set(pred_elements) == set(gt_elements)

        result = SingleResult(
            log_file=log_file,
            attack_type=attack_type,
            gt_agent=gt_agent,
            gt_module=gt_module,
            pred_agent=pred_agent,
            pred_module=pred_module,
            level1_correct=level1_correct,
            level2_correct=level2_correct,
            full_correct=full_correct,
            level2_skipped=level2_skipped,
            level2_skip_reason=level2_skip_reason,
            gain_margin=gain_margin,
            level2_gain_margin=level2_gain_margin,
            gains=gains,
            module_gains=module_gains,
            gt_elements=gt_elements,
            pred_elements=pred_elements,
            level3_correct=level3_correct,
            level3_skipped=level3_skipped,
            level3_skip_reason=level3_skip_reason,
            level3_inconclusive=level3_inconclusive,
            level3_gain_margin=level3_gain_margin,
            level3_element_gains=level3_element_gains,
            level3_mode=level3_mode,
            l1_scores=l1_extra.get("scores", []),
            llm_judge_enabled=llm_judge_enabled,
            llm_judge_answer=llm_judge_answer,
            llm_judge_raw_output=llm_judge_raw_output,
            llm_judge_extracted_sentences=llm_judge_extracted_sentences,
            llm_judge_gold_sentences=llm_judge_gold_sentences,
            llm_judge_span_precision=llm_judge_span_precision,
            llm_judge_span_recall=llm_judge_span_recall,
            llm_judge_span_f1=llm_judge_span_f1,
            llm_judge_span_hit=llm_judge_span_hit,
            llm_judge_short_circuit=llm_judge_short_circuit,
            llm_judge_skip_reason=llm_judge_skip_reason,
        )

        return _attach_unit_metrics(result, log)

    except Exception as e:
        logger.error("Error processing %s: %s", log_file, e, exc_info=True)
        return _attach_unit_metrics(SingleResult(
            log_file=log_file,
            attack_type=attack_type,
            gt_agent=gt_agent,
            gt_module=gt_module,
            pred_agent=-1,
            pred_module="",
            level1_correct=False,
            level2_correct=None,
            full_correct=None,
            level2_skipped=True,
            level2_skip_reason="Error during evaluation.",
            gain_margin=0.0,
            level2_gain_margin=None,
            error=str(e),
            llm_judge_enabled=llm_judge_enabled,
            llm_judge_answer=llm_judge_answer,
            llm_judge_raw_output=llm_judge_raw_output,
            llm_judge_extracted_sentences=llm_judge_extracted_sentences,
            llm_judge_gold_sentences=llm_judge_gold_sentences,
            llm_judge_span_precision=llm_judge_span_precision,
            llm_judge_span_recall=llm_judge_span_recall,
            llm_judge_span_f1=llm_judge_span_f1,
            llm_judge_span_hit=llm_judge_span_hit,
            llm_judge_short_circuit=llm_judge_short_circuit,
            llm_judge_skip_reason=llm_judge_skip_reason,
        ), log)


def run_evaluation(
    log_dir: str,
    model,
    tokenizer,
    config: dict,
) -> EvalReport:
    """
    Run the full evaluation pipeline over all attack logs in log_dir.

    Args:
        log_dir:    Directory containing JSON log files (searched recursively).
        model:      HuggingFace causal LM (loaded once, reused).
        tokenizer:  Corresponding tokenizer.
        config:     Config dict.

    Returns:
        EvalReport with all results and computed metrics.
    """
    log_files = sorted(Path(log_dir).rglob("*.json"))
    logger.info("Found %d JSON files in %s", len(log_files), log_dir)

    results: List[SingleResult] = []
    skipped: List[str] = []
    n_total = 0

    for log_path in log_files:
        try:
            log = load_log(log_path)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", log_path, e)
            skipped.append(str(log_path))
            continue

        n_total += 1

        if not log.is_attack_log():
            logger.debug("Skipping log (CLEAN or attack_succeeded=False): %s", log_path.name)
            skipped.append(str(log_path))
            continue

        n_done = len(results) + 1
        logger.info(
            "[%d] Processing [%s] %s",
            n_done, log.metadata.attack_config.attack_type, log_path.name,
        )
        result = _run_single(
            log, str(log_path), model, tokenizer, config,
        )
        results.append(result)

        acc_unit = "✓" if result.unit_correct else "✗"
        err_tag = f"  ERROR: {result.error}" if result.error else ""
        agent_tag = ""
        if result.attack_type != "DPI":
            acc_agent = "✓" if result.level1_correct else "✗"
            acc_module = "✓" if (result.level1_correct and result.level2_correct) else "✗"
            agent_tag = (
                f"ACC_agent={acc_agent}(pred={result.pred_agent},gt={result.gt_agent})  "
                f"ACC_module={acc_module}(pred={result.pred_module or '-'},gt={result.gt_module})  "
            )
        print(
            f"[{n_done}] {log_path.name}  {agent_tag}"
            f"ACC_unit={acc_unit}({result.correctly_classified_units}/{result.total_units})"
            f"{err_tag}",
            flush=True,
        )

    report = EvalReport(
        results=results,
        n_total=n_total,
        n_skipped=len(skipped),
        skipped_files=skipped,
    )
    compute_metrics(report)
    return report
