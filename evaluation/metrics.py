"""Evaluation metrics."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from data.schema import ExecutionLog

logger = logging.getLogger(__name__)


@dataclass
class SingleResult:
    """Result for one log file."""
    log_file: str
    attack_type: str
    gt_agent: int
    gt_module: str
    pred_agent: int
    pred_module: str
    level1_correct: bool
    level2_correct: Optional[bool]
    full_correct: Optional[bool]
    level2_skipped: bool
    gain_margin: float
    level2_skip_reason: str = ""
    level2_gain_margin: Optional[float] = None
    gains: List[float] = field(default_factory=list)
    module_gains: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    l1_scores: List[float] = field(default_factory=list)
    gt_elements: List[str] = field(default_factory=list)
    pred_elements: List[str] = field(default_factory=list)
    level3_correct: Optional[bool] = None
    level3_skipped: bool = True
    level3_skip_reason: str = ""
    level3_inconclusive: bool = False
    level3_gain_margin: Optional[float] = None
    level3_element_gains: Dict[str, float] = field(default_factory=dict)
    level3_mode: str = ""
    llm_judge_enabled: bool = False
    llm_judge_answer: str = ""
    llm_judge_raw_output: str = ""
    llm_judge_extracted_sentences: str = ""
    llm_judge_gold_sentences: str = ""
    llm_judge_span_precision: Optional[float] = None
    llm_judge_span_recall: Optional[float] = None
    llm_judge_span_f1: Optional[float] = None
    llm_judge_span_hit: Optional[bool] = None
    llm_judge_short_circuit: bool = False
    llm_judge_skip_reason: str = ""

    gt_unit_ids: List[str] = field(default_factory=list)
    pred_unit_ids: List[str] = field(default_factory=list)
    total_units: int = 0
    malicious_units: int = 0
    benign_units: int = 0
    missed_malicious_units: int = 0
    false_positive_units: int = 0
    correctly_classified_units: int = 0
    unit_correct: bool = False


@dataclass
class EvalReport:
    """Aggregated evaluation results."""
    results: List[SingleResult]
    n_total: int
    n_skipped: int
    skipped_files: List[str] = field(default_factory=list)

    acc_agent: Optional[float] = None
    acc_module: Optional[float] = None
    acc_unit: float = 0.0
    fnr: float = 0.0
    fpr: float = 0.0
    dacc: float = 0.0
    acc_agent_by_type: Dict[str, float] = field(default_factory=dict)
    acc_module_by_type: Dict[str, float] = field(default_factory=dict)
    acc_unit_by_type: Dict[str, float] = field(default_factory=dict)
    fnr_by_type: Dict[str, float] = field(default_factory=dict)
    fpr_by_type: Dict[str, float] = field(default_factory=dict)
    dacc_by_type: Dict[str, float] = field(default_factory=dict)
    total_units: int = 0
    malicious_units: int = 0
    benign_units: int = 0
    missed_malicious_units: int = 0
    false_positive_units: int = 0
    correctly_classified_units: int = 0

    module_accuracy_given_l1: float = 0.0

    gain_margin_mean: float = 0.0
    gain_margin_std: float = 0.0
    level2_gain_margin_mean: float = 0.0
    level2_gain_margin_std: float = 0.0

    agent_confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    module_confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)

    level3_gain_margin_mean: float = 0.0
    level3_gain_margin_std: float = 0.0
    llm_judge_span_precision_mean: float = 0.0
    llm_judge_span_recall_mean: float = 0.0
    llm_judge_span_f1_mean: float = 0.0
    llm_judge_span_hit_rate: float = 0.0
    llm_judge_span_precision_mean_by_type: Dict[str, float] = field(default_factory=dict)
    llm_judge_span_recall_mean_by_type: Dict[str, float] = field(default_factory=dict)
    llm_judge_span_f1_mean_by_type: Dict[str, float] = field(default_factory=dict)
    llm_judge_span_hit_rate_by_type: Dict[str, float] = field(default_factory=dict)


def extract_ground_truth(log: ExecutionLog) -> Tuple[int, str]:
    """
    Extract ground truth (agent_id, module) from a log.

    Also cross-validates against module_labels and logs warnings.

    Returns:
        (gt_agent_id, gt_module_str)
    """
    gt = log.ground_truth
    gt_agent = gt.earliest_poisoned_agent_id
    gt_module = gt.poisoned_module

    if gt_agent is None or gt_module is None:
        raise ValueError(f"Log {log.metadata.run_id} missing ground truth agent/module.")

    label1_agents = [
        a.agent_id for a in log.agents
        if any(getattr(a.module_labels, m) == 1 for m in ("POT", "MEM", "INP", "OBS"))
    ]
    if label1_agents and min(label1_agents) != gt_agent:
        logger.warning(
            "Log %s: ground_truth agent=%d but module_labels scan found agent %d with label=1. "
            "Using ground_truth for evaluation.",
            log.metadata.run_id, gt_agent, min(label1_agents),
        )

    return gt_agent, gt_module


def compute_metrics(report: EvalReport) -> EvalReport:
    """
    Compute all aggregate metrics from report.results and fill in the report fields.
    Modifies report in place and returns it.
    """
    valid = [r for r in report.results if r.error is None]
    if not valid:
        logger.warning("No valid results to compute metrics from.")
        return report

    n_valid = len(valid)

    # DPI logs are evaluated at the unit level only.
    agent_level = [r for r in valid if r.attack_type != "DPI"]
    if agent_level:
        report.acc_agent = sum(r.level1_correct for r in agent_level) / len(agent_level)
        report.acc_module = (
            sum(bool(r.level1_correct and r.level2_correct) for r in agent_level)
            / len(agent_level)
        )
    report.acc_unit = sum(bool(r.unit_correct) for r in valid) / n_valid

    l2_valid = [r for r in agent_level if not r.level2_skipped and r.level2_correct is not None]
    l2_given_l1 = [r for r in l2_valid if r.level1_correct]

    if l2_given_l1:
        report.module_accuracy_given_l1 = (
            sum(bool(r.level2_correct) for r in l2_given_l1) / len(l2_given_l1)
        )

    by_type: Dict[str, List[SingleResult]] = defaultdict(list)
    for r in valid:
        by_type[r.attack_type].append(r)

    for atype, results in by_type.items():
        if atype != "DPI":
            report.acc_agent_by_type[atype] = (
                sum(r.level1_correct for r in results) / len(results)
            )
            report.acc_module_by_type[atype] = (
                sum(bool(r.level1_correct and r.level2_correct) for r in results)
                / len(results)
            )
        report.acc_unit_by_type[atype] = (
            sum(bool(r.unit_correct) for r in results) / len(results)
        )

        type_malicious = sum(r.malicious_units for r in results)
        type_benign = sum(r.benign_units for r in results)
        type_total = sum(r.total_units for r in results)
        report.fnr_by_type[atype] = (
            sum(r.missed_malicious_units for r in results) / type_malicious
            if type_malicious else 0.0
        )
        report.fpr_by_type[atype] = (
            sum(r.false_positive_units for r in results) / type_benign
            if type_benign else 0.0
        )
        report.dacc_by_type[atype] = (
            sum(r.correctly_classified_units for r in results) / type_total
            if type_total else 0.0
        )
    gain_margins = [r.gain_margin for r in agent_level]
    report.gain_margin_mean = float(np.mean(gain_margins)) if gain_margins else 0.0
    report.gain_margin_std = float(np.std(gain_margins)) if gain_margins else 0.0
    level2_gain_margins = [
        r.level2_gain_margin for r in l2_valid if r.level2_gain_margin is not None
    ]
    report.level2_gain_margin_mean = (
        float(np.mean(level2_gain_margins)) if level2_gain_margins else 0.0
    )
    report.level2_gain_margin_std = (
        float(np.std(level2_gain_margins)) if level2_gain_margins else 0.0
    )

    agent_conf: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in agent_level:
        agent_conf[str(r.gt_agent)][str(r.pred_agent)] += 1
    report.agent_confusion = {k: dict(v) for k, v in agent_conf.items()}

    module_conf: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in l2_valid:
        module_conf[r.gt_module][r.pred_module] += 1
    report.module_confusion = {k: dict(v) for k, v in module_conf.items()}

    report.total_units = sum(r.total_units for r in valid)
    report.malicious_units = sum(r.malicious_units for r in valid)
    report.benign_units = sum(r.benign_units for r in valid)
    report.missed_malicious_units = sum(r.missed_malicious_units for r in valid)
    report.false_positive_units = sum(r.false_positive_units for r in valid)
    report.correctly_classified_units = sum(r.correctly_classified_units for r in valid)
    report.fnr = (
        report.missed_malicious_units / report.malicious_units
        if report.malicious_units else 0.0
    )
    report.fpr = (
        report.false_positive_units / report.benign_units
        if report.benign_units else 0.0
    )
    report.dacc = (
        report.correctly_classified_units / report.total_units
        if report.total_units else 0.0
    )

    l3_ran = [r for r in valid if not r.level3_skipped and r.level3_correct is not None]
    if l3_ran:
        l3_margins = [r.level3_gain_margin for r in l3_ran if r.level3_gain_margin is not None]
        report.level3_gain_margin_mean = float(np.mean(l3_margins)) if l3_margins else 0.0
        report.level3_gain_margin_std = float(np.std(l3_margins)) if l3_margins else 0.0

    span_eval = [r for r in valid if r.llm_judge_span_hit is not None]
    if span_eval:
        report.llm_judge_span_precision_mean = float(np.mean([
            r.llm_judge_span_precision for r in span_eval if r.llm_judge_span_precision is not None
        ]))
        report.llm_judge_span_recall_mean = float(np.mean([
            r.llm_judge_span_recall for r in span_eval if r.llm_judge_span_recall is not None
        ]))
        report.llm_judge_span_f1_mean = float(np.mean([
            r.llm_judge_span_f1 for r in span_eval if r.llm_judge_span_f1 is not None
        ]))
        report.llm_judge_span_hit_rate = (
            sum(bool(r.llm_judge_span_hit) for r in span_eval) / len(span_eval)
        )

        span_by_type: Dict[str, List[SingleResult]] = defaultdict(list)
        for r in span_eval:
            span_by_type[r.attack_type].append(r)
        for atype, results in span_by_type.items():
            report.llm_judge_span_precision_mean_by_type[atype] = float(np.mean([
                r.llm_judge_span_precision for r in results if r.llm_judge_span_precision is not None
            ]))
            report.llm_judge_span_recall_mean_by_type[atype] = float(np.mean([
                r.llm_judge_span_recall for r in results if r.llm_judge_span_recall is not None
            ]))
            report.llm_judge_span_f1_mean_by_type[atype] = float(np.mean([
                r.llm_judge_span_f1 for r in results if r.llm_judge_span_f1 is not None
            ]))
            report.llm_judge_span_hit_rate_by_type[atype] = (
                sum(bool(r.llm_judge_span_hit) for r in results) / len(results)
            )

    return report
