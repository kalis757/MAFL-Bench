"""Report generation for evaluation results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from evaluation.metrics import EvalReport, SingleResult

logger = logging.getLogger(__name__)


def _result_to_dict(r: SingleResult) -> Dict[str, Any]:
    d = {
        "log_file": r.log_file,
        "attack_type": r.attack_type,
        "gt_agent": r.gt_agent,
        "gt_module": r.gt_module,
        "pred_agent": r.pred_agent,
        "pred_module": r.pred_module,
        "level1_correct": r.level1_correct,
        "level2_correct": r.level2_correct,
        "full_correct": r.full_correct,
        "level2_skipped": r.level2_skipped,
        "level2_skip_reason": r.level2_skip_reason,
        "gain_margin": r.gain_margin,
        "level2_gain_margin": r.level2_gain_margin,
        "gains": r.gains,
        "l1_scores": r.l1_scores,
        "module_gains": r.module_gains,
        "error": r.error,
        "gt_elements": r.gt_elements,
        "pred_elements": r.pred_elements,
        "level3_correct": r.level3_correct,
        "level3_skipped": r.level3_skipped,
        "level3_skip_reason": r.level3_skip_reason,
        "level3_inconclusive": r.level3_inconclusive,
        "level3_gain_margin": r.level3_gain_margin,
        "level3_element_gains": r.level3_element_gains,
        "level3_mode": r.level3_mode,
        "llm_judge_enabled": r.llm_judge_enabled,
        "llm_judge_answer": r.llm_judge_answer,
        "llm_judge_raw_output": r.llm_judge_raw_output,
        "llm_judge_extracted_sentences": r.llm_judge_extracted_sentences,
        "llm_judge_gold_sentences": r.llm_judge_gold_sentences,
        "llm_judge_span_precision": r.llm_judge_span_precision,
        "llm_judge_span_recall": r.llm_judge_span_recall,
        "llm_judge_span_f1": r.llm_judge_span_f1,
        "llm_judge_span_hit": r.llm_judge_span_hit,
        "llm_judge_short_circuit": r.llm_judge_short_circuit,
        "llm_judge_skip_reason": r.llm_judge_skip_reason,
        "gt_unit_ids": r.gt_unit_ids,
        "pred_unit_ids": r.pred_unit_ids,
        "total_units": r.total_units,
        "malicious_units": r.malicious_units,
        "benign_units": r.benign_units,
        "missed_malicious_units": r.missed_malicious_units,
        "false_positive_units": r.false_positive_units,
        "correctly_classified_units": r.correctly_classified_units,
        "unit_correct": r.unit_correct,
    }
    return d


def save_json_report(report: EvalReport, output_path: str) -> None:
    """Save the full evaluation report as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "n_total": report.n_total,
        "n_evaluated": len([r for r in report.results if r.error is None]),
        "n_skipped": report.n_skipped,
        "metrics": {
            **({"ACC_agent": report.acc_agent, "ACC_module": report.acc_module}
               if report.acc_agent is not None else {}),
            "ACC_unit": report.acc_unit,
            "FNR": report.fnr,
            "FPR": report.fpr,
            "DACC": report.dacc,
            "LLM_judge_span_precision_mean": report.llm_judge_span_precision_mean,
            "LLM_judge_span_recall_mean": report.llm_judge_span_recall_mean,
            "LLM_judge_span_f1_mean": report.llm_judge_span_f1_mean,
            "LLM_judge_span_hit_rate": report.llm_judge_span_hit_rate,
            **({"ACC_agent_by_type": report.acc_agent_by_type,
                "ACC_module_by_type": report.acc_module_by_type}
               if report.acc_agent_by_type else {}),
            "ACC_unit_by_type": report.acc_unit_by_type,
            "LLM_judge_span_precision_mean_by_type": report.llm_judge_span_precision_mean_by_type,
            "LLM_judge_span_recall_mean_by_type": report.llm_judge_span_recall_mean_by_type,
            "LLM_judge_span_f1_mean_by_type": report.llm_judge_span_f1_mean_by_type,
            "LLM_judge_span_hit_rate_by_type": report.llm_judge_span_hit_rate_by_type,
            "FNR_by_type": report.fnr_by_type,
            "FPR_by_type": report.fpr_by_type,
            "DACC_by_type": report.dacc_by_type,
            "unit_counts": {
                "total_units": report.total_units,
                "malicious_units": report.malicious_units,
                "benign_units": report.benign_units,
                "missed_malicious_units": report.missed_malicious_units,
                "false_positive_units": report.false_positive_units,
                "correctly_classified_units": report.correctly_classified_units,
            },
            "definitions": {
                "ACC_agent": "original L1",
                "ACC_module": "L1 x L2",
                "ACC_unit": "L1 x L2 x L3",
                "LLM_judge_span_precision_mean": "mean token overlap precision for extracted malicious sentences",
                "LLM_judge_span_recall_mean": "mean token overlap recall for extracted malicious sentences",
                "LLM_judge_span_f1_mean": "mean token overlap F1 for extracted malicious sentences",
                "LLM_judge_span_hit_rate": "share of logs with recall >= 0.8 and precision >= 0.5",
                "FNR": "missed malicious units / total malicious units",
                "FPR": "benign units classified as malicious / total benign units",
                "DACC": "correctly classified units / total units",
            },
        },
        "diagnostics": {
            "gain_margin": {
                "mean": report.gain_margin_mean,
                "std": report.gain_margin_std,
            },
            "module_accuracy_given_l1": report.module_accuracy_given_l1,
            "level2_gain_margin": {
                "mean": report.level2_gain_margin_mean,
                "std": report.level2_gain_margin_std,
            },
            "level3_gain_margin": {
                "mean": report.level3_gain_margin_mean,
                "std": report.level3_gain_margin_std,
            },
        },
    }

    data = {
        "summary": summary,
        "confusion_matrices": {
            "agent": report.agent_confusion,
            "module": report.module_confusion,
        },
        "per_log_results": [_result_to_dict(r) for r in report.results],
        "skipped_files": report.skipped_files,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("JSON report saved to %s", output_path)


def _pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def _conf_matrix_md(conf: Dict[str, Dict[str, int]], label: str) -> str:
    if not conf:
        return f"*No {label} confusion matrix data.*\n"

    all_keys = sorted(set(list(conf.keys()) + [v for row in conf.values() for v in row]))
    header = "| GT \\ Pred | " + " | ".join(all_keys) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(all_keys)) + " |"
    rows = [header, sep]
    for gt in all_keys:
        row_data = conf.get(gt, {})
        cells = [str(row_data.get(pred, 0)) for pred in all_keys]
        rows.append(f"| {gt} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def save_markdown_report(report: EvalReport, output_path: str) -> None:
    """Save a human-readable Markdown evaluation report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_eval = len([r for r in report.results if r.error is None])
    n_errors = len([r for r in report.results if r.error is not None])

    lines = [
        "# Evaluation Report",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"| --- | --- |",
        f"| Total logs processed | {report.n_total} |",
        f"| Evaluated (attack, non-error) | {n_eval} |",
        f"| Skipped (clean or parse error) | {report.n_skipped} |",
        f"| Errors during evaluation | {n_errors} |",
        *(
            [
                f"| **ACC_agent** | **{_pct(report.acc_agent)}** |",
                f"| **ACC_module** | **{_pct(report.acc_module)}** |",
            ]
            if report.acc_agent is not None else []
        ),
        f"| **ACC_unit** | **{_pct(report.acc_unit)}** |",
        f"| **LLM judge span precision (mean)** | **{_pct(report.llm_judge_span_precision_mean)}** |",
        f"| **LLM judge span recall (mean)** | **{_pct(report.llm_judge_span_recall_mean)}** |",
        f"| **LLM judge span F1 (mean)** | **{_pct(report.llm_judge_span_f1_mean)}** |",
        f"| **LLM judge span hit rate** | **{_pct(report.llm_judge_span_hit_rate)}** |",
        f"| **FNR** | **{_pct(report.fnr)}** |",
        f"| **FPR** | **{_pct(report.fpr)}** |",
        f"| **DACC** | **{_pct(report.dacc)}** |",
        f"| Total units | {report.total_units} |",
        f"| Malicious units | {report.malicious_units} |",
        f"| Benign units | {report.benign_units} |",
    ]

    lines += [
        "",
        "## Accuracy by Attack Type",
        "",
    ]

    from collections import Counter
    type_counts = Counter(r.attack_type for r in report.results if r.error is None)

    show_agent = bool(report.acc_agent_by_type)
    lines += [
        "| Attack Type | "
        + ("ACC_agent | ACC_module | " if show_agent else "")
        + "ACC_unit | Span P | Span R | Span F1 | Span Hit | FNR | FPR | DACC | N |",
        "| --- | " + ("--- | --- | " if show_agent else "") + "--- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for atype in sorted(report.acc_unit_by_type.keys()):
        agent_cells = ""
        if show_agent:
            if atype in report.acc_agent_by_type:
                agent_cells = (
                    f"{_pct(report.acc_agent_by_type[atype])} | "
                    f"{_pct(report.acc_module_by_type.get(atype, 0.0))} | "
                )
            else:
                agent_cells = "- | - | "
        acc_unit = report.acc_unit_by_type.get(atype, 0.0)
        span_precision = report.llm_judge_span_precision_mean_by_type.get(atype, 0.0)
        span_recall = report.llm_judge_span_recall_mean_by_type.get(atype, 0.0)
        span_f1 = report.llm_judge_span_f1_mean_by_type.get(atype, 0.0)
        span_hit = report.llm_judge_span_hit_rate_by_type.get(atype, 0.0)
        fnr = report.fnr_by_type.get(atype, 0.0)
        fpr = report.fpr_by_type.get(atype, 0.0)
        dacc = report.dacc_by_type.get(atype, 0.0)
        n = type_counts.get(atype, 0)
        lines.append(
            f"| {atype} | {agent_cells}{_pct(acc_unit)} | "
            f"{_pct(span_precision)} | {_pct(span_recall)} | {_pct(span_f1)} | {_pct(span_hit)} | "
            f"{_pct(fnr)} | {_pct(fpr)} | {_pct(dacc)} | {n} |"
        )

    if show_agent:
        lines += [
            "",
            "## Confusion Matrix",
            "",
            "### Agent (Level 1)",
            "",
        ]
        lines.append(_conf_matrix_md(report.agent_confusion, "agent"))
        lines += [
            "",
            "### Module (Level 2)",
            "",
        ]
        lines.append(_conf_matrix_md(report.module_confusion, "module"))

    lines += [
        "",
        "## Per-Log Results",
        "",
        "| File | Attack | "
        + ("GT Agent | Pred Agent | GT Module | Pred Module | ACC_agent | ACC_module | " if show_agent else "")
        + "ACC_unit | FNR Units | FPR Units | DACC Units |",
        "| --- | --- | " + ("--- | --- | --- | --- | --- | --- | " if show_agent else "") + "--- | --- | --- | --- |",
    ]
    for r in report.results:
        fname = Path(r.log_file).name
        acc_unit = "✓" if r.unit_correct else "✗"
        err = f" ⚠ {r.error}" if r.error else ""
        agent_cells = ""
        if show_agent:
            if r.attack_type == "DPI":
                agent_cells = "- | - | - | - | - | - | "
            else:
                acc_agent = "✓" if r.level1_correct else "✗"
                acc_module = "✓" if (r.level1_correct and r.level2_correct) else "✗"
                agent_cells = (
                    f"{r.gt_agent} | {r.pred_agent} | {r.gt_module} | {r.pred_module or '-'} | "
                    f"{acc_agent} | {acc_module} | "
                )
                if not err and r.level2_skipped and r.level2_skip_reason:
                    err = f" ⚠ L2 skipped: {r.level2_skip_reason}"
        lines.append(
            f"| {fname}{err} | {r.attack_type} | {agent_cells}{acc_unit} | "
            f"{r.missed_malicious_units}/{r.malicious_units} | "
            f"{r.false_positive_units}/{r.benign_units} | "
            f"{r.correctly_classified_units}/{r.total_units} |"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Markdown report saved to %s", output_path)
