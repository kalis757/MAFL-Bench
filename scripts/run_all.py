#!/usr/bin/env python3
"""Batch evaluation entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from evaluation.report import save_json_report, save_markdown_report
from evaluation.runner import run_evaluation
from model_utils import load_model_and_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("run_all")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _print_summary(report, report_name: str) -> None:
    n_eval = len([r for r in report.results if r.error is None])
    print(f"\n{'='*60}")
    print(f"  {report_name}")
    print(f"{'='*60}")
    print(f"  Evaluated: {n_eval}  Skipped: {report.n_skipped}")
    if report.acc_agent is not None:
        print(f"  ACC_agent:  {report.acc_agent*100:.1f}%")
        print(f"  ACC_module: {report.acc_module*100:.1f}%")
    print(f"  ACC_unit:   {report.acc_unit*100:.1f}%")
    print(f"  FNR:        {report.fnr*100:.1f}%")
    print(f"  FPR:        {report.fpr*100:.1f}%")
    print(f"  DACC:       {report.dacc*100:.1f}%")
    print("  By attack type:")
    for atype in sorted(report.acc_unit_by_type):
        agent_part = ""
        if atype in report.acc_agent_by_type:
            agent_part = (
                f"ACC_agent={report.acc_agent_by_type[atype]*100:.1f}%  "
                f"ACC_module={report.acc_module_by_type.get(atype, 0.0)*100:.1f}%  "
            )
        print(f"    {atype:5s}: {agent_part}ACC_unit={report.acc_unit_by_type[atype]*100:.1f}%")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all attack-type directories with one model load."
    )
    parser.add_argument(
        "--base-dirs",
        nargs="+",
        required=True,
        help="One or more base directories. Each immediate subdirectory is evaluated separately.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/",
        help="Directory to save evaluation reports (default: results/)",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent / "config" / "default.yaml"),
        help="Config YAML path",
    )
    parser.add_argument(
        "--judge-non-dpi",
        action="store_true",
        help="Also apply the LLM judge to IPI/MP/PoT logs (default: DPI logs only).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.judge_non_dpi:
        config.setdefault("llm_judge", {})["judge_non_dpi"] = True

    attack_dirs: list[Path] = []
    for base in args.base_dirs:
        base_path = Path(base)
        for d in sorted(base_path.iterdir()):
            if d.is_dir():
                attack_dirs.append(d)

    if not attack_dirs:
        logger.error("No attack directories found under: %s", args.base_dirs)
        sys.exit(1)

    logger.info("Found %d attack directories to evaluate.", len(attack_dirs))
    model, tokenizer = load_model_and_tokenizer(config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, atk_dir in enumerate(attack_dirs, 1):
        report_name = f"eval_report_{atk_dir.name}"
        logger.info(
            "[%d/%d] Evaluating %s → %s",
            i, len(attack_dirs), atk_dir, report_name,
        )

        report = run_evaluation(
            log_dir=str(atk_dir),
            model=model,
            tokenizer=tokenizer,
            config=config,
        )

        json_path = output_dir / f"{report_name}.json"
        md_path = output_dir / f"{report_name}.md"
        save_json_report(report, str(json_path))
        save_markdown_report(report, str(md_path))

        _print_summary(report, report_name)
        print(f"  Reports saved: {json_path}\n              {md_path}\n", flush=True)

    print("All done.", flush=True)


if __name__ == "__main__":
    main()
