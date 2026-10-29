#!/bin/bash
# Evaluate every dataset folder under a data root with the localization method.
#
# Examples:
#   bash run_eval.sh                 # whole benchmark, LLM judge on DPI only
#   bash run_eval.sh --judge_all     # LLM judge on every attack type
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${SCRIPT_DIR}/dataset"
RESULT_ROOT="${SCRIPT_DIR}/results"

SKIP_ATTACK=""
SKIP_DATASET=""
JUDGE_ARGS=()
TEMP_ROOT=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [--data_root DIR] [--skip_attack {DPI|IPI|MP|PoT}] [--skip_dataset DATASET] [--judge_all]

Options:
  --data_root    Folder holding the dataset folders (default: dataset).
  --skip_attack  Skip one attack family:
                 DPI skips dpi_*
                 IPI skips ipi_*
                 MP skips mp_*
                 PoT skips pot_*
  --skip_dataset Skip one dataset, e.g. Finance-Chain.
  --judge_all    Also apply the LLM judge to IPI/MP/PoT logs. Without it only DPI logs are
                 judged, which assumes the user query of the other attacks is clean.
  -h, --help     Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_root)
      if [[ $# -lt 2 ]]; then
        echo "--data_root requires a directory" >&2
        exit 1
      fi
      DATA_ROOT="$(cd "$2" && pwd)"
      shift 2
      ;;
    --skip_attack)
      if [[ $# -lt 2 ]]; then
        echo "--skip_attack requires one value: DPI, IPI, MP, or PoT" >&2
        exit 1
      fi
      SKIP_ATTACK="$2"
      shift 2
      ;;
    --skip_dataset)
      if [[ $# -lt 2 ]]; then
        echo "--skip_dataset requires one value, e.g. Finance-Chain" >&2
        exit 1
      fi
      SKIP_DATASET="$2"
      shift 2
      ;;
    --judge_all)
      JUDGE_ARGS=(--judge-non-dpi)
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${SKIP_ATTACK}" in
  ""|"DPI"|"dpi"|"IPI"|"ipi"|"MP"|"mp"|"PoT"|"POT"|"pot") ;;
  *)
    echo "Invalid --skip_attack value: ${SKIP_ATTACK}. Use DPI, IPI, MP, or PoT." >&2
    exit 1
    ;;
esac

skip_family() {
  local attack_name="$1"
  case "${SKIP_ATTACK}" in
    "DPI"|"dpi") [[ "${attack_name}" == dpi_* ]] ;;
    "IPI"|"ipi") [[ "${attack_name}" == ipi_* ]] ;;
    "MP"|"mp") [[ "${attack_name}" == mp_* ]] ;;
    "PoT"|"POT"|"pot") [[ "${attack_name}" == pot_* ]] ;;
    *) return 1 ;;
  esac
}

make_filtered_base() {
  local source_dir="$1"
  local base_name="$2"
  local filtered_dir="${TEMP_ROOT}/${base_name}"

  mkdir -p "${filtered_dir}"
  for attack_dir in "${source_dir}"/*; do
    [[ -d "${attack_dir}" ]] || continue
    local attack_name
    attack_name="$(basename "${attack_dir}")"
    if skip_family "${attack_name}"; then
      continue
    fi
    ln -s "${attack_dir}" "${filtered_dir}/${attack_name}"
  done

  echo "${filtered_dir}"
}

if [[ -n "${SKIP_ATTACK}" ]]; then
  TEMP_ROOT="$(mktemp -d /tmp/mas_eval_filtered.XXXXXX)"
  trap 'rm -rf "${TEMP_ROOT}"' EXIT
fi

BASE_NAMES=(
  "Finance-Chain"
  "Finance-Tree"
  "Finance-Random"
  "Finance-MD"
  "Computer-Chain"
  "Academic-Chain"
)

for BASE_NAME in "${BASE_NAMES[@]}"; do
  if [[ -n "${SKIP_DATASET}" && "${BASE_NAME}" == "${SKIP_DATASET}" ]]; then
    echo "Skipping dataset ${BASE_NAME}"
    continue
  fi

  BASE_DIR="${DATA_ROOT}/${BASE_NAME}"
  OUTPUT_DIR="${RESULT_ROOT}/${BASE_NAME}"
  RUN_BASE_DIR="${BASE_DIR}"

  if [[ ! -d "${BASE_DIR}" ]]; then
    echo "Missing dataset directory: ${BASE_DIR}" >&2
    echo "Extract the benchmark first: tar -xzf dataset.tar.gz" >&2
    exit 1
  fi

  if [[ -n "${SKIP_ATTACK}" ]]; then
    RUN_BASE_DIR="$(make_filtered_base "${BASE_DIR}" "${BASE_NAME}")"
  fi

  echo "Running evaluation for ${BASE_NAME}"
  echo "  input:  ${BASE_DIR}"
  if [[ -n "${SKIP_ATTACK}" ]]; then
    echo "  skip:   ${SKIP_ATTACK}"
  fi
  echo "  output: ${OUTPUT_DIR}"

  python "${SCRIPT_DIR}/scripts/run_all.py" \
    --base-dirs "${RUN_BASE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    ${JUDGE_ARGS[@]+"${JUDGE_ARGS[@]}"}
done
