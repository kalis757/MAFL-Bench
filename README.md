# Forensic Localization in LLM Multi-Agent Systems

This is the official implementation of the paper `Forensic Localization in LLM Multi-Agent Systems` (NeurIPS 2026). It contains MAFL-Bench, a benchmark of attacked multi-agent-system (MAS) executions, and the code of MATracer, our method that finds where an attack entered. Given the log of one successful attack, MATracer answers three questions: which agent was compromised first, which module of that agent was poisoned, and which units inside the module (planning examples, memory entries, observations or query sentences) carry the attack.

## Getting the data

The whole benchmark is shipped as one archive, `dataset.tar.gz`, in the root of this repository. **Extract it first, before running any code**:

```bash
tar -xzf dataset.tar.gz        # creates ./dataset/ (21,200 logs)
```

The evaluation scripts read the logs from `./dataset/`, so the archive must be extracted in the repository root.

## Benchmark

MAFL-Bench has 21,200 execution logs. One JSON file is one run of an attacked MAS. The evaluation uses the runs in which the attack succeeded, i.e. in which the agents called the malicious tool.

### Scenarios

| Dataset folder | Domain | Topology | Agents |
| --- | --- | --- | --- |
| `Finance-Chain` | financial analysis | chain | planner, executor, summarizer |
| `Finance-Tree` | financial analysis | tree | one delegator and four analysts |
| `Finance-Random` | financial analysis | random | four planners that discuss in random order, one executor |
| `Finance-MD` | financial analysis | multi-agent debate (MD) | two full analysts, each planning and executing |
| `Computer-Chain` | computer-system administration | chain | planner, executor, summarizer |
| `Academic-Chain` | academic research | chain | planner, executor, summarizer |

### Attacks

Every dataset has 12 attack folders. Each attack poisons one module of the MAS:

| Attack | Folders | Poisoned module |
| --- | --- | --- |
| DPI (direct prompt injection) | `dpi_naive`, `dpi_escape_characters`, `dpi_context_ignoring`, `dpi_fake_completion`, `dpi_combined` | `INP` (user query) |
| IPI (indirect prompt injection) | `ipi_naive`, `ipi_escape_characters`, `ipi_context_ignoring`, `ipi_fake_completion`, `ipi_combined` | `OBS` (tool observations) |
| MP (memory poisoning) | `mp_experience` | `MEM` (retrieved memory) |
| PoT (backdoored planning examples) | `pot_backdoor` | `POT` (planning examples in the system prompt) |

### Size

| Dataset | DPI | IPI | MP | PoT | Total |
| --- | --- | --- | --- | --- | --- |
| Finance-Chain | 1000 | 2000 | 200 | 200 | 3400 |
| Finance-Tree | 1000 | 3000 | 400 | 400 | 4800 |
| Finance-Random | 1000 | 1000 | 200 | 200 | 2400 |
| Finance-MD | 1000 | 2000 | 400 | 400 | 3800 |
| Computer-Chain | 1000 | 2000 | 200 | 200 | 3400 |
| Academic-Chain | 1000 | 2000 | 200 | 200 | 3400 |

The logs were generated with Llama-3.3-70B-Instruct (4-bit quantized).

### Log format

Each log contains the original user query, the execution graph (agents and the messages on its edges), and for every agent its system prompt with the planning examples, the retrieved memory, the tool calls with their observations, and its output. The `ground_truth` block names the malicious tool, the agent that called it, the earliest poisoned agent and the poisoned module. Every planning example, memory pair and observation carries a 0/1 poison label.

### Layout

```text
dataset.tar.gz                   archive of the full benchmark (extract first)
dataset/                         created by extraction, 21,200 logs
├── counts.csv
└── <dataset>/<attack folder>/<query id>_atk<NNN>_agent<K>.json
```

## Evaluation

The method is evaluated at three levels:

- **Level 1, agent**: is the earliest compromised agent identified? (`ACC_agent`)
- **Level 2, module**: is, in addition, the poisoned module (`INP`, `POT`, `MEM` or `OBS`) of that agent identified? (`ACC_module`)
- **Level 3, unit**: is, in addition, the exact set of poisoned units identified? (`ACC_unit`)

The units of a log are every planning example, every retrieved memory pair, every observation and the sentences of the user query. A log counts as correct at a level when all levels up to it are correct. DPI logs are evaluated with `ACC_unit` only, without `ACC_agent` and `ACC_module`, consistent with the paper. On the unit level the evaluation also reports

- `FNR`: missed malicious units / all malicious units,
- `FPR`: benign units flagged as malicious / all benign units,
- `DACC`: correctly classified units / all units.

## Method

MATracer scores a candidate cause by how much it raises the log-probability of the malicious tool call τ under a causal language model.

## Usage

### Requirements

- Python 3.11+, one CUDA GPU (the default Llama-3.1-8B-Instruct in fp16 needs about 20 GB; `model.quantization: 4bit` fits smaller cards), internet access to download the model
- `pip install -r requirements.txt`
- Access to the model on the HuggingFace Hub (`huggingface-cli login` for gated models)

The model and the other settings are in [`config/default.yaml`](./config/default.yaml): `model.name_or_path` (Hub ID or local path), `model.quantization` (`none`, `4bit`, `8bit`) and the prompts and thresholds of the method.

### Run

Extract the benchmark first (`tar -xzf dataset.tar.gz`, see [Getting the data](#getting-the-data)), then:

```bash
bash run_eval.sh                  # whole benchmark, LLM judge on DPI only
bash run_eval.sh --judge_all      # LLM judge on every attack type
```

Further options: `--skip_attack {DPI|IPI|MP|PoT}` and `--skip_dataset <dataset folder>`. To choose a GPU, set `CUDA_VISIBLE_DEVICES` as usual.

A single dataset can be evaluated directly:

```bash
python scripts/run_all.py --base-dirs dataset/Finance-Chain --output-dir results/Finance-Chain
```

### Output

For every dataset and attack folder the run writes `results/<dataset>/eval_report_<attack folder>.json` (summary, per-log results) and a Markdown version of it.

## Repository layout

```text
config/default.yaml               model, prompts, thresholds, judge setting
core/                             localization method (levels 1-3, log-probability scoring, LLM judge)
data/                             log schema and loader
evaluation/                       evaluation loop, metrics, reports
scripts/run_all.py                evaluation entry point
run_eval.sh                       batch run over all datasets
dataset.tar.gz                    MAFL-Bench archive; extract it to get dataset/
```
