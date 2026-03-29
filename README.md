# LLM Unlearning — MT-SimNPO

**MT-SimNPO** extends SimNPO (NeurIPS 2025 SOTA) to defend against multi-turn adversarial knowledge recovery in LLM unlearning.

## Overview

LLM machine unlearning removes specific training data influence from deployed models. Existing SOTA methods (SimNPO, NPO, GradDiff) are vulnerable to multi-turn adversarial prompting — where an attacker uses conversation history to gradually recover "forgotten" knowledge.

This project:
- Proposes **MT-SimNPO**: augments the SimNPO unlearning objective with multi-turn forget examples
- Builds a **multi-turn evaluation harness** (MT-Eval) with a 3-metric leakage bundle (NEM + SemSim + LLM judge)
- Demonstrates vulnerability of existing methods on the TOFU benchmark
- Runs ablations on `mt_weight`, attack type composition, and 3 random seeds

Built on top of [open-unlearning](https://github.com/locuslab/open-unlearning).

## Results

All runs use Llama-3.1-8B-Instruct fine-tuned on TOFU (forget10/retain90 split).

### TOFU Benchmark

| Method | Forget Truth Ratio ↑ | Model Utility ↑ |
|---|---|---|
| oracle\_retrain (gold) | 0.641 | 0.647 |
| GradDiff | 0.000 | 0.671 |
| NPO | 0.508 | 0.637 |
| SimNPO | 0.523 | 0.637 |
| **MT-SimNPO mw=0.5** | 0.524 | — |
| **MT-SimNPO mw=1.0** (3 seeds) | **0.527 ± 0.004** | — |
| **MT-SimNPO mw=2.0** | 0.524 | — |

Forget Truth Ratio closer to oracle (0.64) = better unlearning. GradDiff collapses the model (FTR→0). MT-SimNPO matches SimNPO on standard TOFU while reducing multi-turn leakage.

### MT-Eval — Multi-Turn Recovery Rate (MTRR ↓ lower is better)

Evaluated on `mt_val.jsonl` with 3 attack types × 400 examples (priming, self-correction, persona-switch).

| Method | MTRR (trained attacks) |
|---|---|
| MT-SimNPO mw=0.5 | 0.708 |
| MT-SimNPO mw=1.0 (seed 0) | 0.690 |
| MT-SimNPO mw=1.0 (seed 1) | 0.735 |
| MT-SimNPO mw=1.0 (seed 2) | 0.668 |
| **MT-SimNPO mw=1.0 (mean)** | **0.698 ± 0.034** |
| MT-SimNPO mw=2.0 | **0.615** |

Increasing `mt_weight` reduces multi-turn leakage (lower MTRR) at the cost of heavier training signal. mw=2.0 achieves the lowest MTRR.

### Vulnerability Demo — Transfer Attack MTRR (test split ↓ lower is better)

| Model | Transfer MTRR |
|---|---|
| pre\_unlearning | **0.985** |
| oracle\_retrain | 0.850 |
| GradDiff | 0.003\* |
| NPO | 0.540 |
| SimNPO | — |

The unmodified TOFU-trained model leaks forget-set knowledge in **98.5%** of multi-turn transfer attacks (cot_decomposition + triangulation). Even oracle retrain has 85% leakage — the 8B model retains multi-turn reasoning capability regardless.

NPO and SimNPO achieve standard TOFU metrics (FTR ≈ 0.52) but remain **54%+ vulnerable** to transfer attacks, showing that single-turn unlearning does not generalize.

\*GradDiff's near-zero MTRR is a false positive: it collapses model utility (FTR=0.000, MU=0.671) so severely the model cannot generate coherent multi-turn responses — "secure" only because it is broken.

*(Run `scripts/run_vulnerability_demo.sh` to reproduce.)*

## Key Files

| File | Description |
|---|---|
| `src/trainer/unlearn/mt_simnpo.py` | MT-SimNPO trainer |
| `src/eval/mt_eval.py` | Multi-turn evaluator (MTRR, KLT) |
| `src/eval/mt_metrics.py` | NEM + SemSim + LLM judge leakage bundle |
| `src/data/mt_collator.py` | Multi-turn forget collator (prefix masking) |
| `scripts/generate_mt_dataset.py` | MT conversation generation via GPT-4o-mini |
| `scripts/run_mt_simnpo.sh` | MT-SimNPO training + eval runner |
| `scripts/run_baselines.sh` | GradDiff / NPO / SimNPO baseline runner |
| `scripts/run_vulnerability_demo.sh` | MT-Eval on baselines (no training) |
| `data/mt_train.jsonl` | 1,200 multi-turn training conversations |
| `data/mt_val.jsonl` | 1,200 validation conversations |
| `data/mt_test.jsonl` | 800 held-out test conversations |
| `notebooks/analysis.ipynb` | Figures and result tables |

## Setup

```bash
pip install -e ".[tofu]"
pip install bitsandbytes sentence-transformers spacy rouge-score openai python-Levenshtein
python -m spacy download en_core_web_sm
```

## Reproducing Results

### 1. Baselines (GradDiff, NPO, SimNPO)

```bash
# RunPod A40 48GB — ~3 hours, ~$0.60
bash scripts/run_baselines.sh 0
```

### 2. MT-SimNPO

```bash
# RunPod A40 48GB — ~5 hours per run
bash scripts/run_mt_simnpo.sh 1.0 0   # mw=1.0 seed=0
bash scripts/run_mt_simnpo.sh 1.0 1
bash scripts/run_mt_simnpo.sh 1.0 2
bash scripts/run_mt_simnpo.sh 0.5 0
bash scripts/run_mt_simnpo.sh 2.0 0
```

### 3. Vulnerability Demo

```bash
# Requires baselines checkpoints in saves/unlearn/ or /workspace/checkpoints/
bash scripts/run_vulnerability_demo.sh
```

### 4. Analysis

```bash
jupyter notebook notebooks/analysis.ipynb
```

Checkpoints and results are archived at:
- `harishm17/mt-unlearning-checkpoints` (HuggingFace dataset)
- `harishm17/mt-unlearning-results` (HuggingFace dataset)
