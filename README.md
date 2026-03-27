# LLM Unlearning — MT-SimNPO

**MT-SimNPO** extends SimNPO (NeurIPS 2025 SOTA) to defend against multi-turn adversarial knowledge recovery in LLM unlearning.

> Full results, analysis, and writeup coming soon.

## Overview

LLM machine unlearning removes specific training data influence from deployed models. Existing SOTA methods (SimNPO, NPO, GradDiff) are vulnerable to multi-turn adversarial prompting — where an attacker uses conversation history to gradually recover "forgotten" knowledge.

This project:
- Proposes **MT-SimNPO**: augments the SimNPO unlearning objective with multi-turn forget examples
- Builds a **multi-turn evaluation harness** (MT-Eval) with a 3-metric leakage bundle (NEM + SemSim + LLM judge)
- Demonstrates vulnerability of existing methods on the TOFU benchmark
- Runs ablations on mt_weight, attack type composition, and 3 random seeds

Built on top of [open-unlearning](https://github.com/locuslab/open-unlearning).

## Key Files

| File | Description |
|---|---|
| `src/trainer/unlearn/mt_simnpo.py` | MT-SimNPO trainer |
| `src/eval/mt_eval.py` | Multi-turn evaluator (MTRR, KLT) |
| `src/eval/mt_metrics.py` | NEM + SemSim + LLM judge leakage bundle |
| `src/data/mt_collator.py` | Multi-turn forget collator (prefix masking) |
| `scripts/generate_mt_dataset.py` | MT conversation generation via GPT-4o-mini |
| `scripts/run_mt_simnpo.sh` | Training + eval runner |
| `data/mt_train.jsonl` | 1,200 multi-turn training conversations |
| `data/mt_test.jsonl` | 800 held-out test conversations |

## Setup

```bash
pip install -e ".[dev]"
pip install bitsandbytes peft sentence-transformers spacy rouge-score openai
python -m spacy download en_core_web_sm
```

## Results

*Coming soon.*
