# Reproducibility Guide — MT-SimNPO

All experiments use Llama-3.1-8B-Instruct fine-tuned on TOFU (forget10/retain90).
Hardware: RunPod A40 48 GB spot instance ($0.20/hr).

---

## Prerequisites

### 1. Environment

```bash
pip install -e ".[tofu]"
pip install bitsandbytes sentence-transformers spacy rouge-score openai python-Levenshtein
python -m spacy download en_core_web_sm
```

### 2. Model Checkpoints

Pre-trained TOFU checkpoints (from open-unlearning):

```bash
# TOFU fine-tuned (starting point for unlearning)
huggingface-cli download open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
    --local-dir /workspace/checkpoints/tofu_8b_full

# Oracle retain90 (gold standard upper bound)
huggingface-cli download open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90 \
    --local-dir /workspace/checkpoints/tofu_8b_retain90
```

### 3. Multi-Turn Dataset

Already generated and committed to `data/`:
- `data/mt_train.jsonl` — 1,200 training conversations (priming, self_correction, persona_switch)
- `data/mt_val.jsonl`   — 1,200 validation conversations (v2 variants)
- `data/mt_test.jsonl`  — 800 held-out test conversations (cot_decomposition, triangulation, crescendo)

To regenerate (requires OpenAI API key):
```bash
export OPENAI_API_KEY=<key>
python scripts/generate_mt_dataset.py
```

---

## Experiments

### Oracle Retain Baseline (run once)

```bash
PYTHONPATH=src python src/eval_runner.py --config-name=eval.yaml \
    experiment=eval/tofu/default \
    model=Llama-3.1-8B-Instruct \
    model.model_args.pretrained_model_name_or_path=/workspace/checkpoints/tofu_8b_retain90 \
    model.tokenizer_args.pretrained_model_name_or_path=/workspace/checkpoints/tofu_8b_retain90 \
    retain_logs_path=null \
    task_name=oracle_retain90
```

### Baselines (GradDiff, NPO, SimNPO)

```bash
bash scripts/run_baselines.sh 0   # seed=0, ~3 hrs
```

Key hyperparameters (`configs/experiment/unlearn/tofu/default.yaml`):
- `num_train_epochs: 10`, `lr: 1e-5`, `batch_size: 2`, `grad_accum: 4`
- `optim: adafactor` (required for 8B on A40 48 GB)
- NPO uses 4-bit quantized reference model (~4 GB) via bitsandbytes

### MT-SimNPO

```bash
# Core: 3 seeds at mw=1.0
bash scripts/run_mt_simnpo.sh 1.0 0
bash scripts/run_mt_simnpo.sh 1.0 1
bash scripts/run_mt_simnpo.sh 1.0 2

# Ablation: mt_weight
bash scripts/run_mt_simnpo.sh 0.5 0
bash scripts/run_mt_simnpo.sh 2.0 0
```

Key hyperparameters (`configs/experiment/unlearn/tofu/mt_simnpo_8b.yaml`):
- `beta: 4.5`, `gamma: 0.125`, `alpha: 1.0`, `delta: 0.0`
- `mt_weight: 1.0` (override via script)
- `batch_size: 1`, `grad_accum: 8`, `optim: adafactor`

Loss formula:
```
L = γ · L_SimNPO(D_forget)
  + γ · λ · L_SimNPO(D_mt_forget)
  + α · L_NLL(D_retain)
```
where `λ = mt_weight` controls the multi-turn loss weight.

### Vulnerability Demo

Demonstrates that single-turn unlearning methods remain vulnerable to multi-turn adversarial probing:

```bash
# Requires baselines checkpoints in saves/unlearn/ or /workspace/checkpoints/
bash scripts/run_vulnerability_demo.sh
```

Outputs per-model MTRR to `results/vulnerability/`.

---

## Evaluation

### TOFU Benchmark

```bash
PYTHONPATH=src python src/eval_runner.py --config-name=eval.yaml \
    experiment=eval/tofu/default \
    model=Llama-3.1-8B-Instruct \
    model.model_args.pretrained_model_name_or_path=<checkpoint_path> \
    model.tokenizer_args.pretrained_model_name_or_path=/workspace/checkpoints/tofu_8b_full \
    retain_logs_path=saves/eval/oracle_retain90/TOFU_EVAL.json \
    task_name=<run_id>
```

Key metrics:
- `forget_truth_ratio` — higher toward oracle (0.64) is better
- `model_utility` — higher is better

### MT-Eval

```bash
PYTHONPATH=src python src/eval/mt_eval.py \
    --checkpoint <checkpoint_path> \
    --mt_test_path data/mt_val.jsonl \
    --split val \
    --output results/<run_id>/mt_val.json
```

Key metric: `overall_mtrr_trained` — lower is better (fewer multi-turn leakage successes).

---

## Pre-trained Checkpoints

All training checkpoints and eval results are archived on HuggingFace:

| Resource | HF Path |
|---|---|
| Checkpoints | `harishm17/mt-unlearning-checkpoints` |
| TOFU eval results | `harishm17/mt-unlearning-results/results/eval/` |
| MT-Eval results | `harishm17/mt-unlearning-results/results/mt_eval/` |
| Training logs | `harishm17/mt-unlearning-results/logs/` |

Download a checkpoint:
```bash
huggingface-cli download harishm17/mt-unlearning-checkpoints \
    --include "checkpoints/MTSimNPO_mw1.0_seed0/*" \
    --local-dir /workspace/checkpoints \
    --repo-type dataset
```

---

## Results Summary

| Method | FTR ↑ | MU ↑ | MTRR ↓ |
|---|---|---|---|
| oracle_retrain | 0.641 | 0.647 | — |
| GradDiff | 0.000 | 0.671 | — |
| SimNPO | 0.523 | 0.637 | — |
| NPO | TBD | — | — |
| MT-SimNPO mw=0.5 | 0.524 | — | 0.708 |
| MT-SimNPO mw=1.0 | 0.527 ± 0.004 | — | 0.698 ± 0.034 |
| MT-SimNPO mw=2.0 | 0.524 | — | 0.615 |

FTR = Forget Truth Ratio, MU = Model Utility, MTRR = Multi-Turn Recovery Rate.
