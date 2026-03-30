# MT-SimNPO: Multi-Turn Robust Machine Unlearning

MT-SimNPO extends SimNPO-based machine unlearning to remain robust against multi-turn adversarial attacks — conversation strategies that extract forget-set knowledge from a model that appears unlearned under single-turn evaluation.

---

## Method

Standard unlearning methods (NPO, SimNPO) suppress direct recall but stay vulnerable to multi-turn probing: an adversary can prime the model with context, invoke persona shifts, or use chain-of-thought decomposition to recover memorized knowledge across turns.

MT-SimNPO trains jointly on the original forget set and a synthetically generated multi-turn forget set, so the model resists adversarial recovery through conversation.

**Loss function:**

```
L = γ · L_SimNPO(D_forget)
  + γ · λ · L_SimNPO(D_mt_forget)
  + α · L_NLL(D_retain)
```

`λ` (`mt_weight`) controls the multi-turn loss weight. Default hyperparameters: `β=4.5`, `γ=0.125`, `α=1.0`, `λ=1.0`.

**Training attack types** (used to build `D_mt_forget`): `priming`, `self_correction`, `persona_switch`

**Transfer attack types** (unseen at training, used for MTRR evaluation): `cot_decomposition`, `triangulation`

**Stress attack** (held-out generalization): `crescendo`

---

## Results

All results on TOFU forget10/retain90, base model Llama-3.1-8B-Instruct.

**Metrics:**
- FTR (Forget Truth Ratio): higher is better, oracle target ~0.641
- MU (Model Utility): retain-set performance, higher is better
- Transfer MTRR (Multi-Turn Recovery Rate on unseen attack types): fraction of examples where the model leaks forget-set information; lower is better

| Method | FTR | MU | Transfer MTRR |
|---|---|---|---|
| oracle_retain90 | 0.641 | 0.647 | — |
| NPO | 0.508 | 0.637 | 0.540 |
| SimNPO | 0.523 | 0.637 | 0.883 |
| MT-SimNPO mw=0.5 | 0.525 | 0.640 | 0.708 |
| **MT-SimNPO mw=1.0** (3 seeds) | **0.530 ± 0.003** | **0.644 ± 0.001** | **0.690** |
| MT-SimNPO mw=2.0 | 0.524 | 0.641 | 0.615 |

MT-SimNPO (mw=1.0) achieves **0% MTRR on crescendo stress attacks**, demonstrating generalization to unseen adversarial strategies. SimNPO's high transfer MTRR (0.883) shows that single-turn unlearning leaves models exploitable through conversation.

---

## Repo Structure

```
configs/experiment/unlearn/tofu/
  default.yaml            # baseline hyperparameters (NPO, SimNPO)
  mt_simnpo_8b.yaml       # MT-SimNPO hyperparameters
data/
  mt_train.jsonl          # 1,200 training conversations
  mt_val.jsonl            # 1,200 validation conversations
  mt_test.jsonl           # 800 held-out test conversations
scripts/
  generate_mt_dataset.py  # synthesize multi-turn adversarial data (requires OpenAI key)
  run_baselines.sh        # train NPO / SimNPO baselines
  run_mt_simnpo.sh        # train MT-SimNPO (args: mt_weight seed)
  run_vulnerability_demo.sh
src/
  train.py                # training entry point
  eval_runner.py          # TOFU benchmark evaluation
  trainer/unlearn/        # NPO, SimNPO, MT-SimNPO trainer implementations
  eval/
    mt_eval.py            # multi-turn evaluation harness
    mt_metrics.py         # NEM, semantic similarity, leakage detection
```

---

## Quick Start

### 1. Install

```bash
pip install -e ".[tofu]"
pip install bitsandbytes sentence-transformers spacy rouge-score openai python-Levenshtein
python -m spacy download en_core_web_sm
```

### 2. Download base checkpoints

```bash
huggingface-cli download open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
    --local-dir /workspace/checkpoints/tofu_8b_full

huggingface-cli download open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90 \
    --local-dir /workspace/checkpoints/tofu_8b_retain90
```

The multi-turn dataset (`data/mt_*.jsonl`) is already committed. To regenerate:

```bash
export OPENAI_API_KEY=<key>
python scripts/generate_mt_dataset.py
```

### 3. Train

```bash
# Baselines (~3 hrs on A40 48 GB)
bash scripts/run_baselines.sh 0        # seed=0

# MT-SimNPO (3 seeds at mw=1.0)
bash scripts/run_mt_simnpo.sh 1.0 0
bash scripts/run_mt_simnpo.sh 1.0 1
bash scripts/run_mt_simnpo.sh 1.0 2
```

### 4. Evaluate

```bash
# TOFU benchmark (FTR, MU)
PYTHONPATH=src python src/eval_runner.py --config-name=eval.yaml \
    experiment=eval/tofu/default \
    model=Llama-3.1-8B-Instruct \
    model.model_args.pretrained_model_name_or_path=<checkpoint_path> \
    model.tokenizer_args.pretrained_model_name_or_path=/workspace/checkpoints/tofu_8b_full \
    retain_logs_path=saves/eval/oracle_retain90/TOFU_EVAL.json \
    task_name=<run_id>

# MT-Eval (MTRR)
PYTHONPATH=src python src/eval/mt_eval.py \
    --checkpoint <checkpoint_path> \
    --mt_test_path data/mt_test.jsonl \
    --split test \
    --output results/<run_id>/mt_test.json
```

### 5. Pre-trained checkpoints

```bash
huggingface-cli download harishm17/mt-unlearning-checkpoints \
    --include "checkpoints/MTSimNPO_mw1.0_seed0/*" \
    --local-dir /workspace/checkpoints \
    --repo-type dataset
```

Eval results and training logs: `harishm17/mt-unlearning-results`

---

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for full experiment details, all hyperparameters, oracle baseline setup, and the vulnerability demo.
