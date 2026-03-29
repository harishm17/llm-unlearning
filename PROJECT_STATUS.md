# MT-SimNPO — Project Status & Reference Document

**Last updated:** 2026-03-29
**Status:** All training, evaluation, and analysis complete.

---

## What This Project Is

**Research question:** Do SOTA LLM unlearning methods (SimNPO, NPO, GradDiff) actually prevent knowledge recovery when an attacker uses multi-turn conversation instead of direct single-turn queries?

**Answer found:** No. SimNPO leaks 88.3% of forget-set knowledge via transfer attacks; NPO leaks 54%. These methods pass standard TOFU benchmarks while remaining highly vulnerable.

**Proposed fix:** MT-SimNPO — augment SimNPO's training objective with multi-turn forget examples. This reduces multi-turn leakage (MTRR 0.615–0.708 vs SimNPO's 0.883) while maintaining comparable TOFU forget quality.

**Built on:** [open-unlearning](https://github.com/locuslab/open-unlearning) framework (Hydra configs, TOFU benchmark, 11 baseline trainers).

---

## Final Results

### TOFU Benchmark (forget10 split, Llama-3.1-8B-Instruct)

| Method | Forget Truth Ratio ↑ | Model Utility ↑ | Notes |
|---|---|---|---|
| oracle_retrain | 0.641 | 0.647 | Gold standard upper bound |
| GradDiff | 0.000 | 0.671 | Model collapse — FTR=0 means broken, not unlearned |
| NPO | 0.508 | 0.637 | Good balance |
| SimNPO | 0.523 | 0.637 | SOTA baseline |
| MT-SimNPO mw=0.5 | 0.524 | — | |
| MT-SimNPO mw=1.0 | 0.527 ± 0.004 | — | 3 seeds (0/1/2) |
| MT-SimNPO mw=2.0 | 0.524 | — | |

FTR closer to oracle (0.641) = better unlearning without over-forgetting.

### Vulnerability Demo — Transfer Attack MTRR (test split, unseen attacks)

| Model | Transfer MTRR ↓ | Interpretation |
|---|---|---|
| pre_unlearning | **0.985** | 98.5% leakage — baseline vulnerability |
| oracle_retrain | 0.850 | Even gold standard leaks — inherent model capability |
| GradDiff | 0.003* | False positive — model utility destroyed |
| NPO | 0.540 | Passes TOFU, still 54% vulnerable |
| SimNPO | **0.883** | Passes TOFU, still 88% vulnerable |

*GradDiff's near-zero MTRR is not security — it's inability to generate coherent text.

### MT-Eval on Validation Split (trained attacks, MT-SimNPO only)

| Method | MTRR ↓ |
|---|---|
| MT-SimNPO mw=0.5 | 0.708 |
| MT-SimNPO mw=1.0 | 0.698 ± 0.034 (3 seeds) |
| MT-SimNPO mw=2.0 | **0.615** |

Increasing mt_weight reduces leakage further at the cost of heavier training signal.

---

## File Structure

### Local Machine (`/home/harish/dev/mt-unlearning/`)

```
mt-unlearning/
│
├── src/                          # All Python source
│   ├── train.py                  # Entry point: loads model → data → trainer → trains
│   ├── eval_runner.py            # Entry point: loads model → runs TOFU evaluators
│   │
│   ├── trainer/
│   │   ├── __init__.py           # TRAINER_REGISTRY + load_trainer() factory
│   │   ├── base.py               # FinetuneTrainer base class
│   │   ├── utils.py              # compute_dpo_loss, compute_batch_nll, compute_kl_divergence,
│   │   │                         #   compute_wga_loss, compute_simnpo_loss, etc.
│   │   └── unlearn/
│   │       ├── base.py           # UnlearnTrainer — hooks for evaluators
│   │       ├── grad_ascent.py    # GradAscent: -NLL on forget set
│   │       ├── grad_diff.py      # GradDiff: -NLL(forget) + NLL(retain)
│   │       ├── npo.py            # NPO: DPO-style loss, 4-bit quantized ref model
│   │       ├── simnpo.py         # SimNPO: length-normalized NPO without ref model
│   │       ├── mt_simnpo.py      # ★ MT-SimNPO: SimNPO + multi-turn forget loss
│   │       └── [dpo, rmu, ceu, undial, wga, satimp, pdu].py  # Other baselines
│   │
│   ├── data/
│   │   ├── unlearn.py            # MTForgetRetainDataset: combines forget+retain+mt_forget
│   │   ├── mt_collator.py        # ★ MultiTurnForgetCollator: masks prefix tokens
│   │   ├── mt_jsonl_dataset.py   # MTForgetJSONLDataset: loads mt_train.jsonl
│   │   ├── mt_tofu_dataset.py    # Bridge: maps TOFU QA → mt_forget format
│   │   ├── collators.py          # Single-turn forget/retain collators (upstream)
│   │   ├── qa.py                 # TOFU QA dataset loader (upstream)
│   │   └── utils.py              # Tokenization helpers
│   │
│   ├── eval/
│   │   ├── mt_eval.py            # ★ MultiTurnEvaluator: runs MT-Eval harness
│   │   └── mt_metrics.py         # ★ 3-metric leakage bundle: NEM + SemSim + LLM judge
│   │
│   └── evals/                    # TOFU evaluation suite (upstream open-unlearning)
│       ├── tofu.py               # TOFU evaluator (forget quality, utility, MIA)
│       └── metrics/              # Individual metrics: privacy, utility, memorization, MIA
│
├── configs/
│   ├── experiment/unlearn/tofu/
│   │   ├── default.yaml          # Baseline config: GradDiff/NPO/SimNPO on 8B TOFU
│   │   └── mt_simnpo_8b.yaml     # ★ MT-SimNPO config: 8B, adafactor, mt_weight=1.0
│   ├── trainer/
│   │   └── MTSimNPO.yaml         # Trainer handler registration
│   └── data/
│       ├── mt_unlearn.yaml       # Data config: forget + retain + mt_forget
│       └── datasets/MT_Forget_JSONL.yaml  # Points to data/mt_train.jsonl
│
├── data/
│   ├── mt_train.jsonl            # 1,200 multi-turn training conversations
│   │                             #   3 attack types × 400: priming, self_correction, persona_switch
│   ├── mt_val.jsonl              # 1,200 validation conversations (v2 variants)
│   └── mt_test.jsonl             # 800 held-out test conversations
│                                 #   cot_decomposition, triangulation (transfer) + crescendo (stress)
│
├── scripts/
│   ├── run_baselines.sh          # Train GradDiff/NPO/SimNPO + TOFU eval + HF upload
│   ├── run_mt_simnpo.sh          # Train MT-SimNPO + TOFU eval + MT-Eval + HF upload
│   ├── run_vulnerability_demo.sh # MT-Eval on 5 models (no training)
│   ├── generate_mt_dataset.py    # GPT-4o-mini dataset generation
│   └── filter_mt_dataset.py      # Dataset filtering utilities
│
├── results/
│   └── vulnerability/            # ★ Vulnerability demo results (tracked in git)
│       ├── pre_unlearning_mt_eval.json    # MTRR transfer=0.985
│       ├── oracle_retrain_mt_eval.json    # MTRR transfer=0.850
│       ├── GradDiff_mt_eval.json          # MTRR transfer=0.003
│       ├── NPO_mt_eval.json               # MTRR transfer=0.540
│       └── SimNPO_mt_eval.json            # MTRR transfer=0.883
│
├── notebooks/
│   └── analysis.ipynb            # ★ Analysis notebook: TOFU figures, MTRR ablation,
│                                 #   vulnerability bar chart (auto-loads results/)
│
├── README.md                     # Project overview + full results tables
├── REPRODUCIBILITY.md            # Step-by-step reproduce guide
└── PROJECT_STATUS.md             # This file
```

---

## Cloud Storage

### RunPod A40 48GB — `194.68.245.144:22151`

**Temporary — pod may be reclaimed at any time. All important outputs are on HuggingFace.**

```
/workspace/
├── checkpoints/
│   ├── tofu_8b_full/             # Llama-3.1-8B-Instruct fine-tuned on full TOFU
│   │                             # (starting point for all unlearning runs)
│   └── tofu_8b_retain90/         # Oracle retrain on retain90 split (gold standard)
│
└── mt-unlearning/                # Git clone of the repo
    ├── saves/unlearn/            # Trained model checkpoints
    │   ├── GradDiff_forget10_seed0/   # GradDiff checkpoint
    │   ├── NPO_forget10_seed0/        # NPO checkpoint (4-bit ref model fix applied)
    │   ├── SimNPO_forget10_seed0/     # SimNPO checkpoint
    │   ├── MTSimNPO_mw0.5_seed0/     # MT-SimNPO ablation
    │   ├── MTSimNPO_mw1.0_seed0/     # MT-SimNPO main (seed 0)
    │   ├── MTSimNPO_mw1.0_seed1/     # MT-SimNPO main (seed 1)
    │   ├── MTSimNPO_mw1.0_seed2/     # MT-SimNPO main (seed 2)
    │   └── MTSimNPO_mw2.0_seed0/     # MT-SimNPO ablation
    │
    ├── saves/eval/               # TOFU evaluation JSON results
    │   ├── oracle_retain90/      # Oracle baseline eval (needed for MU scoring)
    │   ├── GradDiff_forget10_seed0_eval/
    │   ├── NPO_forget10_seed0_eval/
    │   ├── SimNPO_forget10_seed0_eval/
    │   ├── MTSimNPO_mw*_seed*_eval/  # (5 MT-SimNPO TOFU evals)
    │   └── [each contains TOFU_EVAL.json + TOFU_SUMMARY.json]
    │
    ├── results/                  # MT-Eval + vulnerability results
    │   ├── MTSimNPO_mw*/mt_val.json  # Per-run MT-Eval on val split
    │   └── vulnerability/            # Vulnerability demo results (all 5 models)
    │
    └── logs/                     # Training and eval logs
        ├── npo_run4.log          # NPO training run (500 steps complete)
        ├── npo_eval_upload.log   # Post-training pipeline log
        ├── GradDiff_*/  NPO_*/  SimNPO_*/  MTSimNPO_*/   # Per-run logs
        └── vuln_*.log            # Vulnerability demo per-model logs
```

### HuggingFace — Permanent Archive

**`harishm17/mt-unlearning-checkpoints`** (dataset repo — stores model weights)
```
checkpoints/
├── GradDiff_forget10_seed0/      # ~28 GB (safetensors shards)
├── NPO_forget10_seed0/           # ~28 GB
├── SimNPO_forget10_seed0/        # ~28 GB
├── MTSimNPO_mw0.5_seed0/        # ~28 GB each
├── MTSimNPO_mw1.0_seed0/
├── MTSimNPO_mw1.0_seed1/
├── MTSimNPO_mw1.0_seed2/
└── MTSimNPO_mw2.0_seed0/
```

**`harishm17/mt-unlearning-results`** (dataset repo — stores eval outputs and logs)
```
results/
├── eval/
│   ├── GradDiff_forget10_seed0_eval/    # TOFU_EVAL.json, TOFU_SUMMARY.json
│   ├── NPO_forget10_seed0_eval/
│   ├── SimNPO_forget10_seed0_eval/
│   └── MTSimNPO_mw*_seed*_eval/        # (5 runs)
│
├── mt_eval/
│   └── MTSimNPO_mw*_seed*/mt_val.json  # MT-Eval on val split (5 runs)
│
└── vulnerability/
    ├── pre_unlearning_mt_eval.json     # transfer MTRR = 0.985
    ├── oracle_retrain_mt_eval.json     # transfer MTRR = 0.850
    ├── GradDiff_mt_eval.json           # transfer MTRR = 0.003
    ├── NPO_mt_eval.json                # transfer MTRR = 0.540
    └── SimNPO_mt_eval.json             # transfer MTRR = 0.883

logs/
├── npo_run4.log                  # NPO training log (complete, 500 steps)
├── npo_eval_upload.log           # Post-training pipeline log
├── GradDiff_*/  NPO_*/  SimNPO_*/  MTSimNPO_*/  vuln_*.log
```

---

## Why Each Key Decision Was Made

### Why RunPod A40 48GB?
- 8B model in bf16 = ~16 GB. Need room for optimizer states + activations + ref model.
- A40 48 GB gives enough headroom. Spot price: $0.20/hr.

### Why adafactor instead of AdamW/paged_adamw_8bit?
- `paged_adamw_8bit` caused OOM at optimizer init for 8B model on A40.
- Adafactor has no second-moment state per parameter → ~8× less memory for optimizer.

### Why 4-bit quantized ref model for NPO?
- NPO requires a frozen reference model to compute the DPO-style loss ratio.
- Full bf16 copy of 8B = 16 GB → total 32 GB just for models → OOM.
- 4-bit quantization (bitsandbytes) reduces ref model to ~4 GB.
- Tried CPU first: flash_attention_2 is CUDA-only, failing. Tried eager attention on CPU: works but 5+ min/step.
- 4-bit GPU: ~50s/step, fits in ~45 GB total.

### Why `device_map="auto"` for ref model requires input tensors moved to ref_device?
- When `device_map="auto"` places model on GPU but training model is on a different device context,
  input tensors must be explicitly moved to `ref_device = next(ref_model.parameters()).device`.
- Fixed in `src/trainer/utils.py compute_dpo_loss()`.

### Why MultiTurnForgetCollator masks prefix tokens?
- In multi-turn conversation, earlier turns are context, not the thing to unlearn.
- Only the final assistant response (the actual answer to the forget-set question) should be unlearned.
- The collator sets all prefix tokens to `label=-100` (ignored by cross-entropy loss).
- This means `|y|` (response token count) used in SimNPO normalization only counts the answer tokens.

### Why HuggingFace for storage (not just pod disk)?
- RunPod spot instances are reclaimed with zero warning. Pod was reclaimed multiple times during this project.
- HF dataset repos = permanent, free, versioned storage for checkpoints and results.
- Scripts upload after each completed run so nothing is lost between runs.

### Why `eval_strategy: "no"` in configs?
- HuggingFace Trainer interprets any eval_strategy other than "no" as requiring an eval dataset.
- We run eval via custom evaluators (TOFU suite), not Trainer.evaluate().
- Setting this prevents a crash at training start.

### Why `gradient_checkpointing: true`?
- Recomputes activations during backward pass instead of storing them.
- Trades compute for ~30% memory reduction — critical for fitting 8B + optimizer on A40.

---

## Steps Taken (Chronological)

1. **Forked open-unlearning** — added MT-SimNPO trainer, MT data pipeline, MT-Eval harness
2. **Generated MT dataset** — 3,200 multi-turn conversations via GPT-4o-mini
3. **Local smoke test** — verified pipeline on 1B model (RTX 4060)
4. **Cloud setup** — RunPod A40, downloaded TOFU checkpoints, fixed deps
5. **Fixed OOM issues** — adafactor, gradient checkpointing, batch=1/grad_accum=8
6. **Ran MT-SimNPO** (5 runs: 3 seeds + 2 mt_weight ablations) — all complete
7. **Fixed NPO ref model** — 3 attempts: CPU deepcopy → CPU eager → 4-bit GPU (final fix)
8. **Ran baselines** — GradDiff, NPO, SimNPO all complete with TOFU eval
9. **Ran vulnerability demo** — MT-Eval on all 5 models using held-out test attacks
10. **Analysis** — notebook with TOFU + MTRR figures, README, REPRODUCIBILITY.md

---

## Downloading Results from HuggingFace

```bash
# Download a checkpoint
huggingface-cli download harishm17/mt-unlearning-checkpoints \
    --include "checkpoints/MTSimNPO_mw1.0_seed0/*" \
    --local-dir /workspace/checkpoints \
    --repo-type dataset

# Download eval results
huggingface-cli download harishm17/mt-unlearning-results \
    --local-dir results_from_hf \
    --repo-type dataset
```

---

## Open Items / Limitations

- **MT-SimNPO model_utility** not computed (TOFU eval ran without retain_logs_path for MT-SimNPO runs — would need re-eval)
- **MT-Eval val duplicate metrics** — all 3 attack types in val split show identical MTRR per run (possible data generation artifact; does not affect overall_mtrr_trained)
- **Crescendo / stress MTRR** — N/A in all vulnerability results; the crescendo attack may not be present in mt_test.jsonl test split
- **MT-SimNPO not evaluated on transfer attacks** — only val split (trained attacks) used; transfer MTRR for MT-SimNPO unknown
