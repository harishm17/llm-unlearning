#!/bin/bash
# Run MT-SimNPO training + TOFU eval + MT-Eval on RunPod.
# Usage: bash scripts/run_mt_simnpo.sh [SEED] [MT_WEIGHT] [MT_DATA_PATH]
# Hardware: A100 40GB for core runs, A100 40GB for ablations.
set -e

SEED=${1:-0}
MT_WEIGHT=${2:-1.0}
MT_DATA=${3:-data/mt_train.jsonl}
CKPT_BASE=/workspace/checkpoints/tofu_8b_full
RETAIN_LOGS=saves/eval/oracle_retain90/TOFU_EVAL.json
RUN_ID="MTSimNPO_mw${MT_WEIGHT}_seed${SEED}"
LOG_DIR=logs
mkdir -p ${LOG_DIR} results/${RUN_ID}

echo "=== Training ${RUN_ID} (mt_data=${MT_DATA}) ==="
PYTHONPATH=src python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/mt_simnpo_8b \
    model.model_args.pretrained_model_name_or_path=${CKPT_BASE} \
    trainer.method_args.mt_weight=${MT_WEIGHT} \
    trainer.args.seed=${SEED} \
    trainer.args.output_dir=saves/unlearn/${RUN_ID} \
    "data.mt_forget.MT_Forget_JSONL.args.jsonl_path=${MT_DATA}" \
    task_name=${RUN_ID} \
    2>&1 | tee ${LOG_DIR}/${RUN_ID}_train.log

# Model saved directly to output_dir by train.py (no checkpoint-final subfolder)
CKPT_OUT=saves/unlearn/${RUN_ID}

echo "=== Standard TOFU eval: ${RUN_ID} ==="
PYTHONPATH=src python src/eval_runner.py --config-name=eval.yaml \
    experiment=eval/tofu/default \
    model=Llama-3.1-8B-Instruct \
    model.model_args.pretrained_model_name_or_path=${CKPT_OUT} \
    retain_logs_path=${RETAIN_LOGS} \
    task_name=${RUN_ID}_eval \
    2>&1 | tee ${LOG_DIR}/${RUN_ID}_eval.log

echo "=== MT-Eval (val split — hyperparam selection): ${RUN_ID} ==="
PYTHONPATH=src python src/eval/mt_eval.py \
    --checkpoint ${CKPT_OUT} \
    --mt_test_path data/mt_val.jsonl \
    --split val \
    --output results/${RUN_ID}/mt_val.json \
    2>&1 | tee ${LOG_DIR}/${RUN_ID}_mt_eval.log

echo "=== Done: ${RUN_ID} ==="
echo "Training log:  ${LOG_DIR}/${RUN_ID}_train.log"
echo "TOFU results:  saves/eval/${RUN_ID}_eval/TOFU_SUMMARY.json"
echo "MT-Eval val:   results/${RUN_ID}/mt_val.json"
