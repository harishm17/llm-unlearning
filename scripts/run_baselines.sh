#!/bin/bash
# Run B1-B3 baselines (GradDiff, NPO, SimNPO) + TOFU eval on RunPod.
# Usage: bash scripts/run_baselines.sh [SEED]
# Hardware: A100 40GB ($1.04/hr). ~60-70 min per method, ~$4.50 total.
set -e

SEED=${1:-0}
CKPT_BASE=/workspace/checkpoints/tofu_8b_full
RETAIN_LOGS=saves/eval/oracle_retain90/TOFU_EVAL.json
LOG_DIR=logs
mkdir -p ${LOG_DIR} results

# Build oracle retain logs once (needed for MU scoring)
if [ ! -f "${RETAIN_LOGS}" ]; then
    echo "=== Building oracle retain90 eval logs ==="
    PYTHONPATH=src python src/eval_runner.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model=Llama-3.1-8B-Instruct \
        model.model_args.pretrained_model_name_or_path=/workspace/checkpoints/tofu_8b_retain90 \
        retain_logs_path=null \
        task_name=oracle_retain90 \
        2>&1 | tee ${LOG_DIR}/oracle_retain90.log
    # oracle saves to saves/eval/oracle_retain90/
    mkdir -p saves/eval/oracle_retain90
    cp saves/eval/oracle_retain90/TOFU_EVAL.json ${RETAIN_LOGS} 2>/dev/null || true
fi

for TRAINER in GradDiff NPO SimNPO; do
    RUN_ID="${TRAINER}_forget10_seed${SEED}"
    echo "=== Training ${RUN_ID} ==="
    PYTHONPATH=src python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        trainer=${TRAINER} \
        model=Llama-3.1-8B-Instruct \
        model.model_args.pretrained_model_name_or_path=${CKPT_BASE} \
        trainer.args.seed=${SEED} \
        trainer.args.output_dir=saves/unlearn/${RUN_ID} \
        task_name=${RUN_ID} \
        2>&1 | tee ${LOG_DIR}/${RUN_ID}_train.log

    # Model saved directly to output_dir by train.py (no checkpoint-final subfolder)
    CKPT_OUT=saves/unlearn/${RUN_ID}

    echo "=== Evaluating ${RUN_ID} ==="
    PYTHONPATH=src python src/eval_runner.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model=Llama-3.1-8B-Instruct \
        model.model_args.pretrained_model_name_or_path=${CKPT_OUT} \
        retain_logs_path=${RETAIN_LOGS} \
        task_name=${RUN_ID}_eval \
        2>&1 | tee ${LOG_DIR}/${RUN_ID}_eval.log

    echo "=== Done: ${RUN_ID} ==="
done

echo ""
echo "All baselines complete."
echo "Training logs: ${LOG_DIR}/"
echo "TOFU results:  saves/eval/"
