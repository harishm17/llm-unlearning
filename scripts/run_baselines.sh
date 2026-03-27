#!/bin/bash
# Run B1-B3 baselines (GradDiff, NPO, SimNPO) + TOFU eval on RunPod.
# Usage: bash scripts/run_baselines.sh [SEED]
# Hardware: A40 48GB ($0.20/hr spot). ~60-70 min per method, ~$4.50 total.
#
# Resilience: training steps are fatal (abort on failure); eval steps are
# non-fatal (logged and skipped on failure so remaining training still runs).

SEED=${1:-0}
CKPT_BASE=/workspace/checkpoints/tofu_8b_full
RETAIN_LOGS=saves/eval/oracle_retain90/TOFU_EVAL.json
LOG_DIR=logs
mkdir -p ${LOG_DIR} results saves/eval

_run_eval() {
    # Non-fatal eval wrapper — logs failure but does not abort the script.
    local log="$1"; shift
    echo "[eval] $@" | tee -a ${log}
    "$@" 2>&1 | tee -a ${log} || echo "[WARN] eval step failed — training results are still saved" | tee -a ${log}
}

# Build oracle retain logs once (needed for MU scoring)
if [ ! -f "${RETAIN_LOGS}" ]; then
    echo "=== Building oracle retain90 eval logs ==="
    _run_eval ${LOG_DIR}/oracle_retain90.log \
        env PYTHONPATH=src python src/eval_runner.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model=Llama-3.1-8B-Instruct \
        model.model_args.pretrained_model_name_or_path=/workspace/checkpoints/tofu_8b_retain90 \
        model.tokenizer_args.pretrained_model_name_or_path=/workspace/checkpoints/tofu_8b_retain90 \
        retain_logs_path=null \
        task_name=oracle_retain90
fi

for TRAINER in GradDiff NPO SimNPO; do
    RUN_ID="${TRAINER}_forget10_seed${SEED}"
    CKPT_OUT=saves/unlearn/${RUN_ID}

    # Skip training if checkpoint already exists (allows reruns after eval failure)
    if [ -f "${CKPT_OUT}/config.json" ]; then
        echo "=== Skipping training ${RUN_ID} (checkpoint exists) ==="
    else
        echo "=== Training ${RUN_ID} ==="
        # Training is fatal — if it fails we want to know immediately
        set -e
        PYTHONPATH=src python src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/tofu/default \
            trainer=${TRAINER} \
            model=Llama-3.1-8B-Instruct \
            model.model_args.pretrained_model_name_or_path=${CKPT_BASE} \
            model.tokenizer_args.pretrained_model_name_or_path=${CKPT_BASE} \
            trainer.args.seed=${SEED} \
            trainer.args.output_dir=${CKPT_OUT} \
            task_name=${RUN_ID} \
            2>&1 | tee ${LOG_DIR}/${RUN_ID}_train.log
        set +e
        echo "=== Training complete: ${RUN_ID} ==="
    fi

    echo "=== Evaluating ${RUN_ID} ==="
    _run_eval ${LOG_DIR}/${RUN_ID}_eval.log \
        env PYTHONPATH=src python src/eval_runner.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model=Llama-3.1-8B-Instruct \
        model.model_args.pretrained_model_name_or_path=${CKPT_OUT} \
        model.tokenizer_args.pretrained_model_name_or_path=${CKPT_BASE} \
        retain_logs_path=${RETAIN_LOGS} \
        task_name=${RUN_ID}_eval

    echo "=== Done: ${RUN_ID} ==="
    echo "  Checkpoint: ${CKPT_OUT}/config.json"
    echo "  Train log:  ${LOG_DIR}/${RUN_ID}_train.log"
    echo "  Eval log:   ${LOG_DIR}/${RUN_ID}_eval.log"
done

echo ""
echo "All baselines complete."
echo "Training logs: ${LOG_DIR}/"
echo "TOFU results:  saves/eval/"
