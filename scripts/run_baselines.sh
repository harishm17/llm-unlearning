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
HF_CKPT_REPO=harishm17/mt-unlearning-checkpoints
HF_RESULTS_REPO=harishm17/mt-unlearning-results
mkdir -p ${LOG_DIR} results saves/eval

_run_eval() {
    # Non-fatal eval wrapper — logs failure but does not abort the script.
    local log="$1"; shift
    echo "[eval] $@" | tee -a ${log}
    "$@" 2>&1 | tee -a ${log} || echo "[WARN] eval step failed — training results are still saved" | tee -a ${log}
}

_hf_upload() {
    local local_path="$1" remote_path="$2" repo="$3"
    echo "[upload] ${local_path} -> ${repo}/${remote_path}"
    huggingface-cli upload ${repo} ${local_path} ${remote_path} \
        --repo-type dataset 2>&1 | tail -2 \
        || echo "[WARN] HF upload failed for ${remote_path}"
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
        echo "=== Uploading checkpoint: ${RUN_ID} ==="
        _hf_upload ${CKPT_OUT}/checkpoint-0 checkpoints/${RUN_ID} ${HF_CKPT_REPO}
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

    echo "=== Uploading results + logs: ${RUN_ID} ==="
    _hf_upload saves/eval/${RUN_ID}_eval results/eval/${RUN_ID}_eval ${HF_RESULTS_REPO}
    _hf_upload ${LOG_DIR}/${RUN_ID}_train.log logs/${RUN_ID}_train.log ${HF_RESULTS_REPO}
    _hf_upload ${LOG_DIR}/${RUN_ID}_eval.log logs/${RUN_ID}_eval.log ${HF_RESULTS_REPO}

    echo "=== Done: ${RUN_ID} ==="
    echo "  Checkpoint: ${HF_CKPT_REPO}/checkpoints/${RUN_ID}"
    echo "  Eval:       ${HF_RESULTS_REPO}/results/eval/${RUN_ID}_eval"
done

echo ""
echo "All baselines complete."
echo "Training logs: ${LOG_DIR}/"
echo "TOFU results:  saves/eval/"
