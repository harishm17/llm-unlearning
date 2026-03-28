#!/bin/bash
# Run MT-SimNPO training + TOFU eval + MT-Eval on RunPod.
# Usage: bash scripts/run_mt_simnpo.sh [SEED] [MT_WEIGHT] [MT_DATA_PATH]
# Hardware: A40 48GB for core runs and ablations.
#
# Resilience: training is fatal (abort on failure); eval steps are non-fatal
# so a failed eval does not lose the trained checkpoint.
# Rerunnable: skips training if checkpoint already exists.

SEED=${1:-0}
MT_WEIGHT=${2:-1.0}
MT_DATA=${3:-data/mt_train.jsonl}
CKPT_BASE=/workspace/checkpoints/tofu_8b_full
RETAIN_LOGS=saves/eval/oracle_retain90/TOFU_EVAL.json
RUN_ID="MTSimNPO_mw${MT_WEIGHT}_seed${SEED}"
CKPT_OUT=saves/unlearn/${RUN_ID}
LOG_DIR=logs
HF_CKPT_REPO=harishm17/mt-unlearning-checkpoints
HF_RESULTS_REPO=harishm17/mt-unlearning-results
mkdir -p ${LOG_DIR} results/${RUN_ID}

_run_eval() {
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

# Skip training if checkpoint already exists
if [ -f "${CKPT_OUT}/config.json" ]; then
    echo "=== Skipping training ${RUN_ID} (checkpoint exists) ==="
else
    echo "=== Training ${RUN_ID} (mt_data=${MT_DATA}) ==="
    set -e
    PYTHONPATH=src python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/mt_simnpo_8b \
        model.model_args.pretrained_model_name_or_path=${CKPT_BASE} \
        model.tokenizer_args.pretrained_model_name_or_path=${CKPT_BASE} \
        trainer.method_args.mt_weight=${MT_WEIGHT} \
        trainer.args.seed=${SEED} \
        trainer.args.output_dir=${CKPT_OUT} \
        "data.mt_forget.MT_Forget_JSONL.args.jsonl_path=${MT_DATA}" \
        task_name=${RUN_ID} \
        2>&1 | tee ${LOG_DIR}/${RUN_ID}_train.log
    set +e
    echo "=== Training complete: ${RUN_ID} ==="
    echo "=== Uploading checkpoint: ${RUN_ID} ==="
    _hf_upload ${CKPT_OUT} checkpoints/${RUN_ID} ${HF_CKPT_REPO}
fi

echo "=== Standard TOFU eval: ${RUN_ID} ==="
_run_eval ${LOG_DIR}/${RUN_ID}_eval.log \
    env PYTHONPATH=src python src/eval_runner.py --config-name=eval.yaml \
    experiment=eval/tofu/default \
    model=Llama-3.1-8B-Instruct \
    model.model_args.pretrained_model_name_or_path=${CKPT_OUT} \
    model.tokenizer_args.pretrained_model_name_or_path=${CKPT_BASE} \
    retain_logs_path=${RETAIN_LOGS} \
    task_name=${RUN_ID}_eval

echo "=== MT-Eval (val split — hyperparam selection): ${RUN_ID} ==="
_run_eval ${LOG_DIR}/${RUN_ID}_mt_eval.log \
    env PYTHONPATH=src python src/eval/mt_eval.py \
    --checkpoint ${CKPT_OUT} \
    --mt_test_path data/mt_val.jsonl \
    --split val \
    --output results/${RUN_ID}/mt_val.json

echo "=== Uploading results + logs: ${RUN_ID} ==="
_hf_upload saves/eval/${RUN_ID}_eval results/eval/${RUN_ID}_eval ${HF_RESULTS_REPO}
_hf_upload results/${RUN_ID} results/mt_eval/${RUN_ID} ${HF_RESULTS_REPO}
_hf_upload ${LOG_DIR}/${RUN_ID}_train.log logs/${RUN_ID}_train.log ${HF_RESULTS_REPO}
_hf_upload ${LOG_DIR}/${RUN_ID}_eval.log logs/${RUN_ID}_eval.log ${HF_RESULTS_REPO}
_hf_upload ${LOG_DIR}/${RUN_ID}_mt_eval.log logs/${RUN_ID}_mt_eval.log ${HF_RESULTS_REPO}

echo "=== Done: ${RUN_ID} ==="
echo "  Checkpoint:   ${HF_CKPT_REPO}/checkpoints/${RUN_ID}"
echo "  TOFU results: ${HF_RESULTS_REPO}/results/eval/${RUN_ID}_eval"
echo "  MT-Eval val:  ${HF_RESULTS_REPO}/results/mt_eval/${RUN_ID}"
