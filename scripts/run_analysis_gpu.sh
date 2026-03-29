#!/bin/bash
# Full analysis pass on RunPod — no training, eval only.
#
# Runs:
#   1. Full TOFU eval (with model_utility) for all MT-SimNPO variants
#   2. Transfer MT-Eval (test split, per-example) for MT-SimNPO mw=0.5/1.0/2.0
#   3. Val MT-Eval for mw=1.0 seed0 and seed1 (missing from HF)
#
# Prerequisites (on RunPod):
#   - git pull to get latest scripts
#   - HF_TOKEN set (or exported before calling this script)
#   - oracle_retain90 checkpoint at /workspace/checkpoints/tofu_8b_retain90
#   - TOFU full tokenizer at /workspace/checkpoints/tofu_8b_full
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   bash scripts/run_analysis_gpu.sh
#
# Estimated time: ~2.5 hrs on A40 48GB (each TOFU eval ~15min, each MT-eval ~25min)

set -e
PYTHONPATH=src

HF_CKPT_REPO=harishm17/mt-unlearning-checkpoints
HF_RESULTS_REPO=harishm17/mt-unlearning-results
CKPT_BASE=/workspace/checkpoints/tofu_8b_full
LOG_DIR=logs
mkdir -p ${LOG_DIR}

# Oracle retain eval path — required for model_utility computation
RETAIN_LOGS=saves/eval/oracle_retain90/TOFU_EVAL.json

# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
_hf_upload() {
    local local_path="$1" remote_path="$2" repo="$3"
    echo "[upload] ${local_path} -> ${repo}/${remote_path}"
    huggingface-cli upload ${repo} ${local_path} ${remote_path} \
        --repo-type dataset --token ${HF_TOKEN} 2>&1 | tail -3 \
        || echo "[WARN] HF upload failed for ${remote_path}"
}

_download_ckpt() {
    local run_id="$1"
    local ckpt_dir="saves/unlearn/${run_id}"
    if [ -f "${ckpt_dir}/config.json" ]; then
        echo "[skip] checkpoint already exists: ${ckpt_dir}"
        return 0
    fi
    echo "[download] ${run_id} from HF..."
    huggingface-cli download ${HF_CKPT_REPO} \
        --include "checkpoints/${run_id}/*" \
        --local-dir /workspace/checkpoints \
        --repo-type dataset --token ${HF_TOKEN}
    # Symlink so paths match what scripts expect
    mkdir -p saves/unlearn
    ln -sfn /workspace/checkpoints/checkpoints/${run_id} ${ckpt_dir}
    echo "[downloaded] ${ckpt_dir}"
}

_tofu_eval() {
    local run_id="$1"
    local ckpt_dir="saves/unlearn/${run_id}"
    local eval_dir="saves/eval/${run_id}_eval"
    local log="${LOG_DIR}/${run_id}_eval_full.log"

    if [ -f "${eval_dir}/TOFU_SUMMARY.json" ]; then
        # Check if model_utility exists in the summary (re-run if missing)
        if /usr/bin/python3 -c "
import json, sys
d = json.load(open('${eval_dir}/TOFU_EVAL.json'))
sys.exit(0 if 'model_utility' in d else 1)
" 2>/dev/null; then
            echo "[skip] TOFU eval already complete with model_utility: ${run_id}"
            return 0
        fi
        echo "[rerun] TOFU eval missing model_utility: ${run_id}"
        rm -rf "${eval_dir}"
    fi

    echo "=== TOFU eval: ${run_id} ==="
    env PYTHONPATH=src python src/eval_runner.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model=Llama-3.1-8B-Instruct \
        model.model_args.pretrained_model_name_or_path=${ckpt_dir} \
        model.tokenizer_args.pretrained_model_name_or_path=${CKPT_BASE} \
        retain_logs_path=${RETAIN_LOGS} \
        task_name=${run_id}_eval \
        2>&1 | tee ${log} || echo "[WARN] TOFU eval failed for ${run_id}"

    _hf_upload ${eval_dir} results/eval/${run_id}_eval ${HF_RESULTS_REPO}
    _hf_upload ${log} logs/${run_id}_eval_full.log ${HF_RESULTS_REPO}
}

_mt_eval() {
    local run_id="$1"
    local split="$2"         # val or test
    local ckpt_dir="saves/unlearn/${run_id}"
    local out_dir="results/${run_id}"
    local out_file="${out_dir}/mt_${split}.json"
    local examples_file="${out_dir}/mt_${split}_examples.jsonl"
    local log="${LOG_DIR}/${run_id}_mt_${split}.log"
    mkdir -p ${out_dir}

    if [ -f "${out_file}" ]; then
        echo "[skip] MT-eval ${split} already done: ${run_id}"
        return 0
    fi

    echo "=== MT-Eval (${split} split): ${run_id} ==="
    env PYTHONPATH=src python src/eval/mt_eval.py \
        --checkpoint ${ckpt_dir} \
        --mt_test_path data/mt_${split}.jsonl \
        --split ${split} \
        --output ${out_file} \
        --examples_output ${examples_file} \
        2>&1 | tee ${log} || echo "[WARN] MT-eval ${split} failed for ${run_id}"

    _hf_upload ${out_dir} results/mt_eval/${run_id} ${HF_RESULTS_REPO}
    _hf_upload ${log} logs/${run_id}_mt_${split}.log ${HF_RESULTS_REPO}
}

# --------------------------------------------------------------------------
# Step 0: Ensure oracle retain eval exists (required for model_utility)
# --------------------------------------------------------------------------
if [ ! -f "${RETAIN_LOGS}" ]; then
    echo "=== Running oracle retain90 eval (needed for model_utility) ==="
    env PYTHONPATH=src python src/eval_runner.py --config-name=eval.yaml \
        experiment=eval/tofu/default \
        model=Llama-3.1-8B-Instruct \
        model.model_args.pretrained_model_name_or_path=/workspace/checkpoints/tofu_8b_retain90 \
        model.tokenizer_args.pretrained_model_name_or_path=${CKPT_BASE} \
        retain_logs_path=null \
        task_name=oracle_retain90 \
        2>&1 | tee ${LOG_DIR}/oracle_retain90_full.log
    echo "=== Oracle retain eval done ==="
else
    echo "[skip] oracle retain eval already exists"
fi

# --------------------------------------------------------------------------
# Step 1: Download MT-SimNPO checkpoints from HF
# --------------------------------------------------------------------------
echo "=== Downloading MT-SimNPO checkpoints ==="
for run_id in MTSimNPO_mw0.5_seed0 MTSimNPO_mw1.0_seed0 MTSimNPO_mw1.0_seed1 MTSimNPO_mw2.0_seed0; do
    _download_ckpt ${run_id}
done

# --------------------------------------------------------------------------
# Step 2: Full TOFU eval (with model_utility) for all MT-SimNPO variants
# --------------------------------------------------------------------------
echo "=== Full TOFU eval for MT-SimNPO variants ==="
for run_id in MTSimNPO_mw0.5_seed0 MTSimNPO_mw1.0_seed0 MTSimNPO_mw1.0_seed1 MTSimNPO_mw2.0_seed0; do
    _tofu_eval ${run_id}
done

# --------------------------------------------------------------------------
# Step 3: Transfer MT-Eval (test split) for key MT-SimNPO variants
# --------------------------------------------------------------------------
echo "=== Transfer MT-Eval (test split) ==="
for run_id in MTSimNPO_mw0.5_seed0 MTSimNPO_mw1.0_seed0 MTSimNPO_mw2.0_seed0; do
    _mt_eval ${run_id} test
done

# --------------------------------------------------------------------------
# Step 4: Val MT-Eval for mw=1.0 seeds 0 and 1 (missing from HF)
# --------------------------------------------------------------------------
echo "=== Val MT-Eval for mw=1.0 seeds 0,1 ==="
for run_id in MTSimNPO_mw1.0_seed0 MTSimNPO_mw1.0_seed1; do
    _mt_eval ${run_id} val
done

# --------------------------------------------------------------------------
# Done
# --------------------------------------------------------------------------
echo ""
echo "=== All analyses complete ==="
echo "  TOFU eval results: ${HF_RESULTS_REPO}/results/eval/"
echo "  MT-eval results:   ${HF_RESULTS_REPO}/results/mt_eval/"
echo ""
echo "Key metrics to check:"
for run_id in MTSimNPO_mw0.5_seed0 MTSimNPO_mw1.0_seed0 MTSimNPO_mw1.0_seed1 MTSimNPO_mw2.0_seed0; do
    eval_json="saves/eval/${run_id}_eval/TOFU_EVAL.json"
    if [ -f "${eval_json}" ]; then
        mu=$(/usr/bin/python3 -c "import json; d=json.load(open('${eval_json}')); print(f\"{d.get('model_utility','N/A'):.4f}\" if isinstance(d.get('model_utility'), float) else 'N/A')" 2>/dev/null)
        ftr=$(/usr/bin/python3 -c "import json; d=json.load(open('${eval_json}')); print(f\"{d.get('forget_truth_ratio','N/A'):.4f}\" if isinstance(d.get('forget_truth_ratio'), float) else 'N/A')" 2>/dev/null)
        echo "  ${run_id}: FTR=${ftr} MU=${mu}"
    fi
done
