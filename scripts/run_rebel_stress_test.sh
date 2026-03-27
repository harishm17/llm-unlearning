#!/bin/bash
# Run REBEL evolutionary attack stress test against a checkpoint.
# Requires: separate venv with vLLM + REBEL repo cloned at /workspace/REBEL
# Usage: bash scripts/run_rebel_stress_test.sh [CHECKPOINT_PATH] [OUTPUT_PATH]
#
# Run against SimNPO (baseline):
#   bash scripts/run_rebel_stress_test.sh \
#       /workspace/checkpoints/tofu_8b_unlearn_SimNPO_forget10 \
#       results/vulnerability/rebel_simnpo.json
#
# Run against MT-SimNPO (after Task 10):
#   bash scripts/run_rebel_stress_test.sh \
#       saves/unlearn/MTSimNPO_mw1.0_seed0/checkpoint-final \
#       results/main/rebel_mt_simnpo.json
set -e

TARGET_MODEL=${1:-/workspace/checkpoints/tofu_8b_unlearn_SimNPO_forget10}
OUTPUT=${2:-results/vulnerability/rebel_simnpo.json}
REBEL_DIR=/workspace/REBEL

if [ ! -d "${REBEL_DIR}" ]; then
    echo "Cloning REBEL..."
    git clone https://github.com/patryk-rybak/REBEL ${REBEL_DIR}
fi

mkdir -p "$(dirname ${OUTPUT})"

echo "=== REBEL stress test: ${TARGET_MODEL} ==="
cd ${REBEL_DIR}
python run_rebel.py \
    --target-model "${TARGET_MODEL}" \
    --hacker-model Qwen/Qwen2.5-7B-Instruct \
    --judge-model Qwen/Qwen2.5-7B-Instruct \
    --dataset tofu \
    --forget-split forget10 \
    --mutations-list 1500,80,50,40,40 \
    --top-k-list 20,12,8,5,3 \
    --output "${OUTPUT}"

echo "REBEL results saved to ${OUTPUT}"
python3 -c "
import json
d = json.load(open('${OUTPUT}'))
asr = d.get('asr', d.get('attack_success_rate', 'N/A'))
print(f'ASR: {asr}  (expected ~0.55-0.65 for SimNPO)')
"
