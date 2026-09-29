#!/bin/bash
# Phase 3 structure-weighted fine-tune, one arm per invocation.
#   usage: ARM=armA_control bash ./shell/train/phase3_arm.sh
#
# Arms (see configs/phase3_*.yaml):
#   armA_control    no structure weighting -- matched-budget baseline
#   armB_uniform5x  5x on the whole structure
#   armE_shell5x    5x on a 2px-out / 1px-in band around the boundary
set -euo pipefail

ARM="${ARM:?ARM must be set, e.g. ARM=armB_uniform5x}"
config="./configs/phase3_${ARM}.yaml"
[[ -f "$config" ]] || { echo "no such config: $config" >&2; exit 1; }

date_tag="241213"
HW="256"
plane="axial"
gpu_ids="0,1"
batch=16
ddim_eta=0.0

# Distinct exp_name per arm: a shared output dir would let one arm auto-resume
# from another's checkpoint and silently contaminate the comparison.
exp_name="${date_tag}_${HW}_BBDM_${plane}_DDIM_MR_p3_${ARM}"

# All three arms start from the same Tier 3 checkpoint with a fresh optimizer.
resume_model="/blue/neurology-dept/jlabasbas/new-fine/fine-tune_256/241213_256_BBDM_axial_DDIM_MR_tier3_morphmask/checkpoint/top_model_epoch_472.pth"
result_path="/blue/neurology-dept/jlabasbas/new-fine"

# main.py defaults MASTER_PORT to a fixed 12355, so two arms landing on the same
# node collide with EADDRINUSE. Derive a per-job port instead; a requeued job
# gets a new SLURM_JOB_ID and therefore a new free port.
port=$(( 20000 + ${SLURM_JOB_ID:-$$} % 20000 ))

echo "=== Phase 3 ${ARM}"
echo "  config     ${config}"
echo "  exp_name   ${exp_name}"
echo "  start from ${resume_model}"
echo "  ddp port  ${port}"
grep -E "lambda_deepgray_weight|deepgray_mode|deepgray_shell" "$config" || true

# A config asking for structure weighting but not loading labels crashes ~1 min
# into training, after the model is built and the data is in RAM. Catch it here.
dg=$(grep -oE "lambda_deepgray_weight: *[0-9.]+" "$config" | grep -oE "[0-9.]+$" || echo "1.0")
dtype=$(grep -oE "dataset_type: *'[^']+'" "$config" | grep -oE "'[^']+'" | tr -d "'")
if awk "BEGIN{exit !($dg > 1.0)}" && [[ "$dtype" != *_labels ]]; then
    echo "ERROR: $config sets lambda_deepgray_weight=$dg but dataset_type=$dtype" >&2
    echo "       carries no labels. Use 'ct2mr_aligned_global_hist_context_labels'." >&2
    exit 1
fi
echo "  config check OK (deepgray=$dg, dataset=$dtype)"

python -u ./main.py \
    --train \
    --exp_name "$exp_name" \
    --config "$config" \
    --HW $HW \
    --plane $plane \
    --batch $batch \
    --ddim_eta $ddim_eta \
    --save_top \
    --gpu_ids $gpu_ids \
    --resume_model "$resume_model" \
    --result_path "$result_path" \
    --port "$port"
