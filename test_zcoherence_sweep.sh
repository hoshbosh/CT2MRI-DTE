#!/bin/bash
# Z-coherence sampler sweep on epoch-241 checkpoint.
#
# Goal: find a sampler that keeps the through-plane (z) smoothness that
# `ISTA_average` was buying us, without the in-plane blur it introduced.
#
# Three configs, all at sample_step=50, eta=0 (the sharp-slice sweet spot):
#   1. 'average'      — cross-channel z-averaging, no ISTA proximal step
#                       (likely candidate: keeps coherence, drops blur source)
#   2. 'ISTA_average' with ISTA_step_size=0.5 (down from 2.0)
#                     — gentler proximal step; may preserve detail
#   3. 'normal'       — for reference (already known: sharp xy, jagged z)
#
# Each run lands in its own auto-named subfolder under sample_path:
#   .../average_50/
#   .../ISTA_average_50/ISTA_average_50_0.5_1/
#   .../normal_50/
# Compare the .nii volumes side-by-side and the per-run results.csv.

config_name="fine-tune.yaml"
HW="256"
plane="axial"
gpu_ids="0"

exp_name="241213_256_BBDM_axial_DDIM_MR_global_hist_context"

resume_model="/blue/neurology-dept/jlabasbas/new-fine/fine-tune_256/241213_256_BBDM_axial_DDIM_MR_global_hist_context/checkpoint/top_model_epoch_235.pth"
resume_optim="/blue/neurology-dept/jlabasbas/new-fine/fine-tune_256/241213_256_BBDM_axial_DDIM_MR_global_hist_context/checkpoint/top_optim_sche_epoch_235.pth"
config_backup="/blue/neurology-dept/jlabasbas/new-fine/fine-tune_256/241213_256_BBDM_axial_DDIM_MR_global_hist_context/checkpoint/config_backup.yaml"

run_sample() {
    local tag="$1"
    local inf_type="$2"
    local ista_step_size="$3"
    echo
    echo "================================================================"
    echo "  $tag :: inference_type=$inf_type sample_step=50 eta=0.0 ISTA_step_size=$ista_step_size"
    echo "================================================================"
    python ./main.py \
        --exp_name $exp_name \
        --config $config_backup \
        --sample_to_eval \
        --gpu_ids $gpu_ids \
        --resume_model $resume_model \
        --resume_optim $resume_optim \
        --HW $HW \
        --plane $plane \
        --ddim_eta 0.0 \
        --sample_step 50 \
        --inference_type $inf_type \
        --ISTA_step_size $ista_step_size \
        --num_ISTA_step 1
}

# 1) Cross-channel z-averaging without ISTA. Most likely winner.
run_sample "average_z_only"        "average"      2.0

# 2) Gentle ISTA: keep the proximal step but at 1/4 the strength.
run_sample "ISTA_gentle"           "ISTA_average" 0.5

# 3) Reference: per-slice independent (sharp xy, jagged z).
run_sample "normal_reference"      "normal"       2.0
