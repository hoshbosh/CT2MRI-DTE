#!/bin/bash
# Sampler diagnostic sweep: render the same checkpoint with three sampler
# configs to find out whether the blur is a model problem or a sampler problem.
#
# Output dirs include {inference_type}_{sample_step} so the three runs land in
# separate folders. Compare the .nii volumes and results.csv (SSIM/PSNR) across them.

config_name="fine-tune.yaml"
HW="256"
plane="axial"
gpu_ids="0"

exp_name="241213_256_BBDM_axial_DDIM_MR_global_hist_context"

resume_model="./results/fine-tune_256/fine-tune_256/241213_256_BBDM_axial_DDIM_MR_global_hist_context/checkpoint/top_model_epoch_241.pth"
resume_optim="./results/fine-tune_256/fine-tune_256/241213_256_BBDM_axial_DDIM_MR_global_hist_context/checkpoint/top_optim_sche_epoch_241.pth"
config_backup="./results/fine-tune_256/fine-tune_256/241213_256_BBDM_axial_DDIM_MR_global_hist_context/checkpoint/config_backup.yaml"

run_sample() {
    local tag="$1"
    local eta="$2"
    local steps="$3"
    local inf_type="$4"
    echo
    echo "================================================================"
    echo "  $tag :: eta=$eta sample_step=$steps inference_type=$inf_type"
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
        --ddim_eta $eta \
        --sample_step $steps \
        --inference_type $inf_type \
        --ISTA_step_size 2 \
        --num_ISTA_step 1
}

# 1) Baseline: matches your current production sampler. Sanity check.
run_sample "baseline_current"  0.0 200 "ISTA_average"

# 2) Stochastic DDPM-style: re-injects noise at each step. Should sharpen if
#    the blur is from deterministic mode-averaging.
run_sample "stochastic_eta1"   1.0 50  "normal"

# 3) Few-step deterministic: each step commits more strongly to a mode.
#    Cheap to run and a useful triangulation point between (1) and (2).
run_sample "fewstep_eta0"      0.0 50  "normal"
