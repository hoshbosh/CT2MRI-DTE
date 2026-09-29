#!/bin/bash

# Tier 4 evaluation (z-context = 7 slices).
#
# Run this twice:
#   1. defaults (normal / 200)          -> directly comparable to the Tier 3 baseline
#                                          SSIM 0.7736 / SSIM_mask 0.5329 (job 38791459)
#   2. INFERENCE_TYPE=average           -> exploits the 7 overlapping predictions per
#                                          output slice (Tier 3 only had 3)
#
# Override without editing:  INFERENCE_TYPE=average sbatch ct2mri_test_tier4.sh

config_name="fine-tune_tier4.yaml"
HW="256"
plane="axial"
ddim_eta=0.0

gpu_ids="0"

exp_name="241213_256_BBDM_axial_DDIM_MR_tier4_zcontext7"

# UPDATE after training: set to the best epoch reported in the Tier 4 training log
# (grep for "remove top_model_epoch_" — the one never removed is the best).
test_epoch="542"

ckpt_dir="/blue/neurology-dept/jlabasbas/new-fine/fine-tune_256/241213_256_BBDM_axial_DDIM_MR_tier4_zcontext7/checkpoint"
resume_model="${ckpt_dir}/top_model_epoch_${test_epoch}.pth"
resume_optim="${ckpt_dir}/top_optim_sche_epoch_${test_epoch}.pth"

sample_step="${SAMPLE_STEP:-200}"
inference_type="${INFERENCE_TYPE:-normal}"
ISTA_step_size=2
num_ISTA_step=1

# Eval output tree, relocated 2026-08-08 from $HOME/CT2MRI-DTE/results.
# Not the same tree as training output (--result_path in shell/train/*), which still
# lives under new-fine alongside the checkpoints.
result_path="/blue/neurology-dept/jlabasbas/8-8-26"

if [ "$test_epoch" = "TBD" ]; then
    echo "ERROR: set test_epoch in shell/test/test_tier4.sh before running" >&2
    exit 1
fi

python ./main.py \
    --exp_name $exp_name \
    --config ${ckpt_dir}/config_backup.yaml \
    --sample_to_eval \
    --gpu_ids $gpu_ids \
    --resume_model $resume_model \
    --resume_optim $resume_optim \
    --HW $HW \
    --plane $plane \
    --ddim_eta $ddim_eta \
    --sample_step $sample_step \
    --inference_type $inference_type \
    --ISTA_step_size $ISTA_step_size \
    --num_ISTA_step $num_ISTA_step \
    --result_path $result_path
