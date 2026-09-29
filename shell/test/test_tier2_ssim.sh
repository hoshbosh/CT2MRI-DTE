#!/bin/bash

config_name="fine-tune.yaml"
HW="256"
plane="axial"
ddim_eta=0.0

gpu_ids="0"

exp_name="241213_256_BBDM_axial_DDIM_MR_tier2_ssim"

resume_model="/blue/neurology-dept/jlabasbas/new-fine/fine-tune_256/241213_256_BBDM_axial_DDIM_MR_tier2_ssim/checkpoint/top_model_epoch_412.pth"
resume_optim="/blue/neurology-dept/jlabasbas/new-fine/fine-tune_256/241213_256_BBDM_axial_DDIM_MR_tier2_ssim/checkpoint/top_optim_sche_epoch_412.pth"

sample_step=200
inference_type="normal" # normal, average, ISTA_average, ISTA_mid
ISTA_step_size=2
num_ISTA_step=1

python ./main.py \
    --exp_name $exp_name \
    --config /blue/neurology-dept/jlabasbas/new-fine/fine-tune_256/241213_256_BBDM_axial_DDIM_MR_tier2_ssim/checkpoint/config_backup.yaml \
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
    --num_ISTA_step $num_ISTA_step
