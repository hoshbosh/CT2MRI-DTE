#!/bin/bash

config_name="fine-tune.yaml"
HW="256"
plane="axial"
ddim_eta=0.0

gpu_ids="0"

exp_name="241213_256_BBDM_axial_DDIM_MR_tier3_morphmask"

# Update epoch number after training completes
resume_model="/blue/neurology-dept/jlabasbas/new-fine/fine-tune_256/241213_256_BBDM_axial_DDIM_MR_tier3_morphmask/checkpoint/top_model_epoch_472.pth"
resume_optim="/blue/neurology-dept/jlabasbas/new-fine/fine-tune_256/241213_256_BBDM_axial_DDIM_MR_tier3_morphmask/checkpoint/top_optim_sche_epoch_472.pth"

sample_step=200
inference_type="normal"
ISTA_step_size=2
num_ISTA_step=1

# Eval output tree, relocated 2026-08-08 from $HOME/CT2MRI-DTE/results.
result_path="/blue/neurology-dept/jlabasbas/8-8-26"

python ./main.py \
    --exp_name $exp_name \
    --config /blue/neurology-dept/jlabasbas/new-fine/fine-tune_256/241213_256_BBDM_axial_DDIM_MR_tier3_morphmask/checkpoint/config_backup.yaml \
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
