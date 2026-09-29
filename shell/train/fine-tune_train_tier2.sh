#!/bin/bash

date="241213"

config_name="fine-tune.yaml"
HW="256"
plane="axial"
gpu_ids="0,1"
batch=16
ddim_eta=0.0

# Tier 2 run: loads epoch_390 (Tier 1 best) weights only, fresh optimizer.
# Adds brain-masked loss (lambda_mask_weight=3.0) on top of Tier 1 losses.
prefix="MR_tier2_masked"

exp_name="${date}_${HW}_BBDM_${plane}_DDIM_${prefix}"

resume_model="/blue/neurology-dept/jlabasbas/new-fine/fine-tune_256/241213_256_BBDM_axial_DDIM_MR_tier1_freq/checkpoint/top_model_epoch_390.pth"
result_path="/blue/neurology-dept/jlabasbas/new-fine"

python -u ./main.py \
    --train \
    --exp_name $exp_name \
    --config ./configs/$config_name \
    --HW $HW \
    --plane $plane \
    --batch $batch \
    --ddim_eta $ddim_eta \
    --save_top \
    --gpu_ids $gpu_ids \
    --resume_model $resume_model \
    --result_path $result_path
