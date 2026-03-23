#!/bin/bash

config_name="fine-tune.yaml"
HW="180"
plane="axial"
ddim_eta=0.0

gpu_ids="0"

exp_name="241213_180_BBDM_axial_DDIM_MR_global_hist_context"

# test
test_epoch="81"
resume_model="./results/fine-tune_180/fine-tune_180/241213_180_BBDM_axial_DDIM_MR_global_hist_context/checkpoint/top_model_epoch_81.pth"
resume_optim="./results/fine-tune_180/fine-tune_180/241213_180_BBDM_axial_DDIM_MR_global_hist_context/checkpoint/top_optim_sche_epoch_81.pth"

sample_step=200
inference_type="ISTA_mid" # normal, average, ISTA_average, ISTA_mid
ISTA_step_size=2
num_ISTA_step=1

python ./main.py \
    --exp_name $exp_name \
    --config ./results/fine-tune_180/fine-tune_180/241213_180_BBDM_axial_DDIM_MR_global_hist_context/checkpoint/config_backup.yaml\
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


