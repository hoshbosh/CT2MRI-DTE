#!/bin/bash

date="241213"

config_name="pretrain_bbdm.yaml"
HW="256"
plane="axial"
gpu_ids="0,1"
batch=32
ddim_eta=0.0
dataset_type=""

prefix="MR_global_hist_context"

exp_name="${date}_${HW}_BBDM_${plane}_DDIM_${prefix}"

mkdir ./results/ct2mr_${HW}/$exp_name

    #--sample_at_start \
resume_model="/blue/neurology-dept/jlabasbas/results-newest/ct2mr_256/241213_256_BBDM_axial_DDIM_MR_global_hist_context/checkpoint/last_model.pth"
resume_optim="/blue/neurology-dept/jlabasbas/results-newest/ct2mr_256/241213_256_BBDM_axial_DDIM_MR_global_hist_context/checkpoint/last_optim_sche.pth"
result_path="/blue/neurology-dept/jlabasbas/results-newest/"
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
    --result_path $result_path
    --resume_model $resume_model \
    --resume_optim $resume_optim \

