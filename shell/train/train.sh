#!/bin/bash

date="241213"

config_name="pretrain_bbdm.yaml"
HW="180"
plane="axial"
gpu_ids="0,1"
batch=6
ddim_eta=0.0
dataset_type=""

prefix="MR_global_hist_context"

exp_name="${date}_${HW}_BBDM_${plane}_DDIM_${prefix}"

mkdir ./results/ct2mr_${HW}/$exp_name

resume_model="./results/ct2mr_$HW/$exp_name/checkpoint/last_model.pth"
resume_optim="./results/ct2mr_$HW/$exp_name/checkpoint/last_optim_sche.pth"
result_path="/blue/neurology-dept/jlabasbas/results"
python -u ./main.py \
    --train \
    --exp_name $exp_name \
    --config ./configs/$config_name \
    --HW $HW \
    --plane $plane \
    --batch $batch \
    --ddim_eta $ddim_eta \
    --sample_at_start \
    --save_top \
    --gpu_ids $gpu_ids \
    --result_path $result_path

