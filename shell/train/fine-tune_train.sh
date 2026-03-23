#!/bin/bash

date="241213"

config_name="fine-tune.yaml"
HW="180"
plane="axial"
gpu_ids="0,1"
batch=48
ddim_eta=0.0
dataset_type=""

prefix="MR_global_hist_context"

exp_name="${date}_${HW}_BBDM_${plane}_DDIM_${prefix}"


    #--sample_at_start \
resume_model="./results/fine-tune_180/fine-tune_180/$exp_name/checkpoint/last_model.pth"
resume_optim="./results/fine-tune_180/fine-tune_180/$exp_name/checkpoint/last_optim_sche.pth"
result_path="./results/fine-tune_180"
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

