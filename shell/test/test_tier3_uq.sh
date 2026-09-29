#!/bin/bash

# Uncertainty quantification on the Tier 3 checkpoint.
# Draws uq_samples stochastic samples per subject (needs ddim_eta > 0) and reports
# the per-voxel ensemble mean and predictive std. Members are cached per index, so
# raising uq_samples on a later run only generates the additional members.

config_name="fine-tune.yaml"
HW="256"
plane="axial"

# Must be > 0 or sampling is deterministic and every member comes out identical.
ddim_eta=1.0

gpu_ids="0"

exp_name="241213_256_BBDM_axial_DDIM_MR_tier3_morphmask"

resume_model="/blue/neurology-dept/jlabasbas/new-fine/fine-tune_256/241213_256_BBDM_axial_DDIM_MR_tier3_morphmask/checkpoint/top_model_epoch_472.pth"
resume_optim="/blue/neurology-dept/jlabasbas/new-fine/fine-tune_256/241213_256_BBDM_axial_DDIM_MR_tier3_morphmask/checkpoint/top_optim_sche_epoch_472.pth"

# 50 steps rather than 200: prior sweep found sampler step count is not a quality
# lever, and 200 x 5 members would be ~20 h of GPU on a partition that kills jobs hourly.
sample_step=50
inference_type="normal"
ISTA_step_size=2
num_ISTA_step=1

uq_samples=5

# Eval output tree, relocated 2026-08-08 from $HOME/CT2MRI-DTE/results.
# Without this the default result_path is relative and lands in the home quota.
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
    --uq_samples $uq_samples \
    --result_path $result_path
