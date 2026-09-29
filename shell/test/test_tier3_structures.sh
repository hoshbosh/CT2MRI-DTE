#!/bin/bash
# Phase 2 baseline sampling: Tier 3 ep_472 on the test split, for per-structure
# evaluation. Same checkpoint, sampler, steps and eta as the run that produced
# the headline SSIM 0.7736 / masked 0.5329, so the structure numbers attach to
# the recorded SOTA rather than to a differently-configured run.
#
# Only needed if the existing normal_200 samples cannot be reused. Check first:
#   ls <result_path>/fine-tune_256/<exp>/sample_to_eval/<ckpt>/normal_200/*.nii | wc -l
# 36 files there means sampling can be skipped entirely.
#
# --dataset_path points at fine_v2 so the slice selection matches the HDF5 that
# carries LABEL_dataset. export_eval_volumes.py asserts the per-subject slice
# counts agree, so a mismatch fails loudly rather than misaligning quietly.

config_name="fine-tune.yaml"
HW="256"
plane="axial"
ddim_eta=0.0

gpu_ids="0"

exp_name="241213_256_BBDM_axial_DDIM_MR_tier3_morphmask"

CKPT_DIR="/blue/neurology-dept/jlabasbas/new-fine/fine-tune_256/${exp_name}/checkpoint"
resume_model="${CKPT_DIR}/top_model_epoch_472.pth"
resume_optim="${CKPT_DIR}/top_optim_sche_epoch_472.pth"

sample_step=200
inference_type="normal"
ISTA_step_size=2
num_ISTA_step=1

dataset_path="/blue/neurology-dept/jlabasbas/hdf5s/fine_v2"

# Separate tree from the 8-8-26 results so the historical run stays untouched.
result_path="${RESULT_PATH:-/blue/neurology-dept/jlabasbas/phase2}"

python ./main.py \
    --exp_name $exp_name \
    --config ${CKPT_DIR}/config_backup.yaml \
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
    --dataset_path $dataset_path \
    --result_path $result_path
