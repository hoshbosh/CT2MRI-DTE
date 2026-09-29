#!/bin/bash
# Phase 4 sampling: one Phase 3 arm at epoch 532 on the test split.
#   usage: ARM=armA_control bash ./shell/test/phase4_arm.sh
#
# Epoch 532, not top_model_epoch_476: 476 was selected on a 0.38% validation
# difference that is plausibly noise, and carries only 4 epochs of exposure to
# the deep-gray term. 532 has all 60. All three arms overfit by the same margin
# (train -7.8%, val +5-7% in every arm), so the comparison stays matched.
#
# Uses each arm's own config_backup.yaml, which carries that arm's
# lambda_deepgray_weight. Harmless here -- p_losses never runs during sampling,
# and unpack_batch drops the label tensor explicitly.
set -euo pipefail

ARM="${ARM:?ARM must be set, e.g. ARM=armB_uniform5x}"

HW="256"
plane="axial"
ddim_eta=0.0
gpu_ids="0"
sample_step=200
inference_type="normal"
ISTA_step_size=2
num_ISTA_step=1
epoch="${EPOCH:-532}"

exp_name="241213_256_BBDM_axial_DDIM_MR_p3_${ARM}"
CKPT_DIR="/blue/neurology-dept/jlabasbas/new-fine/fine-tune_256/${exp_name}/checkpoint"
resume_model="${CKPT_DIR}/latest_model_${epoch}.pth"
resume_optim="${CKPT_DIR}/latest_optim_sche_${epoch}.pth"

for f in "$resume_model" "$resume_optim" "${CKPT_DIR}/config_backup.yaml"; do
    [[ -f "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done

dataset_path="/blue/neurology-dept/jlabasbas/hdf5s/fine_v2"
result_path="${RESULT_PATH:-/blue/neurology-dept/jlabasbas/phase4}"

echo "=== Phase 4 sampling: ${ARM} @ epoch ${epoch}"
echo "  ckpt   ${resume_model}"
echo "  out    ${result_path}"

# Sampling resumes per patient: sample_to_eval skips any {pid}.nii already
# written. A drained job picks up where it stopped instead of restarting.
done_n=$(ls "${result_path}/fine-tune_256/${exp_name}/sample_to_eval/"*/normal_200/*.nii 2>/dev/null | wc -l || echo 0)
echo "  already sampled: ${done_n}/36"

python -u ./main.py \
    --exp_name "$exp_name" \
    --config "${CKPT_DIR}/config_backup.yaml" \
    --sample_to_eval \
    --gpu_ids $gpu_ids \
    --resume_model "$resume_model" \
    --resume_optim "$resume_optim" \
    --HW $HW \
    --plane $plane \
    --ddim_eta $ddim_eta \
    --sample_step $sample_step \
    --inference_type $inference_type \
    --ISTA_step_size $ISTA_step_size \
    --num_ISTA_step $num_ISTA_step \
    --dataset_path "$dataset_path" \
    --result_path "$result_path"
