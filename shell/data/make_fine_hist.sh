#!/bin/bash
# Rebuild the style-key histograms for the fine_v2 HDF5s.
#
# The pickles copied from hdf5s/fine were built from an older data.csv and do
# not cover the current split -- sampling died with KeyError: '1BA014'. They are
# keyed by subject name, so reuse is only valid when the subject SET matches,
# which is the assumption that failed. Regenerating from the same data.csv the
# HDF5s were built from removes the coupling.
set -euo pipefail

OUT_DIR="${OUT_DIR:-/blue/neurology-dept/jlabasbas/hdf5s/fine_v2}"
SRC_DIR="${SRC_DIR:-/blue/neurology-dept/jlabasbas/out-fine}"
SIZE="${SIZE:-256}"

mkdir -p ./datasets/hdf5_log "$OUT_DIR"

for which in "train" "valid" "test"
do
    for plane in "axial"
    do
    for hist_type in "normal"
    do
        # The trailing underscore with no suffix is deliberate: datasets/custom.py
        # looks for MR_hist_global_<size>_<stage>_<plane>_.pkl unless the config
        # sets hist_type, and the tier3 config leaves it null.
        python -u ./brain_dataset_utils/generate_total_hist_global.py \
            --plane $plane \
            --hist_type $hist_type \
            --which_set $which \
            --height ${SIZE} \
            --width ${SIZE} \
            --pkl_name "${OUT_DIR}/MR_hist_global_${SIZE}_${which}_${plane}_.pkl" \
            --data_dir "${SRC_DIR}" \
            --data_csv "${SRC_DIR}/data.csv" \
            --CT_name "ct.nii" \
            --MR_name "mr.nii" \
            > ./datasets/hdf5_log/MR_hist_global_${SIZE}_${which}_${plane}_v2.log
        echo "built ${OUT_DIR}/MR_hist_global_${SIZE}_${which}_${plane}_.pkl"
    done
    done
done
