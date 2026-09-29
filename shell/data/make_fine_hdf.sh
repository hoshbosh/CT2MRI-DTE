#!/bin/bash
# Build the fine-tune HDF5s. Writes to a NEW directory by default so the
# existing hdf5s/fine build stays intact for comparison -- this rebuild changes
# index_dataset from uint8 to int32 (fixing the >255-slice desynchronisation)
# and adds LABEL_dataset and spacing_dataset.
set -euo pipefail

OUT_DIR="${OUT_DIR:-/blue/neurology-dept/jlabasbas/hdf5s/fine_v2}"
SRC_DIR="${SRC_DIR:-/blue/neurology-dept/jlabasbas/out-fine}"
OLD_DIR="${OLD_DIR:-/blue/neurology-dept/jlabasbas/hdf5s/fine}"
SEG_NAME="${SEG_NAME:-seg.nii}"
SIZE="${SIZE:-256}"

mkdir -p ./datasets/hdf5_log "$OUT_DIR"

for which in "train" "valid" "test"
do
    for plane in "axial"
    do
    python -u brain_dataset_utils/generate_total_hdf5_csv.py \
            --plane  $plane\
            --which_set $which \
            --height ${SIZE} \
            --width ${SIZE} \
            --hdf5_name "${OUT_DIR}/${SIZE}_${which}_${plane}.hdf5" \
            --data_dir "${SRC_DIR}" \
            --data_csv "${SRC_DIR}/data.csv" \
            --CT_name "ct.nii" \
            --MR_name "mr.nii" \
            --SEG_name "${SEG_NAME}" \
            > ./datasets/hdf5_log/${SIZE}_${which}_${plane}_v2.log
    echo "built ${OUT_DIR}/${SIZE}_${which}_${plane}.hdf5"
    done      
done

# Style-key histograms are NOT copied from the old build. They are keyed by
# subject name, which made reuse look safe, but the old pickles were generated
# from an earlier data.csv and do not cover the current split -- sampling died
# with KeyError on a subject missing from the test pickle. Regenerate instead:
#
#   sbatch pkl_fine_v2.sh        (or: OUT_DIR=... ./shell/data/make_fine_hist.sh)
echo
echo "NOTE: histograms are not copied. Run 'sbatch pkl_fine_v2.sh' to build"
echo "      MR_hist_global_${SIZE}_{train,valid,test}_axial_.pkl into ${OUT_DIR}."
