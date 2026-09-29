#!/bin/bash
# Phase 2, step 2: SynthSeg the exported real and synthetic MR volumes.
#
# One array task per (subject, volume) pair, so 36 subjects -> 72 tasks. Each
# subject's mode comes from subjects.csv, which export_eval_volumes.py filled in
# from the Phase 1 manifest: a subject whose reference labels came from --robust
# must have BOTH its real and synthetic volume segmented with --robust, or Dice
# would be measuring the disagreement between two SynthSeg models.
#
#   sbatch --array=0-71%20 shell/data/synthseg_eval_array.sh <export_dir>
#
#SBATCH --job-name=synthseg_eval
#SBATCH --account=neurology-dept
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=joshua.labasbas@ufl.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=synthseg-eval-%A_%a.log

set -euo pipefail

pwd; hostname; date

module purge
module load freesurfer
unset PYTHONHOME
unset PYTHONPATH

EXPORT_DIR="${1:?usage: sbatch --array=0-71%20 synthseg_eval_array.sh <export_dir>}"
INDEX="${EXPORT_DIR}/subjects.csv"
OUT_DIR="${EXPORT_DIR}/seg"

if [ ! -f "$INDEX" ]; then
    echo "FAIL: no subjects.csv in ${EXPORT_DIR}; run export_eval_volumes.py first"
    exit 1
fi

mkdir -p "$OUT_DIR"

# Two volumes per subject: even task ids take the real volume, odd the synthetic.
N=$(( SLURM_ARRAY_TASK_ID / 2 ))
if [ $(( SLURM_ARRAY_TASK_ID % 2 )) -eq 0 ]; then
    KIND=real
else
    KIND=syn
fi

LINE=$(tail -n +2 "$INDEX" | sed -n "$((N + 1))p")
if [ -z "$LINE" ]; then
    echo "No subject at row ${N} -- array is wider than the cohort"
    exit 0
fi
SUBJECT=$(echo "$LINE" | cut -d, -f1)
MODE=$(echo "$LINE" | cut -d, -f6)

IN="${EXPORT_DIR}/${SUBJECT}_${KIND}.nii.gz"
OUT="${OUT_DIR}/${SUBJECT}_${KIND}_synthseg.nii.gz"
VOL="${OUT_DIR}/${SUBJECT}_${KIND}_vol.csv"
QC="${OUT_DIR}/${SUBJECT}_${KIND}_qc.csv"

if [ -f "$OUT" ]; then
    echo "SKIP ${SUBJECT} ${KIND}: ${OUT} already exists"
    date
    exit 0
fi

if [ ! -f "$IN" ]; then
    echo "FAIL ${SUBJECT} ${KIND}: input not found at ${IN}"
    exit 1
fi

ROBUST_FLAG=""
if [ "$MODE" = "robust" ]; then
    ROBUST_FLAG="--robust"
fi

echo "SEG  ${SUBJECT} ${KIND} (mode=${MODE}): ${IN} -> ${OUT}"

TMP="${OUT_DIR}/.${SUBJECT}_${KIND}_synthseg.partial.nii.gz"
rm -f "$TMP"

# --keepgeom is required, not optional. Without it SynthSeg writes on its own
# internal 1mm grid (256x256x173 came back as 234x213x173), and while the two
# segmentations of a pair stay mutually comparable -- so Dice and centroid would
# still be valid -- the per-structure SSIM/PSNR need the mask to index the image
# volume it was derived from. Keeping the input grid avoids resampling either.
# shellcheck disable=SC2086
env -u PYTHONHOME -u PYTHONPATH mri_synthseg \
    --i "$IN" \
    --o "$TMP" \
    --vol "$VOL" \
    --qc "$QC" \
    --keepgeom \
    $ROBUST_FLAG \
    --threads "${SLURM_CPUS_PER_TASK:-8}" \
    --cpu

mv "$TMP" "$OUT"
echo "OK   ${SUBJECT} ${KIND}"
date
