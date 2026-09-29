#!/bin/bash
# Re-segment the subjects that failed QC in Phase 1, using --robust.
#
# Five subjects came out of the first pass unusable:
#   1BA076  SynthSeg produced no right amygdala at all
#   1BC001  bilateral thalamus ~2x cohort median (27,170 mm3 native)
#   1BB003  bilateral thalamus ~1.6x cohort median (22,285 mm3 native)
#   1BB028  amygdala high
#   1BB189  amygdala low  (TEST split -- see note below)
#
# Output goes to a SEPARATE directory, not on top of the first pass. --robust
# is a different model, so its results have to be compared against the originals
# before anything is replaced; overwriting first would destroy the comparison.
#
#   sbatch --array=0-4 shell/data/synthseg_robust.sh
#
#SBATCH --job-name=synthseg_robust
#SBATCH --account=neurology-dept
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=joshua.labasbas@ufl.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=synthseg-robust-%A_%a.log

set -euo pipefail

pwd; hostname; date

module purge
module load freesurfer
unset PYTHONHOME
unset PYTHONPATH

NATIVE_DIR=/blue/neurology-dept/jlabasbas/synthrad/brain
OUT_DIR=/blue/neurology-dept/jlabasbas/synthseg_robust

mkdir -p "$OUT_DIR"

SUBJECTS=(1BA076 1BC001 1BB003 1BB028 1BB189)
SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

IN="${NATIVE_DIR}/${SUBJECT}/mr.nii.gz"
OUT="${OUT_DIR}/${SUBJECT}_synthseg.nii.gz"
VOL="${OUT_DIR}/${SUBJECT}_vol.csv"
QC="${OUT_DIR}/${SUBJECT}_qc.csv"

if [ -f "$OUT" ]; then
    echo "SKIP ${SUBJECT}: ${OUT} already exists"
    date
    exit 0
fi

if [ ! -f "$IN" ]; then
    echo "FAIL ${SUBJECT}: native MR not found at ${IN}"
    exit 1
fi

echo "SEG  ${SUBJECT} (robust): ${IN} -> ${OUT}"

TMP="${OUT_DIR}/.${SUBJECT}_synthseg.partial.nii.gz"
rm -f "$TMP"

# --robust runs the 1mm hemisphere-wise model, which is slower and more tolerant
# of poor-quality scans. It is incompatible with --fast and, per SynthSeg's
# docs, ignores --keepgeom-style resampling shortcuts on some builds -- so the
# output grid is checked by the comparison step rather than assumed.
env -u PYTHONHOME -u PYTHONPATH mri_synthseg \
    --i "$IN" \
    --o "$TMP" \
    --vol "$VOL" \
    --qc "$QC" \
    --robust \
    --threads "${SLURM_CPUS_PER_TASK:-8}" \
    --cpu

mv "$TMP" "$OUT"
echo "OK   ${SUBJECT}"
date
