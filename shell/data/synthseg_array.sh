#!/bin/bash
# SynthSeg subcortical segmentation of every subject's NATIVE MR.
#
# Native, not preprocessed: SynthSeg is resolution-aware and out-fine/<pid>/mr.nii
# carries a stale affine (the in-plane resize is not reflected in it). Labels are
# transported onto the training grid afterwards by make_labels.py.
#
# CPU array rather than GPU: each task is a few minutes, the array runs 20 wide,
# and CPU partitions do not get preempted the way hpg-b200 does. The job is
# resumable either way -- a subject whose output already exists is skipped.
#
#   sbatch --array=0-179%20 shell/data/synthseg_array.sh
#
#SBATCH --job-name=synthseg
#SBATCH --account=neurology-dept
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=joshua.labasbas@ufl.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=synthseg-%A_%a.log

set -euo pipefail

pwd; hostname; date

# FreeSurfer 8.x bundles its own Python. Loading the python/conda modules sets
# PYTHONHOME/PYTHONPATH, which hijack that interpreter and make mri_synthseg die
# with "No module named 'encodings'". This job needs nothing but FreeSurfer, so
# load only FreeSurfer and strip the two variables as a belt-and-braces guard.
module purge
module load freesurfer
unset PYTHONHOME
unset PYTHONPATH

NATIVE_DIR=/blue/neurology-dept/jlabasbas/synthrad/brain
OUT_DIR=/blue/neurology-dept/jlabasbas/synthseg
DATA_CSV=/blue/neurology-dept/jlabasbas/out-fine/data.csv

mkdir -p "$OUT_DIR"

# Subject list is derived from data.csv so it always matches the split in use.
SUBJECT=$(tail -n +2 "$DATA_CSV" | cut -d, -f1 | sort | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [ -z "$SUBJECT" ]; then
    echo "No subject at array index ${SLURM_ARRAY_TASK_ID} -- array is wider than the cohort"
    exit 0
fi

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

echo "SEG  ${SUBJECT}: ${IN} -> ${OUT}"

# --vol and --qc are free and give a per-subject failure signal that Phase 2
# uses instead of silently averaging a bad segmentation into the results.
# Write to a temp name first so an interrupted task cannot leave a truncated
# .nii.gz that the skip-if-exists check would later treat as complete.
TMP="${OUT_DIR}/.${SUBJECT}_synthseg.partial.nii.gz"
rm -f "$TMP"

# --keepgeom writes the segmentation on the input grid instead of SynthSeg's
# internal 1mm grid, so make_labels.py can skip a nearest-neighbour resample.
# mri_synthseg is an `apptainer exec` wrapper and the container inherits the
# host environment, which is why PYTHONHOME/PYTHONPATH must be stripped here.
env -u PYTHONHOME -u PYTHONPATH mri_synthseg \
    --i "$IN" \
    --o "$TMP" \
    --vol "$VOL" \
    --qc "$QC" \
    --keepgeom \
    --threads "${SLURM_CPUS_PER_TASK:-8}" \
    --cpu

if [ ! -s "$TMP" ]; then
    echo "FAIL ${SUBJECT}: mri_synthseg exited 0 but produced no output"
    rm -f "$TMP"
    exit 1
fi

mv "$TMP" "$OUT"
echo "OK   ${SUBJECT}"
date
