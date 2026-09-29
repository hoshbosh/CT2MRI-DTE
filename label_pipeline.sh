#!/bin/bash
# Phase 1, steps 2-4: backfill geometry records, transport SynthSeg labels onto
# the training grid, and run the verification pass.
#
# Run AFTER the synthseg array has finished:
#   sbatch --array=0-179%20 shell/data/synthseg_array.sh
#   sbatch label_pipeline.sh
#
#SBATCH --job-name=ct2mri_labels
#SBATCH --account=neurology-dept
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joshua.labasbas@ufl.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --output=labels-%j.log

set -euo pipefail

pwd; hostname; date

# No FreeSurfer here: this stage is pure Python and loading FreeSurfer alongside
# conda is what breaks mri_synthseg in the segmentation job.
module purge
module load python
module load conda
conda activate ct2mri

NATIVE_DIR=/blue/neurology-dept/jlabasbas/synthrad/brain
SYNTHSEG_DIR=/blue/neurology-dept/jlabasbas/synthseg
ROBUST_DIR=/blue/neurology-dept/jlabasbas/synthseg_robust
OUT_DIR=/blue/neurology-dept/jlabasbas/out-fine
MANIFEST=./datasets/label_qc/label_manifest.csv

echo "=== 1/3  backfill geometry.json (does not touch existing ct.nii/mr.nii) ==="
python -u finetune_preprocess.py \
    --input_dir "$NATIVE_DIR" \
    --output_dir "$OUT_DIR" \
    --geometry_only \
    --workers 12

echo "=== 2/3  transport labels onto the training grid ==="
python -u brain_dataset_utils/make_labels.py \
    --synthseg_dir "$SYNTHSEG_DIR" \
    --robust_dir "$ROBUST_DIR" \
    --manifest "$MANIFEST" \
    --native_dir "$NATIVE_DIR" \
    --out_dir "$OUT_DIR"

echo "=== 3/3  verification: volumes, outliers, overlays ==="
python -u brain_dataset_utils/verify_labels.py \
    --out_dir "$OUT_DIR" \
    --data_csv "$OUT_DIR/data.csv" \
    --report_dir ./datasets/label_qc

date
