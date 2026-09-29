#!/bin/bash
# Phase 2 supplement: per-structure boundary (ASSD, HD95) and signed centroid
# error, which Dice cannot express. CPU only -- 36 independent subjects, one
# process each.
#
#   sbatch boundary_report.sh
#
#SBATCH --job-name=ct2mri_boundary
#SBATCH --account=neurology-dept
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joshua.labasbas@ufl.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=01:00:00
#SBATCH --output=boundary-%j.log
pwd; hostname; date

set -euo pipefail

module purge
module load python
module load conda
conda activate ct2mri

EXPORT_DIR="${EXPORT_DIR:-/blue/neurology-dept/jlabasbas/phase2/export}"

python -u runners/structure_boundary_report.py \
    --export_dir "$EXPORT_DIR" \
    --manifest ./datasets/label_qc/label_manifest.csv \
    --workers "${SLURM_CPUS_PER_TASK:-16}"

date
