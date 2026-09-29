#!/bin/bash

#SBATCH --job-name=ct2mri_pkl_v2
#SBATCH --account=neurology-dept
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joshua.labasbas@ufl.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180gb
#SBATCH --time=03:00:00
#SBATCH --output=make-hist-fine-v2-%j.log
pwd; hostname; date

module purge
module load python
module load conda

conda activate ct2mri

./shell/data/make_fine_hist.sh

date
