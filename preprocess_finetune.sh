#!/bin/bash

#SBATCH --job-name=multi-test    # Job name 
#SBATCH --account=neurology-dept                    # Run on a single CPU 
#SBATCH --mail-type=END,FAIL          # Mail events 
#SBATCH --mail-user=joshua.labasbas@ufl.edu     # Where to send mail 
#SBATCH --nodes=1                    # Run on a single CPU 
#SBATCH --cpus-per-task=12                    # Run on a single CPU 
#SBATCH --mem=180gb                     # Job memory request 
#SBATCH --time=01:00:00               # Time limit hrs:min:sec 
#SBATCH --output=finetuning_preprocess_%j.log   # Standard output and error log 
pwd; hostname; date 

module purge 
module load python 
module load conda
module load freesurfer

conda activate ct2mri

python finetune_preprocess.py --input_dir /blue/neurology-dept/jlabasbas/synthrad/brain --output_dir /blue/neurology-dept/jlabasbas/out-fine --workers 12

date
