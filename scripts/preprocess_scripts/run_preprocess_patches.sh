#!/bin/bash

#job name
#SBATCH -J liljob_preprocess_patches

#number of nodes
#SBATCH -N 1

#walltime (set to 3 days)
#SBATCH -t 1-00:00:00

#output file
#SBATCH --output=logs/preprocess_patches.log
#SBATCH --error=logs/preprocess_patches.err

#memory size
#SBATCH --mem=64gb

#SBATCH -c 4
#SBATCH -p dpart
#SBATCH --qos=high
#SBATCH --account=abhinav
#SBATCH --gres=gpu:p6000:4


cd ../

source env3-9-5/bin/activate
 
# module load cuda/10.0.130
# module load cudnn/v7.6.5

srun python preprocess_high_flow_patches.py

