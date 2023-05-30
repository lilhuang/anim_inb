#!/bin/bash

#job name
#SBATCH -J liljob_Suzanne_exr_csv

#array
#SBATCH --array=0-119

#number of nodes
#SBATCH -N 1

#walltime (set to 1 day)
#SBATCH -t 1-00:00:00

#output file
#SBATCH --output=logs/run_Suzanne_exr_csv.log
#SBATCH --error=logs/run_Suzanne_exr_csv.err

#memory size
#SBATCH --mem=64gb

#SBATCH -c 8
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa6000:1

cd ../

source env3-9-5/bin/activate

echo $(( SLURM_ARRAY_TASK_ID )) !!!

srun python write_Blender_EXR_csv.py --example $(( SLURM_ARRAY_TASK_ID ))

