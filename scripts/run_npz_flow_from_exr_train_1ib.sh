#!/bin/bash

#job name
#SBATCH -J liljob_npz_from_exr_test_1ib

#array
#SBATCH --array=0-100

#number of nodes
#SBATCH -N 1

#walltime (set to 3 days)
#SBATCH -t 1-00:00:00

#output file
#SBATCH --output=logs/npz_from_exr_test_1ib_git.log
#SBATCH --error=logs/npz_from_exr_test_1ib_git.err

#memory size
#SBATCH --mem=64gb

#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa6000:1


cd ../

source env3-9-5/bin/activate
 
module load cuda/11.3.1 cudnn/v8.2.1

srun python write_Blender_png_from_EXR_csv.py --example $(( SLURM_ARRAY_TASK_ID )) --split test --ib $(( 1 ))

