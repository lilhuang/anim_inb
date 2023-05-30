#!/bin/bash

#job name
#SBATCH -J liljob_png_5fin_from_exr_train_1ib

#array
#SBATCH --array=6-100

#number of nodes
#SBATCH -N 1

#walltime (set to 3 days)
#SBATCH -t 1-00:00:00

#output file
#SBATCH --output=../logs/png_5fin_from_exr_train_1ib.log
#SBATCH --error=../logs/png_5fin_from_exr_train_1ib.err

#memory size
#SBATCH --mem=64gb

#SBATCH -c 4
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:rtxa6000:1


cd ../../

source env3-9-5/bin/activate

srun python write_Blender_png_from_EXR_csv.py --example $(( SLURM_ARRAY_TASK_ID )) --split train --ib $(( 1 )) > 5fin_log.txt

