#!/bin/bash

#job name
#SBATCH -J lil_flowjob

#number of nodes
#SBATCH -N 1

#walltime (set to 1 day)
#SBATCH -t 1-00:00:00

#output file
#SBATCH --error=logs/run_cv2_tvl1_flow_Blender_cubes.err
#SBATCH --output=logs/run_cv2_tvl1_flow_Blender_cubes.log

#memory size
#SBATCH --mem=128gb

#SBATCH -c 8
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1


cd ../

srun python cv2_tvl1.py










