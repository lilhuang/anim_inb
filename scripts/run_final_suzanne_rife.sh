#!/bin/bash

#job name
#SBATCH -J suzanne_final_rife

#number of nodes
#SBATCH -N 1

#walltime (set to 3 days)
#SBATCH -t 1-00:00:00

#output file
#SBATCH --output=new_logs/suzanne_final_rife.log
#SBATCH --error=new_logs/suzanne_final_rife.err

#memory size
#SBATCH --mem=64gb

#SBATCH -c 4
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1


cd /fs/cfar-projects/anim_inb

source env3-9-5/bin/activate

module unload cuda
module load cuda/11.3.1 cudnn/v8.2.1 ffmpeg gcc

cd arXiv2020-RIFE

python benchmark/Suzannes_exr.py


