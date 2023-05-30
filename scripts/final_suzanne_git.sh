#!/bin/bash

#job name
#SBATCH -J suzanne_final_git

#number of nodes
#SBATCH -N 1

#walltime (set to 3 days)
#SBATCH -t 1-00:00:00

#output file
#SBATCH --output=new_logs/suzanne_final_git.log
#SBATCH --error=new_logs/suzanne_final_git.err

#memory size
#SBATCH --mem=64gb

#SBATCH -c 4
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:4


cd /fs/cfar-projects/anim_inb

source env3-9-5/bin/activate

module unload cuda
module load cuda/11.3.1 cudnn/v8.2.1 ffmpeg gcc

python train_anime_sequence_2_stream.py configs/seg_config/config_final_suzannes_git.py

