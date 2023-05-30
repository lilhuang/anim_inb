#!/bin/bash

#job name
#SBATCH -J 2final_SU_soumik

#number of nodes
#SBATCH -N 1

#walltime (set to 3 days)
#SBATCH -t 1-00:00:00

#output file
#SBATCH --output=final_SU_soumik_2.log
#SBATCH --error=final_SU_soumik_2.err

#memory size
#SBATCH --mem=64gb

#SBATCH -c 4
#SBATCH --account=abhinav
#SBATCH --partition=dpart
#SBATCH --qos=high
##SBATCH --qos=exempt
#SBATCH --gres=gpu:rtxa6000:2
##SBATCH --exclude=legacy[00-09]


cd /fs/cfar-projects/anim_inb

source env3-9-5/bin/activate

module unload cuda
module load cuda/11.3.1 cudnn/v8.2.1 ffmpeg gcc

python train_anime_sequence_2_stream.py configs/seg_config/config_final_SU_soumik_2.py

