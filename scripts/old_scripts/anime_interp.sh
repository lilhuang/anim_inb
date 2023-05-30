#!/bin/bash

#job name
#SBATCH -J anime_interp_SU

#number of nodes
#SBATCH -N 1

#walltime (set to 3 days)
#SBATCH -t 1-00:00:00

#output file
#SBATCH --output=anime_interp_SU.log
#SBATCH --error=anime_interp_SU.err

#memory size
#SBATCH --mem=64gb

#SBATCH -c 4
##SBATCH --account=abhinav
##SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --exclude=legacy[00-09]

##SBATCH --gres=gpu:2
##SBATCH --gres=gpu:4



source /fs/cfar-projects/anim_inb/env3-9-5/bin/activate
module unload cuda
module load cuda/11.3.1 cudnn/v8.2.1 ffmpeg gcc

cd /fs/cfar-projects/anim_inb/AnimeInterp/
python -u test_anime_sequence_one_by_one_no_annotation.py configs/config_Suzannes_exr_final_test.py