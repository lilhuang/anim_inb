#!/bin/bash

#job name
#SBATCH -J liljob_train_deep_unet_SU_1ib_recon_lrg_1e-3

#number of nodes
#SBATCH -N 1

#walltime (set to 3 days)
#SBATCH -t 1-00:00:00

#output file
#SBATCH --output=/fs/cfar-projects/anim_inb/scripts/logs/train_deep_unet_SU_1ib_recon_lrg_1e-3_soumik.log
#SBATCH --error=/fs/cfar-projects/anim_inb/scripts/logs/train_deep_unet_SU_1ib_recon_lrg_1e-3_soumik.err

#memory size
#SBATCH --mem=64gb

#SBATCH -c 4
#SBATCH --account=abhinav
#SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:4
##SBATCH --exclude=legacy[00-09]


# cd ../
cd /fs/cfar-projects/anim_inb/

source env3-9-5/bin/activate
 
module unload cuda
module load cuda/11.3.1 cudnn/v8.2.1 ffmpeg

python train_anime_sequence.py configs/seg_config/config_train_deep_unet_SU_1ib_noflow_recon_lrg_1e-3.py

