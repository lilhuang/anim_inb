#!/bin/bash

#job name
#SBATCH -J liljob_train_deep_unet_SU_smol_3ib_recon_gan_lrg_1e-3_lrd_1e-4

#number of nodes
#SBATCH -N 1

#walltime (set to 3 days)
#SBATCH -t 1-00:00:00

#output file
#SBATCH --output=logs/train_deep_unet_SU_smol_3ib_recon_gan_lrg_1e-3_lrd_1e-4.log
#SBATCH --error=logs/train_deep_unet_SU_smol_3ib_recon_gan_lrg_1e-3_lrd_1e-4.err

#memory size
#SBATCH --mem=64gb

#SBATCH -c 4
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --exclude=legacy[00-09]


cd /fs/cfar-projects/anim_inb

source env3-9-5/bin/activate

module load cuda/11.3.1 cudnn/v8.2.1 ffmpeg

srun python train_anime_sequence.py configs/seg_config/config_train_deep_unet_SU_smol_1ib_recon_gan_lrg_1e-3_lrd_1e-4.py

