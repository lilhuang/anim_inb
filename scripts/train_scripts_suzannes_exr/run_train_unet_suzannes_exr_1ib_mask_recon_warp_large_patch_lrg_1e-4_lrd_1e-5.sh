#!/bin/bash

#job name
#SBATCH -J liljob_train_unet_suzannes_exr_1ib_mask_recon_warp_large_patch_lrg_1e-4_lrd_1e-5

#number of nodes
#SBATCH -N 1

#walltime (set to 3 days)
#SBATCH -t 0-03:00:00

#output file
#SBATCH --output=logs/train_unet_suzannes_exr_1ib_mask_recon_warp_large_patch_lrg_1e-4_lrd_1e-5.log
#SBATCH --error=logs/train_unet_suzannes_exr_1ib_mask_recon_warp_large_patch_lrg_1e-4_lrd_1e-5.err

#memory size
#SBATCH --mem=64gb

#SBATCH -c 4
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:4


cd ../

source env3-9-5/bin/activate
 
module load cuda/11.3.1 cudnn/v8.2.1 ffmpeg

srun python train_anime_sequence.py configs/seg_config/config_train_unet_suzannes_exr_1ib_large_patch_recon_warp_lrg_1e-4_lrd_1e-5.py

