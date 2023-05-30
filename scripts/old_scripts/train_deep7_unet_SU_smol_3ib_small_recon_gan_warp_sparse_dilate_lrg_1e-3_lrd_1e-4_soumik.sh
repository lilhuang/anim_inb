#!/bin/bash

#job name
#SBATCH -J train_deep7_unet_SU_smol_3ib_small_recon_gan_warp_sparse_dilate_lrg_1e-3_lrd_1e-4_soumik

#number of nodes
#SBATCH -N 1

#walltime (set to 3 days)
#SBATCH -t 1-00:00:00

#output file
#SBATCH --output=train_deep7_unet_SU_smol_3ib_small_recon_gan_warp_sparse_dilate_lrg_1e-3_lrd_1e-4_soumik.log
#SBATCH --error=train_deep7_unet_SU_smol_3ib_small_recon_gan_warp_sparse_dilate_lrg_1e-3_lrd_1e-4_soumik.err

#memory size
#SBATCH --mem=64gb

#SBATCH -c 4
#SBATCH --account=abhinav
#SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:4
##SBATCH --gres=gpu:2
##SBATCH --gres=gpu:4



cd /fs/cfar-projects/anim_inb

source env3-9-5/bin/activate
module unload cuda
module load cuda/11.3.1 cudnn/v8.2.1 ffmpeg

python train_anime_sequence_soumik.py /fs/cfar-projects/anim_inb/scripts/config_train_deep7_unet_SU_smol_3ib_small_recon_gan_warp_sparse_dilate_lrg_1e-3_lrd_1e-4_soumik.py

