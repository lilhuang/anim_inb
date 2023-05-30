#!/bin/bash

#job name
#SBATCH -J liljob_train_unet_rrdb_atd12k_mask_dt_raft_smooth_centerish_patchgan_patch_warp_lrg_1e-3_lrd_2e-4

#number of nodes
#SBATCH -N 1

#walltime (set to 3 days)
#SBATCH -t 1-00:00:00

#output file
#SBATCH --output=logs/train_unet_rrdb_atd12k_mask_dt_raft_smooth_centerish_patchgan_patch_warp_lrg_1e-3_lrd_2e-4.log
#SBATCH --error=logs/train_unet_rrdb_atd12k_mask_dt_raft_smooth_centerish_patchgan_patch_warp_lrg_1e-3_lrd_2e-4.err

#memory size
#SBATCH --mem=64gb

#SBATCH -c 4
#SBATCH -p dpart
#SBATCH --qos=high
#SBATCH --account=abhinav
#SBATCH --gres=gpu:p6000:4


cd ../

source env3-9-5/bin/activate
 
module load cuda/10.0.130
module load cudnn/v7.6.5

srun python train_anime_sequence.py configs/seg_config/config_train_unet_rrdb_atd12k_mask_dt_raft_smooth_centerish_patchgan_patch_warp_lrg_1e-3_lrd_2e-4.py

