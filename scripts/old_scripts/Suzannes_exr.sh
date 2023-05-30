#!/bin/bash

#job name
#SBATCH -J liljob_train_deep7_unet_suzannes_exr_1ib_recon_warp_lrg_1e-3

#number of nodes
#SBATCH -N 1

#walltime (set to 3 days)
#SBATCH -t 1-00:00:00

#output file
#SBATCH --output=Suzannes_exr.log
#SBATCH --error=Suzannes_exr.err

#memory size
#SBATCH --mem=64gb

#SBATCH -c 4
##SBATCH --account=abhinav
##SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:4

##SBATCH --gres=gpu:2
##SBATCH --gres=gpu:4



source /fs/cfar-projects/anim_inb/env3-9-5/bin/activate
module unload cuda
module load cuda/11.3.1 cudnn/v8.2.1 ffmpeg gcc

cd /fs/cfar-projects/anim_inb/arXiv2020-RIFE/
python benchmark/Suzannes_exr.py