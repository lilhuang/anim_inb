#!/bin/bash

#job name
#SBATCH -J rife

#number of nodes
#SBATCH -N 1

#walltime (set to 3 days)
#SBATCH -t 1-00:00:00

#output file
#SBATCH --output=train_RIFE_SU_halfhalf_2.log
#SBATCH --error=train_RIFE_SU_halfhalf_2.err

#memory size
#SBATCH --mem=20gb

#SBATCH -c 16
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:4

##SBATCH --account=abhinav
##SBATCH --partition=dpart


source /fs/cfar-projects/anim_inb/env3-9-5_rife/bin/activate

nvidia-smi

module unload cuda
# module load cuda/11.3.1 cudnn/v8.2.1 ffmpeg gcc 
module load ffmpeg gcc 

cd /fs/cfar-projects/anim_inb/arXiv2020-RIFE
# We use 16 CPUs, 4 GPUs and 20G memory for training:
# train1.py, dataset1 has my updated code



python3 -m torch.distributed.launch --nproc_per_node=4 train1.py --world_size=4 \
--epoch=300 \
--batch_size=1 \
--csv_root=/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_smol_sequential_halfhalf_2_csv/ \
--log_path=train_log_bs1_e300_halfhalf_2