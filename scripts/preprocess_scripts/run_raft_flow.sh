#!/bin/bash

#job name
#SBATCH -J liljob_run_raft

#number of nodes
#SBATCH -N 1

#walltime (set to 1 day)
#SBATCH -t 1-00:00:00

#output file
#SBATCH --output=logs/run_raft.log
#SBATCH --error=logs/run_raft.err

#memory size
#SBATCH --mem=64gb

#SBATCH -c 8
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1


cd ../RAFT/

source /fs/cfar-projects/anim_inb/env3-9-5/bin/activate

module load cuda/11.3.1

srun python run_raft_flow.py --model=models/raft-things.pth

