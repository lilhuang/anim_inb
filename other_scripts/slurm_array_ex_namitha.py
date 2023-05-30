import os
from datetime import datetime
import argparse
import time
import socket
import pdb
import numpy as np
import glob 
​
import socket
from contextlib import closing
import glob
​
def find_free_port():
	with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
		s.bind(('', 0))
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		return s.getsockname()[1]
​
​
# Function to chec for validity of QOS
#TODO: Add time check for QOS
def check_qos(args):
    qos_dict = {"high" : {"gpu":4, "cores": 16, "mem":128},
            "medium" : {"gpu":2, "cores": 8, "mem":64},
            "default" : {"gpu":1, "cores": 4, "mem":32}}
    for key, max_value in qos_dict[args.qos].items():
        val_from_args = getattr(args, key)
        if val_from_args != None:
            if val_from_args > max_value:
                raise ValueError("Invalid paramter for {} for {}".format(key, args.qos))
        else:
            setattr(args, key, max_value)
            print("Setting {} to max value of {} in QOS {} as not specified in arguements.".format(key, max_value, args.qos))
    return args
​
​
#TODO: Add day funtionality too 
parser = argparse.ArgumentParser()
parser.add_argument('--nhrs', type=int, default=72)
parser.add_argument('--base-dir', default=f'{os.getcwd()}')
parser.add_argument('--output-dirname', default='output')
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--scav', action='store_true')
parser.add_argument('--vulcan', action='store_true')
parser.add_argument('--cml', action='store_true')
parser.add_argument('--qos', default=None, type=str, help='Qos to run')
parser.add_argument('--env', type=str, help = "Set the name of the dir you want to dump",default='env')
parser.add_argument('--gpu', default=None, type=int, help='Number of gpus')
parser.add_argument('--cores', default=None, type=int, help='Number of cpu cores')
parser.add_argument('--mem', default=None, type=int, help='RAM in G')
args = parser.parse_args()
​
output_dir = os.path.join(args.base_dir, args.output_dirname, args.env)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("Output Directory: %s" % output_dir)
​
​
​
# Setting the paramters for the scripts to run, modify as per your need
params = [(lr,lr_steps)
            for lr in [1e-3, 1e-4, 1e-5]
            for lr_steps in [ None,[10000,50000],[20000,70000] ]
          ]
​
num_commands = 0
#######################################################################
# Making text files which will store the python command to run, stdout, and error if any  
with open(f'{args.base_dir}/output/{args.env}/now.txt', "w") as nowfile,\
     open(f'{args.base_dir}/output/{args.env}/log.txt', "w") as output_namefile,\
     open(f'{args.base_dir}/output/{args.env}/err.txt', "w") as error_namefile,\
     open(f'{args.base_dir}/output/{args.env}/name.txt', "w") as namefile:
​
     # Iterate over all hyper parameters
    for idx, (lr, lr_steps) in enumerate(params):
​
​
        now = datetime.now()
        datetimestr = now.strftime("%m%d_%H%M:%S.%f")
        name = str(lr)+'_'+str(lr_steps)+ '_' + datetimestr
​
        cmd = "python train_net_video.py  --dist-url tcp://0.0.0.0:{port}".format(port=find_free_port()) 
​
        cmd += " --num-gpus 8 --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8ep.yaml \
                MODEL.WEIGHTS ytvis_2019_r50.pkl SOLVER.MAX_ITER 100000 "
​
        cmd += 'SOLVER.BASE_LR ' + str(lr)
​
        out = ' OUTPUT_DIR /fs/cfar-projects/misc_satellite_as/namithap/Mask2Former/experiments/slurm_runs/wv_LR_'+str(lr) +'_r50'
​
        if lr_steps is not None:
            cmd += " SOLVER.STEPS \"(" 
            for i in range(len(lr_steps)):
                cmd+= str(lr_steps[i]) 
                if i != len(lr_steps)-1:
                    cmd += ","
            cmd += ")\" "
        
            out += '_'+ str(lr_steps[0]) + '_' + str(lr_steps[-1])+'/'
        
        cmd += out
​
​
        #/fs/cfar-projects/misc_satellite_as/namithap/Mask2Former/overfit_experiments/LR_0.001_r50_all_60kiter_overfit
​
        
​
        num_commands +=1
        print(cmd)
​
        nowfile.write(f'{cmd}\n')
        namefile.write(f'{(os.path.join(output_dir, name))}.log\n')
        output_namefile.write(f'{(os.path.join(output_dir, name))}.log\n')
        error_namefile.write(f'{(os.path.join(output_dir, name))}.error\n')
​
###########################################################################
# Make a {name}.slurm file in the {output_dir} which defines this job.
#slurm_script_path = os.path.join(output_dir, '%s.slurm' % name)
start=1
slurm_script_path = os.path.join(output_dir, f'submit_{start}_{num_commands}.slurm')
slurm_command = "sbatch %s" % slurm_script_path
​
print('total commands: ',num_commands)
​
# Make the .slurm file
with open(slurm_script_path, 'w') as slurmfile:
    slurmfile.write("#!/bin/bash\n")
    slurmfile.write(f"#SBATCH --array=1-{num_commands}%64\n") #parallelize across commands.
    slurmfile.write("#SBATCH --output=/dev/null\n")
    slurmfile.write("#SBATCH --error=/dev/null\n")
    slurmfile.write("#SBATCH --requeue\n") #fuck. Restart the job 
    
    #slurmfile.write("#SBATCH --cpus-per-task=16\n")
    if args.scav:
        if args.cml:
            slurmfile.write("#SBATCH --account=scavenger\n")
            slurmfile.write("#SBATCH --partition scavenger\n")
            slurmfile.write("#SBATCH --gres=gpu:%d\n" % args.gpu)
        else:
            #vulcan
            slurmfile.write("#SBATCH --account=abhinav\n")
            slurmfile.write("#SBATCH --partition=scavenger\n")
            slurmfile.write("#SBATCH --qos=scavenger\n")
            slurmfile.write("#SBATCH --gres=gpu:gtx1080ti:%d\n" % args.gpu)
​
​
        slurmfile.write("#SBATCH --time=%d:00:00\n" % args.nhrs)
        slurmfile.write("#SBATCH --cpus-per-task=%d\n" % args.cores)
        slurmfile.write("#SBATCH --mem=%dG\n" % args.mem)
​
    else:
        args = check_qos(args)
        slurmfile.write("#SBATCH --qos=%s\n" % args.qos)
        slurmfile.write("#SBATCH --time=%d:00:00\n" % args.nhrs)
        
        #if args.cml:
        #    slurmfile.write("#SBATCH --gres=gpu:rtx2080ti:%d\n" % args.gpu)
        #else:
        #slurmfile.write("#SBATCH --gres=gpu:p6000:%d\n" % args.gpu)
        slurmfile.write("#SBATCH --gres=gpu:%d\n" % args.gpu)
        slurmfile.write("#SBATCH --cpus-per-task=%d\n" % args.cores)
        slurmfile.write("#SBATCH --mem=%dG\n" % args.mem)
​
        
    slurmfile.write("\n")
    slurmfile.write("cd " + args.base_dir + '\n')
    slurmfile.write("eval \"$(conda shell.zsh hook)\"" '\n')
    slurmfile.write("source /vulcanscratch/namithap/miniconda3/bin/activate \n")
    slurmfile.write("conda activate iai_sat\n")
    #slurmfile.write("export DETECTRON2_DATASETS=/fs/vulcan-projects/misc_satellite/namithap/data/image_inst_seg/smart_copy\n")
​
    # slurmfile.write(f"srun --output=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/ouput.txt | tail -n 1) $(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/jobs.txt | tail -n 1)\n")
    slurmfile.write(f"srun --output=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/log.txt | tail -n 1) --error=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/err.txt | tail -n 1)  $(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/now.txt | tail -n 1)\n")
    # slurmfile.write("rm -rf /scratch0/pulkit/ \n")
    slurmfile.write("\n")
​
print(slurm_command)
print("Running on {}, with {} gpus, {} cores, {} mem for {} hour".format(args.qos, args.gpu, args.cores, args.mem , args.nhrs))
if not args.dryrun:
   os.system("%s &" % slurm_command)