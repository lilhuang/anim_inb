#!/bin/bash

#job name
#SBATCH -J lil_flowjob

#number of nodes
#SBATCH -N 1

#walltime (set to 3 days)
#SBATCH -t 1-00:00:00

#output file
#SBATCH --error=logs/run_tvl1_flow_atd12k_train.err
#SBATCH --output=logs/run_tvl1_flow_atd12k_train.log

#memory size
#SBATCH --mem=128gb

#SBATCH --account=abhinav
#SBATCH -c 8
#SBATCH -p dpart
#SBATCH --qos=high
#SBATCH --gres=gpu:1

# set -x

source /vulcanscratch/lilhuang/env3-7-6/bin/activate
module load cuda/10.1.243

cd /fs/vulcan-projects/anim_inb_lilhuang/datasets/

DATASETDIR=/fs/vulcan-projects/anim_inb_lilhuang/datasets/train_10k_preprocess_dog
PREPROCESSDIR=/fs/vulcan-projects/anim_inb_lilhuang/datasets/train_10k_preprocess_dog
FLOWDIR=/fs/vulcan-projects/anim_inb_lilhuang/datasets/train_10k_preprocess_tvl1_flows
WORKINGFLOWDIR=/fs/vulcan-projects/anim_inb_lilhuang/datasets/working_flow_dir

if [ ! -d $WORKINGFLOWDIR ]
then
    mkdir $WORKINGFLOWDIR
fi

if [ ! -d $FLOWDIR ]
then
    mkdir $FLOWDIR
fi

cd $DATASETDIR

# COUNT=0

for SAMPLE_DIR in *
do
    # if [[ COUNT -ge 1 ]]
    # then
    #     break
    # fi
    # ((COUNT=COUNT+1))

    IMG_LIST=(${SAMPLE_DIR}/*)
    FRAME1=${IMG_LIST[0]}
    FRAME3=${IMG_LIST[4]}

    echo $FRAME1 $FRAME3

    if [ ! -d  ${FLOWDIR}"/"${SAMPLE_DIR} ]
    then
        mkdir ${FLOWDIR}"/"${SAMPLE_DIR}
    fi

    echo ${FLOWDIR}"/"${SAMPLE_DIR}

    cp $FRAME1 $WORKINGFLOWDIR
    cp $FRAME3 $WORKINGFLOWDIR

    cd /fs/vulcan-projects/anim_inb_lilhuang/datasets

    /vulcanscratch/rssaketh/actionlets/actionlet/src/denseflow/build/denseflow ${WORKINGFLOWDIR} -b=100 -a=tvl1 -s=1 -if -v

    ls ./working_flow_dir
    echo !!!

    mv ${WORKINGFLOWDIR}/flow_x_* ${FLOWDIR}"/"${SAMPLE_DIR}"/flow_x_1_to_3.jpg"
    mv ${WORKINGFLOWDIR}/flow_y_* ${FLOWDIR}"/"${SAMPLE_DIR}"/flow_y_1_to_3.jpg"

    mv ${WORKINGFLOWDIR}/frame1.jpg ${WORKINGFLOWDIR}/TEMP.jpg
    mv ${WORKINGFLOWDIR}/frame3.jpg ${WORKINGFLOWDIR}/frame1.jpg
    mv ${WORKINGFLOWDIR}/TEMP.jpg ${WORKINGFLOWDIR}/frame3.jpg

    /vulcanscratch/rssaketh/actionlets/actionlet/src/denseflow/build/denseflow ${WORKINGFLOWDIR} -b=100 -a=tvl1 -s=1 -if -v

    mv ${WORKINGFLOWDIR}/flow_x_* ${FLOWDIR}"/"${SAMPLE_DIR}"/flow_x_3_to_1.jpg"
    mv ${WORKINGFLOWDIR}/flow_y_* ${FLOWDIR}"/"${SAMPLE_DIR}"/flow_y_3_to_1.jpg"

    rm ${WORKINGFLOWDIR}/*

    cd $DATASETDIR


done







# for file in "$INPUTDIR"/*
# do
#   echo $file
# done





