#!/bin/bash

#job name
#SBATCH -J lil_flowjob

#number of nodes
#SBATCH -N 1

#walltime (set to 1 day)
#SBATCH -t 1-00:00:00

#output file
#SBATCH --error=logs/run_tvl1_flow_Blender_cubes_1ib.err
#SBATCH --output=logs/run_tvl1_flow_Blender_cubes_1ib.log

#memory size
#SBATCH --mem=128gb

#SBATCH -c 8
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1

# set -x

source /fs/cfar-projects/anim_inb/env3-9-5/bin/activate

module load cuda/11.3.1

cd /fs/cfar-projects/anim_inb/datasets/

DATASETDIR=/fs/cfar-projects/anim_inb/datasets/Blender_cubes_dog_patches_large/1ib
PREPROCESSDIR=/fs/cfar-projects/anim_inb/datasets/Blender_cubes_dog_patches_large/1ib
FLOWDIR=/fs/cfar-projects/anim_inb/datasets/Blender_cubes_tvl1_flows_patches_large/1ib
WORKINGFLOWDIR=/fs/cfar-projects/anim_inb/datasets/working_flow_dir

if [ ! -d $WORKINGFLOWDIR ]
then
    echo making $WORKINGFLOWDIR
    mkdir $WORKINGFLOWDIR
fi

if [ ! -d $FLOWDIR ]
then
    mkdir $FLOWDIR
fi

cd $DATASETDIR

COUNT=0

for SAMPLE_DIR in *
do
    if [[ COUNT -ge 1 ]]
    then
        break
    fi
    ((COUNT=COUNT+1))

    IMG_LIST=(${SAMPLE_DIR}/*)
    FRAME1=${IMG_LIST[0]}
    FRAME3=${IMG_LIST[2]}

    echo $FRAME1 $FRAME3

    if [ ! -d  ${FLOWDIR}"/"${SAMPLE_DIR} ]
    then
        mkdir ${FLOWDIR}"/"${SAMPLE_DIR}
    fi

    echo ${FLOWDIR}"/"${SAMPLE_DIR}

    cp ${SAMPLE_DIR}"/"${FRAME1} ${WORKINGFLOWDIR}
    cp ${SAMPLE_DIR}"/"${FRAME3} ${WORKINGFLOWDIR}

    cd /fs/cfar-projects/anim_inb/datasets

    # /vulcanscratch/rssaketh/actionlets/actionlet/src/denseflow/build/denseflow ${WORKINGFLOWDIR} -b=100 -a=tvl1 -s=1 -if -v
    /fs/cfar-projects/anim_inb/denseflow/build/denseflow ${WORKINGFLOWDIR} -b=100 -a=tvl1 -s=1 -if -v

    ls ./working_flow_dir
    echo !!!

    mv ${WORKINGFLOWDIR}/flow_x_* ${FLOWDIR}"/"${SAMPLE_DIR}"/flow_x_1_to_3.jpg"
    mv ${WORKINGFLOWDIR}/flow_y_* ${FLOWDIR}"/"${SAMPLE_DIR}"/flow_y_1_to_3.jpg"

    mv ${WORKINGFLOWDIR}/frame1.jpg ${WORKINGFLOWDIR}/TEMP.jpg
    mv ${WORKINGFLOWDIR}/frame3.jpg ${WORKINGFLOWDIR}/frame1.jpg
    mv ${WORKINGFLOWDIR}/TEMP.jpg ${WORKINGFLOWDIR}/frame3.jpg

    # /vulcanscratch/rssaketh/actionlets/actionlet/src/denseflow/build/denseflow ${WORKINGFLOWDIR} -b=100 -a=tvl1 -s=1 -if -v
    /fs/cfar-projects/anim_inb/denseflow/build/denseflow ${WORKINGFLOWDIR} -b=100 -a=tvl1 -s=1 -if -v

    mv ${WORKINGFLOWDIR}/flow_x_* ${FLOWDIR}"/"${SAMPLE_DIR}"/flow_x_3_to_1.jpg"
    mv ${WORKINGFLOWDIR}/flow_y_* ${FLOWDIR}"/"${SAMPLE_DIR}"/flow_y_3_to_1.jpg"

    rm ${WORKINGFLOWDIR}/*

    cd $DATASETDIR


done







# for file in "$INPUTDIR"/*
# do
#   echo $file
# done





