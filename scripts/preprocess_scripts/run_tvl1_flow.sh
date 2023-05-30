#!/bin/bash

#job name
#SBATCH -J lil_flowjob

#number of nodes
#SBATCH -N 1

#walltime (set to 3 days)
#SBATCH -t 1-00:00:00

#output file
#SBATCH --error=logs/run_tvl1_flow.err
#SBATCH --output=logs/run_tvl1_flow.log

#memory size
#SBATCH --mem=128gb

#SBATCH -c 8
#SBATCH -p dpart
#SBATCH --qos=high
#SBATCH --gres=gpu:1

# set -x

source /vulcanscratch/lilhuang/env3-7-6/bin/activate
module load cuda/10.1.243

cd /vulcanscratch/lilhuang/Blender/datasets/

DATASETDIR=/vulcanscratch/lilhuang/Blender/datasets/data
PREPROCESSDIR=/vulcanscratch/lilhuang/Blender/datasets/preprocess_dog
CSVDIR=/vulcanscratch/lilhuang/Blender/datasets/csv
# FLOWDIR=/vulcanscratch/lilhuang/Blender/datasets/flow
FLOWDIR=/vulcanscratch/lilhuang/Blender/datasets/preprocess_dog_flow
WORKINGFLOWDIR=/vulcanscratch/lilhuang/Blender/datasets/working_flow_dir

if [ ! -d $WORKINGFLOWDIR ]
then
    mkdir $WORKINGFLOWDIR
fi

cd $DATASETDIR

ORIGDIRNAME_REGEX="([.*]_jpg)_256x128$"

CSVFILE_TRAIN=${CSVDIR}"/train_triplets_7ib.csv"
CSVFILE_TEST=${CSVDIR}"/test_triplets_7ib.csv"

# COUNT=0

while IFS=" " read -r frame1 frame2 frame3 flow1 flow2 flow3 flow4
    do
        # if [[ COUNT -ge 100 ]]
        # then
        #     break
        # fi
        # ((COUNT=COUNT+1))

        # frame3=${frame3%$'\r'}

        REGEX=${DATASETDIR}"/(.*)/frame_([0-9]+).jpg"

        if [[ $frame1 =~ $REGEX ]]
        then
            INPUTDIR="${BASH_REMATCH[1]}"
            FRAME1_NUM="${BASH_REMATCH[2]}"
        fi

        if [[ $frame3 =~ $REGEX ]]
        then
            FRAME3_NUM="${BASH_REMATCH[2]}"
        fi

        echo $FRAME1_NUM $FRAME3_NUM $INPUTDIR

        FRAME1_PATH=$PREPROCESSDIR"/"$INPUTDIR"_256x128/frame_"$FRAME1_NUM".jpg"
        FRAME3_PATH=$PREPROCESSDIR"/"$INPUTDIR"_256x128/frame_"$FRAME3_NUM".jpg"

        if [ ! -d  ${FLOWDIR}"/"${INPUTDIR}"_256x128_7ib" ]
        then
            mkdir ${FLOWDIR}"/"${INPUTDIR}"_256x128_7ib"
        fi 

        echo ${FLOWDIR}"/"${INPUTDIR}"_256x128_7ib"

        cp $FRAME1_PATH $WORKINGFLOWDIR
        cp $FRAME3_PATH $WORKINGFLOWDIR

        cd /vulcanscratch/lilhuang/Blender/datasets

        /vulcanscratch/rssaketh/actionlets/actionlet/src/denseflow/build/denseflow ${WORKINGFLOWDIR} -b=100 -a=tvl1 -s=1 -if -v

        mv ${WORKINGFLOWDIR}/flow_x_* ${FLOWDIR}"/"${INPUTDIR}"_256x128_7ib/flow_x_"${FRAME1_NUM}"_to_"${FRAME3_NUM}".jpg"
        mv ${WORKINGFLOWDIR}/flow_y_* ${FLOWDIR}"/"${INPUTDIR}"_256x128_7ib/flow_y_"${FRAME1_NUM}"_to_"${FRAME3_NUM}".jpg"

        mv ${WORKINGFLOWDIR}/frame_${FRAME1_NUM}.jpg ${WORKINGFLOWDIR}/TEMP.jpg
        mv ${WORKINGFLOWDIR}/frame_${FRAME3_NUM}.jpg ${WORKINGFLOWDIR}/frame_${FRAME1_NUM}.jpg
        mv ${WORKINGFLOWDIR}/TEMP.jpg ${WORKINGFLOWDIR}/frame_${FRAME3_NUM}.jpg

        /vulcanscratch/rssaketh/actionlets/actionlet/src/denseflow/build/denseflow ${WORKINGFLOWDIR} -b=100 -a=tvl1 -s=1 -if -v

        mv ${WORKINGFLOWDIR}/flow_x_* ${FLOWDIR}"/"${INPUTDIR}"_256x128_7ib/flow_x_"${FRAME3_NUM}"_to_"${FRAME1_NUM}".jpg"
        mv ${WORKINGFLOWDIR}/flow_y_* ${FLOWDIR}"/"${INPUTDIR}"_256x128_7ib/flow_y_"${FRAME3_NUM}"_to_"${FRAME1_NUM}".jpg"

        rm ${WORKINGFLOWDIR}/*

        cd $DATASETDIR


done < $CSVFILE_TEST







# for file in "$INPUTDIR"/*
# do
#   echo $file
# done





