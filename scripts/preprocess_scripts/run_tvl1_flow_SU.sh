#!/bin/bash

#job name
#SBATCH -J lil_flowjob

#number of nodes
#SBATCH -N 1

#walltime (set to 3 days)
#SBATCH -t 1-00:00:00

#output file
#SBATCH --error=logs/run_tvl1_flow_SU.err
#SBATCH --output=logs/run_tvl1_flow_SU.log

#memory size
#SBATCH --mem=128gb

#SBATCH -c 8
#SBATCH -p dpart
#SBATCH --qos=high
#SBATCH --gres=gpu:1

# set -x

source /vulcanscratch/lilhuang/env3-7-6/bin/activate
module load cuda/10.1.243

cd /fs/vulcan-projects/anim_inb_lilhuang/datasets/

DATASETDIR=/fs/vulcan-projects/anim_inb_lilhuang/datasets/SU_24fps/preprocess_dog
PREPROCESSDIR=/fs/vulcan-projects/anim_inb_lilhuang/datasets/SU_24fps/preprocess_dog
CSVDIR=/fs/vulcan-projects/anim_inb_lilhuang/datasets/SU_csv
FLOWDIR=/fs/vulcan-projects/anim_inb_lilhuang/datasets/SU_24fps/preprocess_dog_flows
WORKINGFLOWDIR=/fs/vulcan-projects/anim_inb_lilhuang/datasets/working_flow_dir

if [ ! -d $WORKINGFLOWDIR ]
then
    mkdir $WORKINGFLOWDIR
fi

cd $DATASETDIR

ORIGDIRNAME_REGEX="([.*]_jpg)_256x128$"

CSVFILE_TEST=${CSVDIR}"/SU_triplet_7ib.csv"

# COUNT=0

while IFS=" " read -r frame1 frame2 frame3
    do
        # if [[ COUNT -ge 1 ]]
        # then
        #     break
        # fi
        # ((COUNT=COUNT+1))

        frame3=${frame3%$'\r'}

        REGEX="frame([0-9]+).png"

        if [[ $frame1 =~ $REGEX ]]
        then
            FRAME1_NUM="${BASH_REMATCH[1]}"
        fi

        if [[ $frame3 =~ $REGEX ]]
        then
            FRAME3_NUM="${BASH_REMATCH[1]}"
        fi

        FRAME1_PATH=$PREPROCESSDIR"/StevenHug_256x128/frame"$FRAME1_NUM".jpg"
        FRAME3_PATH=$PREPROCESSDIR"/StevenHug_256x128/frame"$FRAME3_NUM".jpg"

        echo $FRAME1_PATH $FRAME3_PATH

        if [ ! -d  ${FLOWDIR}"/StevenHug_256x128_7ib" ]
        then
            mkdir ${FLOWDIR}"/StevenHug_256x128_7ib"
        fi 

        echo ${FLOWDIR}"/StevenHug_256x128_7ib"

        cp $FRAME1_PATH $WORKINGFLOWDIR
        cp $FRAME3_PATH $WORKINGFLOWDIR

        cd /fs/vulcan-projects/anim_inb_lilhuang/datasets

        /vulcanscratch/rssaketh/actionlets/actionlet/src/denseflow/build/denseflow ${WORKINGFLOWDIR} -b=100 -a=tvl1 -s=1 -if -v

        ls ./working_flow_dir
        echo !!!

        mv ${WORKINGFLOWDIR}/flow_x_* ${FLOWDIR}"/StevenHug_256x128_7ib/flow_x_"${FRAME1_NUM}"_to_"${FRAME3_NUM}".jpg"
        mv ${WORKINGFLOWDIR}/flow_y_* ${FLOWDIR}"/StevenHug_256x128_7ib/flow_y_"${FRAME1_NUM}"_to_"${FRAME3_NUM}".jpg"

        mv ${WORKINGFLOWDIR}/frame${FRAME1_NUM}.jpg ${WORKINGFLOWDIR}/TEMP.jpg
        mv ${WORKINGFLOWDIR}/frame${FRAME3_NUM}.jpg ${WORKINGFLOWDIR}/frame${FRAME1_NUM}.jpg
        mv ${WORKINGFLOWDIR}/TEMP.jpg ${WORKINGFLOWDIR}/frame${FRAME3_NUM}.jpg

        /vulcanscratch/rssaketh/actionlets/actionlet/src/denseflow/build/denseflow ${WORKINGFLOWDIR} -b=100 -a=tvl1 -s=1 -if -v

        mv ${WORKINGFLOWDIR}/flow_x_* ${FLOWDIR}"/StevenHug_256x128_7ib/flow_x_"${FRAME3_NUM}"_to_"${FRAME1_NUM}".jpg"
        mv ${WORKINGFLOWDIR}/flow_y_* ${FLOWDIR}"/StevenHug_256x128_7ib/flow_y_"${FRAME3_NUM}"_to_"${FRAME1_NUM}".jpg"

        rm ${WORKINGFLOWDIR}/*

        cd $DATASETDIR


done < $CSVFILE_TEST







# for file in "$INPUTDIR"/*
# do
#   echo $file
# done





