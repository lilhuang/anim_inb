import cv2
import numpy as lumpy
import os
import csv
import re

import pdb


def main_testset():
    tvl1_flow = cv2.optflow.DualTVL1OpticalFlow_create()

    # root = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_dog"
    # flowroot = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_tvl1_flows"
    # csvroot = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_csv"
    root = "/fs/cfar-projects/anim_inb/datasets/Suzanne_dog"
    flowroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_tvl1_flows"
    csvroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_csv"
    ib_arr = ["1ib", "3ib"]

    print("test going")

    for ib in ib_arr:
        csvfilename = os.path.join(csvroot, "test_triplets_"+ib+".csv")
        with open(csvfilename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                frame1_path = row[0]
                frame3_path = row[2]
                print(frame1_path)
                regex = "/fs/cfar-projects/anim_inb/datasets/Suzanne_dog/(.*)/frame_([0-9]+).png"
                match = re.search(regex, frame1_path)
                match2 = re.search(regex, frame3_path)
                _dir = match.group(1)
                frame1_num = match.group(2)
                frame3_num = match2.group(2)

                frame1 = cv2.imread(frame1_path, 0)
                frame3 = cv2.imread(frame3_path, 0)
                flow13 = tvl1_flow.calc(frame1, frame3, None)
                flow13 = lumpy.concatenate((flow13, lumpy.zeros((1024, 2048, 1))), axis=2)
                flow31 = tvl1_flow.calc(frame3, frame1, None)
                flow31 = lumpy.concatenate((flow31, lumpy.zeros((1024, 2048, 1))), axis=2)
                savepath = os.path.join(flowroot, ib, _dir)
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                
                cv2.imwrite(os.path.join(savepath, "flo_"+frame1_num+"_to_"+frame3_num+".png"), flow13)
                cv2.imwrite(os.path.join(savepath, "flo_"+frame3_num+"_to_"+frame1_num+".png"), flow31)



def main_SU_testset():
    tvl1_flow = cv2.optflow.DualTVL1OpticalFlow_create()

    root = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_dog_patches_large"
    flowroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_tvl1_flows"
    csvroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_csv"
    ib_arr = ["1ib", "3ib"]

    for ib in ib_arr:
        csvfilename = os.path.join(csvroot, "test_triplets_"+ib+".csv")
        with open(csvfilename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                frame1_path = row[0]
                frame3_path = row[2]
                print(frame1_path)
                regex = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog/StevenHug_2048x1024/frame([0-9]+).png"
                match = re.search(regex, frame1_path)
                match2 = re.search(regex, frame3_path)
                frame1_num = match.group(1)
                frame3_num = match2.group(1)

                # frame1 = cv2.imread(frame1_path, 0)
                # frame3 = cv2.imread(frame3_path, 0)
                # flow13 = tvl1_flow.calc(frame1, frame3, None)
                # flow13 = lumpy.concatenate((flow13, lumpy.zeros((1024, 2048, 1))), axis=2)
                # flow31 = tvl1_flow.calc(frame3, frame1, None)
                # flow31 = lumpy.concatenate((flow31, lumpy.zeros((1024, 2048, 1))), axis=2)
                savepath = os.path.join(flowroot, ib)
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                
                # cv2.imwrite(os.path.join(savepath, "flo_"+frame1_num+"_to_"+frame3_num+".png"), flow13)
                # cv2.imwrite(os.path.join(savepath, "flo_"+frame3_num+"_to_"+frame1_num+".png"), flow31)

                frame1_dt_path = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog/StevenHug_2048x1024/dt_frame"+frame1_num+".png"
                frame3_dt_path = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog/StevenHug_2048x1024/dt_frame"+frame3_num+".png"
                frame1_dt = cv2.imread(frame1_dt_path, 0)
                frame3_dt = cv2.imread(frame3_dt_path, 0)
                flow13_dt = tvl1_flow.calc(frame1_dt, frame3_dt, None)
                flow13_dt = lumpy.concatenate((flow13_dt, lumpy.zeros((1024, 2048, 1))), axis=2)
                flow31_dt = tvl1_flow.calc(frame3_dt ,frame1_dt, None)
                flow31_dt = lumpy.concatenate((flow31_dt, lumpy.zeros((1024, 2048, 1))), axis=2)
                cv2.imwrite(os.path.join(savepath, "flo_dt_"+frame1_num+"_to_"+frame3_num+".png"), flow13_dt)
                cv2.imwrite(os.path.join(savepath, "flo_dt_"+frame3_num+"_to_"+frame1_num+".png"), flow31_dt)



def main():
    tvl1_flow = cv2.optflow.DualTVL1OpticalFlow_create()

    # root = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_dog_patches_large"
    # flowroot = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_tvl1_flows_patches_large"
    # root = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_dog"
    # flowroot = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_tvl1_flows"
    # csvroot = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_csv"
    root = "/fs/cfar-projects/anim_inb/datasets/Suzanne_dog_patches_large"
    flowroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_tvl1_flows"
    csvroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_csv"
    ib_arr = ["1ib", "3ib"]

    print("train going")

    for ib in ib_arr:
        curroot = os.path.join(root, ib)
        all_dirs = os.listdir(curroot)
        for _dir in all_dirs:
            print(_dir)
            curpath = os.path.join(curroot, _dir)
            frame1_path = os.path.join(curpath, "frame1.png")
            frame3_path = os.path.join(curpath, "frame3.png")

            frame1 = cv2.imread(frame1_path, 0)
            frame3 = cv2.imread(frame3_path, 0)
            flow13 = tvl1_flow.calc(frame1, frame3, None)
            flow13 = lumpy.concatenate((flow13, lumpy.zeros((512, 512, 1))), axis=2)
            flow31 = tvl1_flow.calc(frame3, frame1, None)
            flow31 = lumpy.concatenate((flow31, lumpy.zeros((512, 512, 1))), axis=2)
            savepath = os.path.join(flowroot, ib, _dir)
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            
            cv2.imwrite(os.path.join(savepath, "flo13.png"), flow13)
            cv2.imwrite(os.path.join(savepath, "flo31.png"), flow31)


def main_SU():
    tvl1_flow = cv2.optflow.DualTVL1OpticalFlow_create()

    root = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_dog_patches_large"
    flowroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_tvl1_flows_patches_large"
    csvroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_csv"
    ib_arr = ["1ib", "3ib"]

    for ib in ib_arr:
        curroot = os.path.join(root, ib)
        all_dirs = os.listdir(curroot)
        for _dir in all_dirs:
            curpath = os.path.join(curroot, _dir)
            frame1_path = os.path.join(curpath, "frame1.png")
            frame3_path = os.path.join(curpath, "frame3.png")

            # frame1 = cv2.imread(frame1_path, 0)
            # frame3 = cv2.imread(frame3_path, 0)
            # flow13 = tvl1_flow.calc(frame1, frame3, None)
            # flow13 = lumpy.concatenate((flow13, lumpy.zeros((512, 512, 1))), axis=2)
            # flow31 = tvl1_flow.calc(frame3, frame1, None)
            # flow31 = lumpy.concatenate((flow31, lumpy.zeros((512, 512, 1))), axis=2)
            savepath = os.path.join(flowroot, ib, _dir)
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            
            # cv2.imwrite(os.path.join(savepath, "flo13.png"), flow13)
            # cv2.imwrite(os.path.join(savepath, "flo31.png"), flow31)

            frame1_dt_path = os.path.join(curpath, "frame1_dt.png")
            frame3_dt_path = os.path.join(curpath, "frame3_dt.png")

            frame1_dt = cv2.imread(frame1_dt_path, 0)
            frame3_dt = cv2.imread(frame3_dt_path, 0)
            flow13_dt = tvl1_flow.calc(frame1_dt, frame3_dt, None)
            flow13_dt = lumpy.concatenate((flow13_dt, lumpy.zeros((512, 512, 1))), axis=2)
            flow31_dt = tvl1_flow.calc(frame3_dt, frame1_dt, None)
            flow31_dt = lumpy.concatenate((flow31_dt, lumpy.zeros((512, 512, 1))), axis=2)
            cv2.imwrite(os.path.join(savepath, "flow13_dt.png"), flow13_dt)
            cv2.imwrite(os.path.join(savepath, "flow31_dt.png"), flow31_dt)



if __name__ == "__main__":
    # main()
    # main_testset()
    main_SU()
    main_SU_testset()

