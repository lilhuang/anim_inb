import os
import re
import csv
import random
import numpy as lumpy

import pdb




def split_train_test(dataroot):
    all_examples = os.listdir(dataroot)
    random.shuffle(all_examples)

    idx80 = int(len(all_examples)*0.8)

    #split is 80-20 train to test
    return all_examples[:idx80], all_examples[idx80:]


def split_train_test_SU_small(dataroot, start, end):
    framerange = lumpy.arange(start, end+1)
    frames = []
    for i in framerange:
        curframe = "frame{:03d}.png".format(i)
        print(curframe)
        frames.append(curframe)
    random.shuffle(frames)
    idx80 = int(len(frames)*0.8)

    return frames[:idx80], frames[idx80:]


def main_SU():
    #HI MAX, PLEASE CHANGE THESE ACCORDINGLY
    dataroot = "[blah/blah/blah/]datasets/StevenHug_2048x1024"
    flowroot = "[blah/blah/blah/]datasets/flows_3ib"
    flowroot_2 = "[blah/blah/blah/]datasets/flows_1ib"
    #this next one is your choice on what to call it, it will be created by you
    csvroot = "[blah/blah/blah/]datasets/StevenHug_2048x1024_csv"
    
    if not os.path.exists(csvroot):
        os.makedirs(csvroot)
    num_ibs = [3]
    to_avoid = [71, 228, 410]

    start = 1
    end = 70
    num_frames = (end - start) + 1
    train_examples, test_examples = split_train_test_SU_small(dataroot, start, end)

    for ib in num_ibs:
        trainfile = os.path.join(csvroot, "train_triplets_2_"+str(ib)+"ib.csv")
        testfile = os.path.join(csvroot, "test_triplets_2_"+str(ib)+"ib.csv")

        traincsv = open(trainfile, "w", newline="")
        testcsv = open(testfile, "w", newline="")

        trainwriter = csv.writer(traincsv, delimiter = ' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        testwriter = csv.writer(testcsv, delimiter = ' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for framename in train_examples:
            print(framename)
            regex = "frame([0-9]+).png"
            framenum = re.search(regex, framename).group(1)
            framenum_int = int(framenum)
            framenum_end = framenum_int + ib + 1
            avoid = False
            for a in to_avoid:
                if framenum_int < a and framenum_end >= a:
                    avoid = True
            if not avoid and framenum_end <= end:
                ibframe_int = framenum_int + (ib//2) + 1

                frame1 = os.path.join(dataroot, "frame{:03d}.png".format(framenum_int))
                frame2 = os.path.join(dataroot, "frame{:03d}.png".format(ibframe_int))
                frame3 = os.path.join(dataroot, "frame{:03d}.png".format(framenum_end))

                flo_triplet = os.path.join(flowroot, "frame{:03d}_to_frame{:03d}.npz".format(framenum_int, framenum_end))
                flo_triplet_2_1 = os.path.join(flowroot_2, "frame{:03d}_to_frame{:03d}.npz".format(framenum_int, ibframe_int))
                flo_triplet_2_2 = os.path.join(flowroot_2, "frame{:03d}_to_frame{:03d}.npz".format(ibframe_int, framenum_end))

                trainwriter.writerow([frame1, frame2, frame3, flo_triplet, flo_triplet_2_1, flo_triplet_2_2])
        
        for framename in test_examples:
            print(framename)
            regex = "frame([0-9]+).png"
            framenum = re.search(regex, framename).group(1)
            framenum_int = int(framenum)
            framenum_end = framenum_int + ib + 1
            avoid = False
            for a in to_avoid:
                if framenum_int < a and framenum_end >= a:
                    avoid = True
            if not avoid and framenum_end <= end:
                ibframe_int = framenum_int + (ib//2) + 1

                frame1 = os.path.join(dataroot, "frame{:03d}.png".format(framenum_int))
                frame2 = os.path.join(dataroot, "frame{:03d}.png".format(ibframe_int))
                frame3 = os.path.join(dataroot, "frame{:03d}.png".format(framenum_end))

                flo_triplet = os.path.join(flowroot, "frame{:03d}_to_frame{:03d}.npz".format(framenum_int, framenum_end))
                flo_triplet_2_1 = os.path.join(flowroot_2, "frame{:03d}_to_frame{:03d}.npz".format(framenum_int, ibframe_int))
                flo_triplet_2_2 = os.path.join(flowroot_2, "frame{:03d}_to_frame{:03d}.npz".format(ibframe_int, framenum_end))

                testwriter.writerow([frame1, frame2, frame3, flo_triplet, flo_triplet_2_1, flo_triplet_2_2])


        traincsv.close()
        testcsv.close()


if __name__ == "__main__":
    main_SU()



