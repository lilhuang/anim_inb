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


def split_train_test_seq_halfhalf(dataroot, ib):
    all_examples = os.listdir(dataroot)
    all_examples.sort()

    if ib == 1:
        return all_examples[::2], all_examples[1::2]
    elif ib == 3:
        return all_examples[1::4], all_examples[3::4]
    else:
        print("GIMME A VALID IB")
        return 0


def split_testonly_seq_(dataroot, ib):
    all_examples = os.listdir(dataroot)
    all_examples.sort()
    if ib == 1:
        return all_examples
    elif ib == 3:
        return all_examples[1::2]
    else:
        print("GIMME A VALID IB")
        return 0


# def split_train_test_SU_small(dataroot, start, end):
#     framerange = lumpy.arange(start, end+1)
#     frames = []
#     for i in framerange:
#         curframe = "frame{:03d}.png".format(i)
#         print(curframe)
#         frames.append(curframe)
#     random.shuffle(frames)
#     idx80 = int(len(frames)*0.8)

#     return frames[:idx80], frames[idx80:]


# def split_train_test_SU_small_seq(dataroot, start, end):
#     framerange = lumpy.arange(start, end+1)
#     frames = []
#     for i in framerange:
#         curframe = "frame{:03d}.png".format(i)
#         print(curframe)
#         frames.append(curframe)
#     idx80 = int(len(frames)*0.8)

#     return frames[:idx80], frames[idx80:]

def split_train_test_SU_small_seq_halfhalf(dataroot, start, end):
    framerange = lumpy.arange(start, end+1)
    frames = []
    for i in framerange:
        curframe = "frame{:03d}.png".format(i)
        print(curframe)
        frames.append(curframe)

    # return frames[::2], frames[1::2]
    return frames[1::4], frames[3::4]


def split_test_only_SU_small_seq_halfhalf(dataroot, start, end):
    framerange = lumpy.arange(start, end+1)
    frames = []
    for i in framerange:
        curframe = "frame{:03d}.png".format(i)
        print(curframe)
        frames.append(curframe)
    
    return frames[1::2]


def main_SU_testonly():
    dataroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog/StevenHug_2048x1024"
    flowroot = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_2048x1024"
    flowroot_2 = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_2048x1024_1ib"

    csvroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_smol_sequential_halfhalf_testonly_2_csv"
    
    if not os.path.exists(csvroot):
        os.makedirs(csvroot)
    num_ibs = [3]

    start = 1
    end = 70
    num_frames = (end - start) + 1
    test_examples = split_test_only_SU_small_seq_halfhalf(dataroot, start, end)

    for ib in num_ibs:
        testfile = os.path.join(csvroot, "test_triplets_"+str(ib)+"ib.csv")

        testcsv = open(testfile, "w", newline="")
        testwriter = csv.writer(testcsv, delimiter = ' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        for framename in test_examples:
            print(framename)
            regex = "frame([0-9]+).png"
            framenum = re.search(regex, framename).group(1)
            framenum_int = int(framenum)
            framenum_end = framenum_int + ib + 1
            framenum_end_5 = framenum_int + 2*ib + 2
            if framenum_end <= end:
                ibframe_int = framenum_int + (ib//2) + 1
                ib2frame_int = framenum_int + ib + (ib//2) + 2

                frame1 = os.path.join(dataroot, "frame{:03d}.png".format(framenum_int))
                frame2 = os.path.join(dataroot, "frame{:03d}.png".format(ibframe_int))
                frame3 = os.path.join(dataroot, "frame{:03d}.png".format(framenum_end))
                frame4 = os.path.join(dataroot, "frame{:03d}.png".format(ib2frame_int))
                frame5 = os.path.join(dataroot, "frame{:03d}.png".format(framenum_end_5))

                flo_triplet = os.path.join(flowroot, "frame{:03d}_to_frame{:03d}.npz".format(framenum_int, framenum_end))
                flo_quintlet = os.path.join(flowroot, "frame{:03d}_to_frame{:03d}.npz".format(framenum_int, framenum_end_5))

                flo_triplet_2_1 = os.path.join(flowroot_2, "frame{:03d}_to_frame{:03d}.npz".format(framenum_int, ibframe_int))
                flo_triplet_2_2 = os.path.join(flowroot_2, "frame{:03d}_to_frame{:03d}.npz".format(ibframe_int, framenum_end))

                testwriter.writerow([frame1, frame2, frame3, flo_triplet, flo_triplet_2_1, flo_triplet_2_2])
                
        testcsv.close()


def main_SU():
    dataroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog/StevenHug_2048x1024"
    # flowroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_tvl1_flows"
    # flowroot = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_640x360_3ib"
    flowroot = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_2048x1024"
    flowroot_2 = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_2048x1024_1ib"

    # csvroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_640x360_smol_csv
    # csvroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_smol_sequential_csv"
    csvroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_smol_sequential_halfhalf_2_csv"
    
    if not os.path.exists(csvroot):
        os.makedirs(csvroot)
    # num_ibs = [1, 3, 7]
    num_ibs = [3]
    # to_avoid = [71, 228, 410]
    # num_ibs = [7]

    start = 1
    # end = 577
    end = 70
    num_frames = (end - start) + 1
    train_examples, test_examples = split_train_test_SU_small_seq_halfhalf(dataroot, start, end)

    for ib in num_ibs:
        trainfile = os.path.join(csvroot, "train_triplets_"+str(ib)+"ib.csv")
        testfile = os.path.join(csvroot, "test_triplets_"+str(ib)+"ib.csv")
        trainfile_5 = os.path.join(csvroot, "train_quintlets_"+str(ib)+"ib.csv")
        testfile_5 = os.path.join(csvroot, "test_quintlets_"+str(ib)+"ib.csv")
        # trainfile = os.path.join(csvroot, "train_triplets_2_"+str(ib)+"ib.csv")
        # testfile = os.path.join(csvroot, "test_triplets_2_"+str(ib)+"ib.csv")

        traincsv = open(trainfile, "w", newline="")
        testcsv = open(testfile, "w", newline="")
        # traincsv_5 = open(trainfile_5, "w", newline="")
        # testcsv_5 = open(testfile_5, "w", newline="")

        trainwriter = csv.writer(traincsv, delimiter = ' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        testwriter = csv.writer(testcsv, delimiter = ' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # trainwriter_5 = csv.writer(traincsv_5, delimiter = ' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # testwriter_5 = csv.writer(testcsv_5, delimiter = ' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for framename in train_examples:
            print(framename)
            regex = "frame([0-9]+).png"
            framenum = re.search(regex, framename).group(1)
            framenum_int = int(framenum)
            framenum_end = framenum_int + ib + 1
            framenum_end_5 = framenum_int + 2*ib + 2
            if framenum_end <= end:
                ibframe_int = framenum_int + (ib//2) + 1
                ib2frame_int = framenum_int + ib + (ib//2) + 2

                frame1 = os.path.join(dataroot, "frame{:03d}.png".format(framenum_int))
                frame2 = os.path.join(dataroot, "frame{:03d}.png".format(ibframe_int))
                frame3 = os.path.join(dataroot, "frame{:03d}.png".format(framenum_end))
                frame4 = os.path.join(dataroot, "frame{:03d}.png".format(ib2frame_int))
                frame5 = os.path.join(dataroot, "frame{:03d}.png".format(framenum_end_5))

                flo_triplet = os.path.join(flowroot, "frame{:03d}_to_frame{:03d}.npz".format(framenum_int, framenum_end))
                flo_quintlet = os.path.join(flowroot, "frame{:03d}_to_frame{:03d}.npz".format(framenum_int, framenum_end_5))
                
                flo_triplet_2_1 = os.path.join(flowroot_2, "frame{:03d}_to_frame{:03d}.npz".format(framenum_int, ibframe_int))
                flo_triplet_2_2 = os.path.join(flowroot_2, "frame{:03d}_to_frame{:03d}.npz".format(ibframe_int, framenum_end))

                trainwriter.writerow([frame1, frame2, frame3, flo_triplet, flo_triplet_2_1, flo_triplet_2_2])

                # if not avoid_5 and framenum_end_5 <= end:
                #     trainwriter_5.writerow([frame1, frame2, frame3, frame4, frame5, flo_quintlet])
        
        for framename in test_examples:
            print(framename)
            regex = "frame([0-9]+).png"
            framenum = re.search(regex, framename).group(1)
            framenum_int = int(framenum)
            framenum_end = framenum_int + ib + 1
            framenum_end_5 = framenum_int + 2*ib + 2
            if framenum_end <= end:
                ibframe_int = framenum_int + (ib//2) + 1
                ib2frame_int = framenum_int + ib + (ib//2) + 2

                frame1 = os.path.join(dataroot, "frame{:03d}.png".format(framenum_int))
                frame2 = os.path.join(dataroot, "frame{:03d}.png".format(ibframe_int))
                frame3 = os.path.join(dataroot, "frame{:03d}.png".format(framenum_end))
                frame4 = os.path.join(dataroot, "frame{:03d}.png".format(ib2frame_int))
                frame5 = os.path.join(dataroot, "frame{:03d}.png".format(framenum_end_5))

                flo_triplet = os.path.join(flowroot, "frame{:03d}_to_frame{:03d}.npz".format(framenum_int, framenum_end))
                flo_quintlet = os.path.join(flowroot, "frame{:03d}_to_frame{:03d}.npz".format(framenum_int, framenum_end_5))

                flo_triplet_2_1 = os.path.join(flowroot_2, "frame{:03d}_to_frame{:03d}.npz".format(framenum_int, ibframe_int))
                flo_triplet_2_2 = os.path.join(flowroot_2, "frame{:03d}_to_frame{:03d}.npz".format(ibframe_int, framenum_end))

                testwriter.writerow([frame1, frame2, frame3, flo_triplet, flo_triplet_2_1, flo_triplet_2_2])
                
                # if not avoid_5 and framenum_end_5 <= end:
                #     testwriter_5.writerow([frame1, frame2, frame3, frame4, frame5, flo_quintlet])


        traincsv.close()
        testcsv.close()
        # traincsv_5.close()
        # testcsv_5.close()



def main():
    # dataroot = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_dog"
    dataroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_dog"
    # flowroot = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_tvl1_flows"
    flowroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_tvl1_flows"

    # csvroot = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_csv"
    csvroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_csv"
    if not os.path.exists(csvroot):
        os.makedirs(csvroot)
    # num_ibs = [1, 3, 7]
    num_ibs = [1, 3]
    # num_ibs = [7]

    train_examples, test_examples = split_train_test(dataroot)
    num_frames = 242

    for ib in num_ibs:
        trainfile = os.path.join(csvroot, "train_triplets_"+str(ib)+"ib.csv")
        testfile = os.path.join(csvroot, "test_triplets_"+str(ib)+"ib.csv")

        traincsv = open(trainfile, "w", newline="")
        testcsv = open(testfile, "w", newline="")

        trainwriter = csv.writer(traincsv, delimiter = ' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        testwriter = csv.writer(testcsv, delimiter = ' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for train_ex in train_examples:
            print(train_ex)
            regex = "(t_.*_s_.*_r_.*_[0-9])_png_2048x1024"
            ex_core_name = re.search(regex, train_ex).group(1)
            print(ex_core_name)
            flow_train_ex = ex_core_name+"_jpg"
            for frame in range(num_frames):
                if frame + ib + 1 >= num_frames:
                    break
                ibframe = frame + (ib//2) + 1
                trgframe = frame + ib + 1
                frame1 = os.path.join(dataroot, train_ex, "frame_{:04d}.png".format(frame))
                frame2 = os.path.join(dataroot, train_ex, "frame_{:04d}.png".format(ibframe))
                frame3 = os.path.join(dataroot, train_ex, "frame_{:04d}.png".format(trgframe))

                flo13_x = os.path.join(flowroot, flow_train_ex, "flow_x_{:04d}_to_{:04d}.jpg".format(frame, trgframe))
                flo31_x = os.path.join(flowroot, flow_train_ex, "flow_x_{:04d}_to_{:04d}.jpg".format(trgframe, frame))
                # flo13_dt_x = os.path.join(flowroot, train_ex, "flo_{:04d}_to_{:04d}_dt_x.png".format(frame, trgframe))
                # flo31_dt_x = os.path.join(flowroot, train_ex, "flo_{:04d}_to_{:04d}_dt_x.png".format(trgframe, frame))

                flo13_y = os.path.join(flowroot, flow_train_ex, "flow_y_{:04d}_to_{:04d}.jpg".format(frame, trgframe))
                flo31_y = os.path.join(flowroot, flow_train_ex, "flow_y_{:04d}_to_{:04d}.jpg".format(trgframe, frame))
                # flo13_dt_y = os.path.join(flowroot, train_ex, "flo_{:04d}_to_{:04d}_dt_y.png".format(frame, trgframe))
                # flo31_dt_y = os.path.join(flowroot, train_ex, "flo_{:04d}_to_{:04d}_dt_y.png".format(trgframe, frame))

                # trainwriter.writerow([frame1, frame2, frame3, flo13_x, flo13_y, flo31_x, flo31_y, flo13_dt_x, flo13_dt_y, flo31_dt_x, flo31_dt_y])
                trainwriter.writerow([frame1, frame2, frame3, flo13_x, flo13_y, flo31_x, flo31_y])
        
        for test_ex in test_examples:
            print(test_ex)
            regex = "(t_.*_s_.*_r_.*_[0-9])_png_2048x1024"
            ex_core_name = re.search(regex, test_ex).group(1)
            print(ex_core_name)
            flow_test_ex = ex_core_name+"_jpg"
            for frame in range(num_frames):
                if frame + ib + 1 >= num_frames:
                    break
                ibframe = frame + (ib//2) + 1
                trgframe = frame + ib + 1
                frame1 = os.path.join(dataroot, test_ex, "frame_{:04d}.png".format(frame))
                frame2 = os.path.join(dataroot, test_ex, "frame_{:04d}.png".format(ibframe))
                frame3 = os.path.join(dataroot, test_ex, "frame_{:04d}.png".format(trgframe))

                flo13_x = os.path.join(flowroot, flow_test_ex, "flow_x_{:04d}_to_{:04d}.jpg".format(frame, trgframe))
                flo31_x = os.path.join(flowroot, flow_test_ex, "flow_x_{:04d}_to_{:04d}.jpg".format(trgframe, frame))
                # flo13_dt_x = os.path.join(flowroot, test_ex, "flo_{:04d}_to_{:04d}_dt_x.png".format(frame, trgframe))
                # flo31_dt_x = os.path.join(flowroot, test_ex, "flo_{:04d}_to_{:04d}_dt_x.png".format(trgframe, frame))

                flo13_y = os.path.join(flowroot, flow_test_ex, "flow_y_{:04d}_to_{:04d}.jpg".format(frame, trgframe))
                flo31_y = os.path.join(flowroot, flow_test_ex, "flow_y_{:04d}_to_{:04d}.jpg".format(trgframe, frame))
                # flo13_dt_y = os.path.join(flowroot, test_ex, "flo_{:04d}_to_{:04d}_dt_y.png".format(frame, trgframe))
                # flo31_dt_y = os.path.join(flowroot, test_ex, "flo_{:04d}_to_{:04d}_dt_y.png".format(trgframe, frame))

                # testwriter.writerow([frame1, frame2, frame3, flo13_x, flo13_y, flo31_x, flo31_y, flo13_dt_x, flo13_dt_y, flo31_dt_x, flo31_dt_y])
                testwriter.writerow([frame1, frame2, frame3, flo13_x, flo13_y, flo31_x, flo31_y])


        traincsv.close()
        testcsv.close()


def main_pencil_tests():
    root = "/fs/cfar-projects/anim_inb/datasets/vid_pngs_dog"
    flowroot = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_vid_pngs_dog"
    csvroot = "/fs/cfar-projects/anim_inb/datasets/vid_pngs_csvs_halfhalf_seq"
    
    if not os.path.exists(csvroot):
        os.makedirs(csvroot)
    
    regex = "1s"
    regex_frame = "frame_([0-9]+).png"
    all_pencil_tests = os.listdir(root)
    for pt in all_pencil_tests:
        fullroot = os.path.join(root, pt)
        if re.search(regex, pt):
            ib = 1
        else:
            ib = 3
        # train_examples, test_examples = split_train_test(fullroot)
        train_examples, test_examples = split_train_test_seq_halfhalf(fullroot, ib)
        all_examples = os.listdir(fullroot)
        all_examples.sort()
        final_ex = all_examples[-1]
        end = int(re.search(regex_frame, final_ex).group(1))

        trainfile = os.path.join(csvroot, "train_"+pt+"_"+str(ib)+"ib.csv")
        testfile = os.path.join(csvroot, "test_"+pt+"_"+str(ib)+"ib.csv")

        traincsv = open(trainfile, "w", newline="")
        testcsv = open(testfile, "w", newline="")

        trainwriter = csv.writer(traincsv, delimiter = ' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        testwriter = csv.writer(testcsv, delimiter = ' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for framename in train_examples:
            print(pt, framename)
            framenum = re.search(regex_frame, framename).group(1)
            framenum_int = int(framenum)
            framenum_end = framenum_int + ib + 1
            if framenum_end <= end:
                ibframe_int = framenum_int + (ib//2) + 1

                frame1 = os.path.join(root, pt, "frame_{:03d}.png".format(framenum_int))
                frame2 = os.path.join(root, pt, "frame_{:03d}.png".format(ibframe_int))
                frame3 = os.path.join(root, pt, "frame_{:03d}.png".format(framenum_end))

                flo_triplet = os.path.join(flowroot, pt+"_2048x1024_"+str(ib)+"ib", "frame_{:03d}_to_frame_{:03d}.npz".format(framenum_int, framenum_end))
                flo_triplet_2_1 = os.path.join(flowroot, pt+"_2048x1024_"+str(ib//2)+"ib", "frame_{:03d}_to_frame_{:03d}.npz".format(framenum_int, ibframe_int))
                flo_triplet_2_2 = os.path.join(flowroot, pt+"_2048x1024_"+str(ib//2)+"ib", "frame_{:03d}_to_frame_{:03d}.npz".format(ibframe_int, framenum_end))

                trainwriter.writerow([frame1, frame2, frame3, flo_triplet, flo_triplet_2_1, flo_triplet_2_2])
        
        for framename in test_examples:
            print(pt, framename)

            framenum = re.search(regex_frame, framename).group(1)
            framenum_int = int(framenum)
            framenum_end = framenum_int + ib + 1
            if framenum_end <= end:
                ibframe_int = framenum_int + (ib//2) + 1

                frame1 = os.path.join(root, pt, "frame_{:03d}.png".format(framenum_int))
                frame2 = os.path.join(root, pt, "frame_{:03d}.png".format(ibframe_int))
                frame3 = os.path.join(root, pt, "frame_{:03d}.png".format(framenum_end))

                flo_triplet = os.path.join(flowroot, pt+"_2048x1024_"+str(ib)+"ib", "frame_{:03d}_to_frame_{:03d}.npz".format(framenum_int, framenum_end))
                flo_triplet_2_1 = os.path.join(flowroot, pt+"_2048x1024_"+str(ib//2)+"ib", "frame_{:03d}_to_frame_{:03d}.npz".format(framenum_int, ibframe_int))
                flo_triplet_2_2 = os.path.join(flowroot, pt+"_2048x1024_"+str(ib//2)+"ib", "frame_{:03d}_to_frame_{:03d}.npz".format(ibframe_int, framenum_end))

                testwriter.writerow([frame1, frame2, frame3, flo_triplet, flo_triplet_2_1, flo_triplet_2_2])


        traincsv.close()
        testcsv.close()




if __name__ == "__main__":
    # main()
    # main_SU()
    # main_SU_testonly()
    main_pencil_tests()



