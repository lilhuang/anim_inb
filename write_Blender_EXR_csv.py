import os
import re
import csv
import random
import pyexr
import cv2
import argparse
import numpy as lumpy

import pdb


def split_train_test(dataroot):
    all_examples = os.listdir(dataroot)
    random.shuffle(all_examples)

    idx80 = int(len(all_examples)*0.8)

    #split is 80-20 train to test
    return all_examples[:idx80], all_examples[idx80:]


def check_valid_patches(img1, img3):
    patch_size = 512

    leftmost_idx = 0
    rightmost_idx = img1.shape[1] - patch_size - 1
    topmost_idx = 0
    bottommost_idx = img1.shape[0] - patch_size - 1

    patches_1_names = []
    patches_3_names = [] 

    for i in range(topmost_idx, bottommost_idx, 64):
        for j in range(leftmost_idx, rightmost_idx, 64):
            name = str(i)+"_"+str(j)

            patch_1 = img1[i:i+patch_size, j:j+patch_size]
            patch_3 = img3[i:i+patch_size, j:j+patch_size]

            patch_1 = 1 - patch_1
            patch_3 = 1 - patch_3

            #all we care about is if suzanne is in it at all
            #flow should basically never be zero
            if lumpy.sum(patch_1) >= 5500:
                patches_1_names.append(name)
            if lumpy.sum(patch_3) >= 5500:
                patches_3_names.append(name)

    used_names = []
    perm = lumpy.random.permutation(len(patches_1_names))
    for i in perm:
        if len(used_names) >= 4:
            break
        patchname = patches_1_names[i]
        if patchname in patches_3_names and patchname not in used_names:
            print("valid")
            #just want to make sure the image still exists in the third frame (didn't exit frame entirely lmao)
            used_names.append(patchname)
        else:
            print("invalid :(")
    return used_names


def check_valid_patches_5fin(img1, img3, img5):
    patch_size = 512

    leftmost_idx = 0
    rightmost_idx = img1.shape[1] - patch_size - 1
    topmost_idx = 0
    bottommost_idx = img1.shape[0] - patch_size - 1

    patches_1_names = []
    patches_3_names = [] 
    patches_5_names = []

    for i in range(topmost_idx, bottommost_idx, 64):
        for j in range(leftmost_idx, rightmost_idx, 64):
            name = str(i)+"_"+str(j)

            patch_1 = img1[i:i+patch_size, j:j+patch_size]
            patch_3 = img3[i:i+patch_size, j:j+patch_size]
            patch_5 = img5[i:i+patch_size, j:j+patch_size]

            patch_1 = 1 - patch_1
            patch_3 = 1 - patch_3
            patch_5 = 1 - patch_5

            #all we care about is if suzanne is in it at all
            #flow should basically never be zero
            if lumpy.sum(patch_1) >= 5500:
                patches_1_names.append(name)
            if lumpy.sum(patch_3) >= 5500:
                patches_3_names.append(name)
            if lumpy.sum(patch_5) >= 5500:
                patches_5_names.append(name)

    used_names = []
    perm = lumpy.random.permutation(len(patches_1_names))
    for i in perm:
        if len(used_names) >= 4:
            break
        patchname = patches_1_names[i]
        if patchname in patches_3_names and patchname in patches_5_names and patchname not in used_names:
            print("valid")
            #just want to make sure the image still exists in the third/fifth frame (didn't exit frame entirely lmao)
            used_names.append(patchname)
        else:
            print("invalid :(")
    return used_names


def get_image_from_exr(exr_filename):
    frame_exr = pyexr.open(exr_filename)
    frame_R = frame_exr.get("Composite.Combined.R")
    frame_G = frame_exr.get("Composite.Combined.G")
    frame_B = frame_exr.get("Composite.Combined.B")
    frame = lumpy.concatenate((frame_R, frame_G, frame_B), axis=2)
    frame_resized = cv2.resize(frame, (2048, 1024))
    frame_binarized = lumpy.where(frame_resized > 0.5, 1, 0)

    return frame_binarized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', type=int, required=True)
    args = parser.parse_args()

    dataroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr"

    csvroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_csv"
    if not os.path.exists(csvroot):
        os.makedirs(csvroot)

    num_ibs = [1, 3, 7]

    train_examples_filename = "train_examples_Suzanne_exr.txt"
    test_examples_filename = "test_examples_Suzanne_exr.txt"
    if not os.path.exists(train_examples_filename) or not os.path.exists(test_examples_filename):
        train_examples, test_examples = split_train_test(dataroot)
        if ".DS_Store" in train_examples:
            train_examples.remove(".DS_Store")
        elif ".DS_Store" in test_examples:
            test_examples.remove(".DS_Store")
        with open(train_examples_filename, "w") as train_ex_file:
            train_ex_file.write("\n".join(train_examples) + "\n")
        with open(test_examples_filename, "w") as test_ex_file:
            test_ex_file.writelines("\n".join(test_examples) + "\n")
    else:
        train_ex_file = open(train_examples_filename, "r")
        train_examples = train_ex_file.readlines()
        test_ex_file = open(test_examples_filename, "r")
        test_examples = test_ex_file.readlines()
        train_ex_file.close()
        test_ex_file.close()
    
    if args.example >= len(train_examples):
        return
        cur_example = test_examples[args.example - len(train_examples)].rstrip()
    else:
        cur_example = train_examples[args.example].rstrip()

    num_frames = 242
    regex = "(t_.*_s_.*_r_.*_[0-9])_exr"

    for ib in num_ibs:
        if args.example >= len(train_examples):
            csvroot2 = os.path.join(csvroot, "test", str(ib)+"ib")
        else:
            csvroot2 = os.path.join(csvroot, "train", str(ib)+"ib")
        if not os.path.exists(csvroot2):
            os.makedirs(csvroot2)
        
        csv_filename = os.path.join(csvroot2, cur_example+".csv")
        csvfile = open(csv_filename, "w", newline="")

        writer = csv.writer(csvfile, delimiter = ' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        # if not re.search(regex, cur_example):
        #     continue
        print(cur_example)
        # ex_core_name = re.search(regex, train_ex).group(1)
        # print(ex_core_name)
        for frame in range(num_frames):
            if frame + ib + 1 >= num_frames:
                break
            ibframe = frame + (ib//2) + 1
            trgframe = frame + ib + 1
            print(frame)
            frame1_exr_path = os.path.join(dataroot, cur_example, "multilayer_{:04d}.exr".format(frame))
            frame2_exr_path = os.path.join(dataroot, cur_example, "multilayer_{:04d}.exr".format(ibframe))
            frame3_exr_path = os.path.join(dataroot, cur_example, "multilayer_{:04d}.exr".format(trgframe))

            csv_row = [frame1_exr_path, frame2_exr_path, frame3_exr_path]
            if args.example < len(train_examples):
                frame1 = get_image_from_exr(frame1_exr_path)
                frame3 = get_image_from_exr(frame3_exr_path)
                patchnames = check_valid_patches(frame1, frame3)
                if len(patchnames) > 0:
                    for patchname in patchnames:
                        csv_row.append(patchname)
                    writer.writerow(csv_row)
            else:
                writer.writerow(csv_row)

        csvfile.close()
    

def main_5fin():
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', type=int, required=True)
    args = parser.parse_args()

    dataroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr"

    csvroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_csv_5fin"
    if not os.path.exists(csvroot):
        os.makedirs(csvroot)

    num_ibs = [1, 3, 7]

    train_examples_filename = "train_examples_Suzanne_exr_5fin.txt"
    test_examples_filename = "test_examples_Suzanne_exr_5fin.txt"
    if not os.path.exists(train_examples_filename) or not os.path.exists(test_examples_filename):
        train_examples, test_examples = split_train_test(dataroot)
        if ".DS_Store" in train_examples:
            train_examples.remove(".DS_Store")
        elif ".DS_Store" in test_examples:
            test_examples.remove(".DS_Store")
        with open(train_examples_filename, "w") as train_ex_file:
            train_ex_file.write("\n".join(train_examples) + "\n")
        with open(test_examples_filename, "w") as test_ex_file:
            test_ex_file.writelines("\n".join(test_examples) + "\n")
    else:
        train_ex_file = open(train_examples_filename, "r")
        train_examples = train_ex_file.readlines()
        test_ex_file = open(test_examples_filename, "r")
        test_examples = test_ex_file.readlines()
        train_ex_file.close()
        test_ex_file.close()
    
    if args.example >= len(train_examples):
        return
        cur_example = test_examples[args.example - len(train_examples)].rstrip()
    else:
        cur_example = train_examples[args.example].rstrip()

    num_frames = 242
    regex = "(t_.*_s_.*_r_.*_[0-9])_exr"

    for ib in num_ibs:
        if args.example >= len(train_examples):
            csvroot2 = os.path.join(csvroot, "test", str(ib)+"ib")
        else:
            csvroot2 = os.path.join(csvroot, "train", str(ib)+"ib")
        if not os.path.exists(csvroot2):
            os.makedirs(csvroot2)
        
        csv_filename = os.path.join(csvroot2, cur_example+".csv")
        csvfile = open(csv_filename, "w", newline="")

        writer = csv.writer(csvfile, delimiter = ' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        # if not re.search(regex, cur_example):
        #     continue
        print(cur_example)
        # ex_core_name = re.search(regex, train_ex).group(1)
        # print(ex_core_name)
        for frame in range(num_frames):
            if frame + 2*ib + 2 >= num_frames:
                break
            ibframe_1 = frame + (ib//2) + 1
            ibframe_2 = frame + ib + (ib//2) + 2
            trgframe_1 = frame + ib + 1
            trgframe_2 = frame + 2*ib + 2
            print(frame)
            frame1_exr_path = os.path.join(dataroot, cur_example, "multilayer_{:04d}.exr".format(frame))
            frame2_exr_path = os.path.join(dataroot, cur_example, "multilayer_{:04d}.exr".format(ibframe_1))
            frame3_exr_path = os.path.join(dataroot, cur_example, "multilayer_{:04d}.exr".format(trgframe_1))
            frame4_exr_path = os.path.join(dataroot, cur_example, "multilayer_{:04d}.exr".format(ibframe_2))
            frame5_exr_path = os.path.join(dataroot, cur_example, "multilayer_{:04d}.exr".format(trgframe_2))

            csv_row = [frame1_exr_path, frame2_exr_path, frame3_exr_path, frame4_exr_path, frame5_exr_path]
            if args.example < len(train_examples):
                frame1 = get_image_from_exr(frame1_exr_path)
                frame3 = get_image_from_exr(frame3_exr_path)
                frame5 = get_image_from_exr(frame5_exr_path)
                patchnames = check_valid_patches_5fin(frame1, frame3, frame5)
                if len(patchnames) > 0:
                    for patchname in patchnames:
                        csv_row.append(patchname)
                    writer.writerow(csv_row)
            else:
                writer.writerow(csv_row)

        csvfile.close()
    

def main_5fin_from_3fin_csv():
    root_3fin = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_csv"
    root_5fin = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_csv_5fin"

    split = ["train", "test"]
    ibs = [1, 3, 7]

    for t in split:
        for ib in ibs:
            print(t, str(ib))
            newroot_5fin = os.path.join(root_5fin, t, str(ib)+"ib")
            oldroot = os.path.join(root_3fin, t, str(ib)+"ib")
            if not os.path.exists(newroot_5fin):
                os.makedirs(newroot_5fin)

            all_csv_files = os.listdir(oldroot)
            for csv_file in all_csv_files:
                oldcsv = open(os.path.join(oldroot, csv_file), "r", newline='')
                newcsv = open(os.path.join(newroot_5fin, csv_file), "w", newline='')    
                reader = csv.reader(oldcsv, delimiter=' ', quotechar="|")
                writer = csv.writer(newcsv, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                all_original_rows = []
                for row in reader:
                    all_original_rows.append(row)
                print(len(all_original_rows), "rows written to array")
                for i, row in enumerate(all_original_rows):
                    if i >= len(all_original_rows) - 2:
                        break
                    frame1 = row[0]
                    frame2 = row[1]
                    frame3 = row[2]
                    frame4 = all_original_rows[i+1][2]
                    frame5 = all_original_rows[i+2][2]
                    cur_entry = [frame1, frame3, frame3, frame4, frame5]
                    if len(row) > 3:
                        for j in range(3, len(row) - 2):
                            cur_entry.append(row[j])
                    print(cur_entry)
                    writer.writerow(cur_entry)
                oldcsv.close()
                newcsv.close()
                






if __name__ == "__main__":
    # main()
    # main_5fin()
    main_5fin_from_3fin_csv()
    # main_SU()



