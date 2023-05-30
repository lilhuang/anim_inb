import os
import re
import cv2
import csv
import numpy as np
import pyexr
import h5py
import argparse

import pdb



def save_mask_to_img(mask, name):
    #note mask should be np array
    if mask.shape[0] == 1:
        mask = np.transpose(mask, (1, 2, 0))
    cv2.imwrite(name, (1 - mask)*255)



def _img_loader(exrfile, patchname, img_size, patch_size):
    frame_R = exrfile.get("Composite.Combined.R")
    frame_G = exrfile.get("Composite.Combined.G")
    frame_B = exrfile.get("Composite.Combined.B")

    frame = np.concatenate((frame_R, frame_G, frame_B), axis=2)
    frame_resized = cv2.resize(frame, img_size)

    if patchname != None:
        patch_stuff = patchname.split("_")
        top = int(patch_stuff[0])
        left = int(patch_stuff[1])
        frame_resized = frame_resized[top:top+patch_size, left:left+patch_size]
    mask = 1 - cv2.cvtColor(frame_resized.astype("uint8"), cv2.COLOR_BGR2GRAY)
    frame_resized = np.transpose(frame_resized, (2, 0, 1))
    mask = np.where(mask > 0.5, 1, 0)
    mask = np.expand_dims(mask, axis=0)

    return frame_resized*255, mask



def _flow_loader_h5(filename, dataname, patchname, img_size, patch_size):
    try:
        h5f = h5py.File(filename, "r")
    except:
        pdb.set_trace()

    flo13_np = h5f[dataname]["flo13"][:]
    flo31_np = h5f[dataname]["flo31"][:]
    h5f.close()

    flo13_np_resized = cv2.resize(flo13_np, img_size)
    flo31_np_resized = cv2.resize(flo31_np, img_size)

    if patchname != None:
        patch_stuff = patchname.split("_")
        top = int(patch_stuff[0])
        left = int(patch_stuff[1])
        flo13_np_resized = flo13_np_resized[top:top+patch_size, left:left+patch_size]
        flo31_np_resized = flo31_np_resized[top:top+patch_size, left:left+patch_size]

    flo13_np_resized = np.transpose(flo13_np_resized, (2, 0, 1))
    flo31_np_resized = np.transpose(flo31_np_resized, (2, 0, 1))

    return flo13_np_resized, flo31_np_resized


def _flow_loader_exr(path1, path2, path3, dataroot, patchname, img_size, patch_size):
    vectorx_arr = []
    vectory_arr = []
    vectorz_arr = []
    vectorw_arr = []

    regex = os.path.join(dataroot, "(t_.*_s_.*_r_.*_[0-9]_exr)", "multilayer_([0-9]+).exr")
    foldername = re.search(regex, path1).group(1)
    start = int(re.search(regex, path1).group(2))
    middle = int(re.search(regex, path2).group(2))
    end = int(re.search(regex, path3).group(2))

    middle_index = (end - start)//2

    for i in range(start, end+1):
        exrpath = os.path.join(dataroot, foldername, "multilayer_{:04d}.exr".format(i))
        exrfile = pyexr.open(exrpath)
        if i < end:
            vectorx_arr.append(exrfile.get("View Layer.Vector.X"))
            vectory_arr.append(exrfile.get("View Layer.Vector.Y"))
        if i > start:
            vectorz_arr.append(exrfile.get("View Layer.Vector.Z"))
            vectorw_arr.append(exrfile.get("View Layer.Vector.W"))
    
    vectorx_arr = np.asarray(vectorx_arr)
    vectory_arr = np.asarray(vectory_arr)
    vectorz_arr = np.asarray(vectorz_arr)
    vectorw_arr = np.asarray(vectorw_arr)
    
    flo13_x = np.sum(vectorz_arr, axis=0)
    flo13_y = np.sum(vectorw_arr, axis=0)
    flo31_x = -1*np.sum(vectorx_arr, axis=0)
    flo31_y = -1*np.sum(vectory_arr, axis=0)

    flo12_x = np.sum(vectorz_arr[:middle_index], axis=0)
    flo12_y = np.sum(vectorw_arr[:middle_index], axis=0)
    flo21_x = -1*np.sum(vectorx_arr[:middle_index], axis=0)
    flo21_y = -1*np.sum(vectory_arr[:middle_index], axis=0)

    flo23_x = np.sum(vectorz_arr[middle_index:], axis=0)
    flo23_y = np.sum(vectorw_arr[middle_index:], axis=0)
    flo32_x = -1*np.sum(vectorx_arr[middle_index:], axis=0)
    flo32_y = -1*np.sum(vectory_arr[middle_index:], axis=0)

    flo13_np = np.concatenate((flo13_x, flo13_y), axis=2)
    flo31_np = np.concatenate((flo31_x, flo31_y), axis=2)
    flo12_np = np.concatenate((flo12_x, flo12_y), axis=2)
    flo21_np = np.concatenate((flo21_x, flo21_y), axis=2)
    flo23_np = np.concatenate((flo23_x, flo23_y), axis=2)
    flo32_np = np.concatenate((flo32_x, flo32_y), axis=2)

    flo13_np_resized = cv2.resize(flo13_np, img_size)
    flo31_np_resized = cv2.resize(flo31_np, img_size)
    flo12_np_resized = cv2.resize(flo12_np, img_size)
    flo21_np_resized = cv2.resize(flo21_np, img_size)
    flo23_np_resized = cv2.resize(flo23_np, img_size)
    flo32_np_resized = cv2.resize(flo32_np, img_size)

    if patchname != None:
        patch_stuff = patchname.split("_")
        top = int(patch_stuff[0])
        left = int(patch_stuff[1])
        flo13_np_resized = flo13_np_resized[top:top+patch_size, left:left+patch_size]
        flo31_np_resized = flo31_np_resized[top:top+patch_size, left:left+patch_size]
        flo12_np_resized = flo12_np_resized[top:top+patch_size, left:left+patch_size]
        flo21_np_resized = flo21_np_resized[top:top+patch_size, left:left+patch_size]
        flo23_np_resized = flo23_np_resized[top:top+patch_size, left:left+patch_size]
        flo32_np_resized = flo32_np_resized[top:top+patch_size, left:left+patch_size]

    flo13_np_resized = np.transpose(flo13_np_resized, (2, 0, 1))
    flo31_np_resized = np.transpose(flo31_np_resized, (2, 0, 1))
    flo12_np_resized = np.transpose(flo12_np_resized, (2, 0, 1))
    flo21_np_resized = np.transpose(flo21_np_resized, (2, 0, 1))
    flo23_np_resized = np.transpose(flo23_np_resized, (2, 0, 1))
    flo32_np_resized = np.transpose(flo32_np_resized, (2, 0, 1))

    return flo13_np_resized, flo31_np_resized, \
            flo12_np_resized, flo21_np_resized, \
            flo23_np_resized, flo32_np_resized



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', type=int, required=True)
    parser.add_argument('--ib', type=int, required=True)
    parser.add_argument('--split', type=str, required=True)
    args = parser.parse_args()

    root = "/fs/cfar-projects/anim_inb/datasets"
    dataroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr"
    csvroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_csv"
    flowroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_flow"

    csv_root_2 = os.path.join(csvroot, args.split, str(args.ib)+"ib")
    all_csv = os.listdir(csv_root_2)
    all_csv.sort()
    if args.example >= len(all_csv):
        print("you're done")
        return
    elif args.example < 0:
        csv_name = all_csv[0]
    else:
        csv_name = all_csv[args.example]

    csv_file = os.path.join(csv_root_2, csv_name)
    flow_root_2 = os.path.join(flowroot, args.split, str(args.ib)+"ib")
    print(flow_root_2)
    regex_exr = os.path.join(dataroot, "(t_.*_s_.*_r_.*_[0-9]_exr)", "multilayer_([0-9]+).exr")
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            print("!another row!")
            src_exr = row[0]
            inb_exr = row[1]
            trg_exr = row[2]
            exr_arr = [src_exr, inb_exr, trg_exr]
            src_num = re.search(regex_exr, src_exr).group(2)
            trg_num = re.search(regex_exr, trg_exr).group(2)
            foldername = re.search(regex_exr, src_exr).group(1)
            flow_filename = os.path.join(flow_root_2, foldername, "flo.h5")
            if args.example >= 0:
                target_flow_root = os.path.join(root, "Suzanne_exr_npz_2stream_flow", args.split, \
                                                str(args.ib)+"ib", foldername)
            else:
                target_flow_root = os.path.join("jiminhearteu")
            if not os.path.exists(target_flow_root):
                os.makedirs(target_flow_root)
            if len(row) > 3:
                for j in range(3, len(row)):
                    patchname = row[j]
                    if args.example >= 0:
                        img_save_root = os.path.join(root, "Suzanne_exr_png", args.split, \
                                                    str(args.ib)+"ib", foldername, \
                                                    "example_"+str(src_num)+"_"+patchname)
                    else:
                        img_save_root = os.path.join("jiminhearteuu", "example_"+str(src_num)+"_"+patchname)
                    if not os.path.exists(img_save_root):
                        os.makedirs(img_save_root)
                    for i, exr_filename in enumerate(exr_arr):
                        img_save_path = os.path.join(img_save_root, "frame_"+str(i)+".png")
                        if not os.path.exists(img_save_path):
                            exrfile = pyexr.open(exr_filename)
                            frame, mask = _img_loader(exrfile, patchname, (2048, 1024), 512)
                            save_mask_to_img(mask, img_save_path)
                    npz_path = os.path.join(target_flow_root, "flows_"+src_num+"_to_"+trg_num+"_"+patchname)
                    if not os.path.exists(npz_path+".npz"):
                        print("save", npz_path)
                        flo13, flo31, \
                            flo12, flo21, \
                            flo23, flo32 = _flow_loader_exr(src_exr, inb_exr, trg_exr, dataroot, patchname, \
                                                            (2048, 1024), 512)
                        np.savez_compressed(npz_path, flo13=flo13, flo31=flo31, flo12=flo12, flo21=flo21, \
                                                        flo23=flo23, flo32=flo32)
            else:
                if args.example >= 0:
                    img_save_root = os.path.join(root, "Suzanne_exr_png", args.split, \
                                                str(args.ib)+"ib", foldername, \
                                                "example_"+str(src_num))
                else:
                    img_save_root = os.path.join("jiminhearteuu", "example_"+str(src_num))
                if not os.path.exists(img_save_root):
                    os.makedirs(img_save_root)
                for i, exr_filename in enumerate(exr_arr):
                    img_save_path = os.path.join(img_save_root, "frame_"+str(i)+".png")
                    if not os.path.exists(img_save_path):
                        exrfile = pyexr.open(exr_filename)
                        frame, mask = _img_loader(exrfile, None, (2048, 1024), 512)
                        save_mask_to_img(mask, img_save_path)
                npz_path = os.path.join(target_flow_root, "flows_"+src_num+"_to_"+trg_num)
                if not os.path.exists(npz_path+".npz"):
                    flo13, flo31, \
                        flo12, flo21, \
                        flo23, flo32 = _flow_loader_exr(src_exr, inb_exr, trg_exr, dataroot, None, \
                                                        (2048, 1024), 512)
                    np.savez_compressed(npz_path, flo13=flo13, flo31=flo31, flo12=flo12, flo21=flo21,\
                                                    flo23=flo23, flo32=flo32)

    # print("num folders", num_folders)


def main_5fin():
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', type=int, required=True)
    parser.add_argument('--ib', type=int, required=True)
    parser.add_argument('--split', type=str, required=True)
    args = parser.parse_args()

    root = "/fs/cfar-projects/anim_inb/datasets"
    dataroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr"
    csvroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_csv_5fin"

    csv_root_2 = os.path.join(csvroot, args.split, str(args.ib)+"ib")
    all_csv = os.listdir(csv_root_2)
    all_csv.sort()
    if args.example >= len(all_csv):
        print("you're done")
        return
    elif args.example < 0:
        csv_name = all_csv[0]
    else:
        csv_name = all_csv[args.example]

    csv_file = os.path.join(csv_root_2, csv_name)
    regex_exr = os.path.join(dataroot, "(t_.*_s_.*_r_.*_[0-9]_exr)", "multilayer_([0-9]+).exr")
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            print("!another row!")
            src_exr = row[0]
            inb_exr = row[1]
            trg_exr = row[2]
            inb2_exr = row[3]
            trg2_exr = row[4]
            exr_arr = [src_exr, inb_exr, trg_exr, inb2_exr, trg2_exr]
            src_num = re.search(regex_exr, src_exr).group(2)
            foldername = re.search(regex_exr, src_exr).group(1)
            if len(row) > 5:
                for j in range(5, len(row)):
                    patchname = row[j]
                    if args.example >= 0:
                        img_save_root = os.path.join(root, "Suzanne_exr_png_5fin", args.split, \
                                                    str(args.ib)+"ib", foldername, \
                                                    "example_"+str(src_num)+"_"+patchname)
                    else:
                        img_save_root = os.path.join("jiminhearteuu", "example_"+str(src_num)+"_"+patchname)
                    if not os.path.exists(img_save_root):
                        os.makedirs(img_save_root)
                    for i, exr_filename in enumerate(exr_arr):
                        img_save_path = os.path.join(img_save_root, "frame_"+str(i)+".png")
                        if not os.path.exists(img_save_path):
                            exrfile = pyexr.open(exr_filename)
                            frame, mask = _img_loader(exrfile, patchname, (2048, 1024), 512)
                            save_mask_to_img(mask, img_save_path)
                            print("saved to", img_save_path)
            else:
                if args.example >= 0:
                    img_save_root = os.path.join(root, "Suzanne_exr_png_5fin", args.split, \
                                                str(args.ib)+"ib", foldername, \
                                                "example_"+str(src_num))
                else:
                    img_save_root = os.path.join("jiminhearteuu", "example_"+str(src_num))
                if not os.path.exists(img_save_root):
                    os.makedirs(img_save_root)
                for i, exr_filename in enumerate(exr_arr):
                    img_save_path = os.path.join(img_save_root, "frame_"+str(i)+".png")
                    if not os.path.exists(img_save_path):
                        exrfile = pyexr.open(exr_filename)
                        frame, mask = _img_loader(exrfile, None, (2048, 1024), 512)
                        save_mask_to_img(mask, img_save_path)
                        print("saved to", img_save_path)
    




if __name__ == "__main__":
    main()
    # main_5fin()


