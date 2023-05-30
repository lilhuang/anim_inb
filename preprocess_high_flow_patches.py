import numpy as lumpy
import os
import re
import csv
import cv2
import pickle
import h5py
import random
import pyexr
from collections import OrderedDict
import matplotlib.pyplot as plt
from PIL import Image

import pdb


def plot_hist(patch_norms_13, patch_norms_31, dt_patch_norms_13, dt_patch_norms_31):
    patchlen = len(patch_norms_13)
    fig, axs = plt.subplots(2,2)
    axs[0,0].hist(patch_norms_13)
    axs[0,0].set_title("13 norms")
    axs[0,1].hist(patch_norms_31)
    axs[0,0].set_title("31 norms")
    axs[1,0].hist(dt_patch_norms_13)
    axs[0,0].set_title("dt 13 norms")
    axs[1,1].hist(dt_patch_norms_31)
    axs[0,0].set_title("dt 31 norms")

    plt.savefig("patch_norms_hist.png")
    plt.close("all")



def main_plot():
    f_norms_13 = open("patch_norms_13.pkl", "rb")
    f_norms_31 = open("patch_norms_31.pkl", "rb")
    f_norms_13_dt = open("patch_norms_13_dt.pkl", "rb")
    f_norms_31_dt = open("patch_norms_31_dt.pkl", "rb")

    norms_13 = pickle.load(f_norms_13)
    norms_31 = pickle.load(f_norms_31)
    norms_13_dt = pickle.load(f_norms_13_dt)
    norms_31_dt = pickle.load(f_norms_31_dt)

    plot_hist(norms_13, norms_31, norms_13_dt, norms_31_dt)

    f_norms_13.close()
    f_norms_31.close()
    f_norms_13_dt.close()
    f_norms_31_dt.close()

    
def make_dataset():
    dataroot = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/train_10k_preprocess_dog"
    flowroot = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/train_10k_preprocess_raft_flows"

    outroot = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/train_10k_preprocess_dog_patches"
    outflowroot = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/train_10k_preprocess_raft_flows_patches"

    patch_size = 256

    f_13 = open("patch_norms_13_names.pkl", "rb")
    # f_31 = open("patch_norms_31_names.pkl", "rb")
    f_13_dt = open("patch_norms_13_dt_names.pkl", "rb")
    # f_31_dt = open("patch_norms_31_dt_names.pkl", "rb")

    names_13 = pickle.load(f_13)
    # names_31 = pickle.load(f_31)
    names_13_dt = pickle.load(f_13_dt)
    # names_31_dt = pickle.load(f_31_dt)

    size = 10000

    names_13 = names_13[-size:]
    # names_31 = names_31[-size:]
    names_13_dt = names_13_dt[-size:]
    # names_31_dt = names_31_dt[-size:]

    names_dirs = [names_13, names_13_dt]

    for i, name_dir in enumerate(names_dirs):
        for name in names_13:
            print(name)
            regex = "(Disney_(.*)_2048x1024_t_3_k_3)_([0-9]+)_([0-9]+)"

            m =  re.search(regex, name)
            sample = m.group(1)
            topmost = int(m.group(3))
            leftmost = int(m.group(4))

            frame1_path = os.path.join(dataroot, sample, "frame1_smooth.npy")
            frame2_path = os.path.join(dataroot, sample, "frame2_smooth.npy")
            frame3_path = os.path.join(dataroot, sample, "frame3_smooth.npy")

            flo13_path = os.path.join(flowroot, sample, "flo_smooth_13.npy")
            flo31_path = os.path.join(flowroot, sample, "flo_smooth_31.npy")
            flo13_dt_path = os.path.join(flowroot, sample, "flo_smooth_dt_13.npy")
            flo31_dt_path = os.path.join(flowroot, sample, "flo_smooth_dt_31.npy")

            frame1 = lumpy.load(frame1_path)
            frame2 = lumpy.load(frame2_path)
            frame3 = lumpy.load(frame3_path)

            flo13 = lumpy.load(flo13_path)
            flo31 = lumpy.load(flo31_path)
            flo13_dt = lumpy.load(flo13_dt_path)
            flo31_dt = lumpy.load(flo31_dt_path)

            frame1_patch = frame1[topmost:topmost+patch_size, leftmost:leftmost+patch_size]
            frame2_patch = frame2[topmost:topmost+patch_size, leftmost:leftmost+patch_size]
            frame3_patch = frame3[topmost:topmost+patch_size, leftmost:leftmost+patch_size]

            flo13_patch = flo13[:,topmost:topmost+patch_size, leftmost:leftmost+patch_size]
            flo31_patch = flo31[:,topmost:topmost+patch_size, leftmost:leftmost+patch_size]
            flo13_dt_patch = flo13_dt[:,topmost:topmost+patch_size, leftmost:leftmost+patch_size]
            flo31_dt_patch = flo31_dt[:,topmost:topmost+patch_size, leftmost:leftmost+patch_size]

            outpath = os.path.join(outroot, name)
            outpath_flow = os.path.join(outflowroot, name)

            if not os.path.exists(outpath):
                os.makedirs(outpath)
            if not os.path.exists(outpath_flow):
                os.makedirs(outpath_flow)
            
            if i == 0:
                frame1_outpath = os.path.join(outpath, "frame1_smooth.npy")
                frame2_outpath = os.path.join(outpath, "frame2_smooth.npy")
                frame3_outpath = os.path.join(outpath, "frame3_smooth.npy")
            else:
                frame1_outpath = os.path.join(outpath, "frame1_smooth_dt.npy")
                frame2_outpath = os.path.join(outpath, "frame2_smooth_dt.npy")
                frame3_outpath = os.path.join(outpath, "frame3_smooth_dt.npy")
            
            flo13_outpath = os.path.join(outpath_flow, "flo_smooth_13.npy")
            flo31_outpath = os.path.join(outpath_flow, "flo_smooth_31.npy")
            flo13_dt_outpath = os.path.join(outpath_flow, "flo_smooth_dt_13.npy")
            flo31_dt_outpath = os.path.join(outpath_flow, "flo_smooth_dt_31.npy")

            lumpy.save(frame1_outpath, frame1_patch)
            lumpy.save(frame2_outpath, frame2_patch)
            lumpy.save(frame3_outpath, frame3_patch)

            lumpy.save(flo13_outpath, flo13_patch)
            lumpy.save(flo31_outpath, flo31_patch)
            lumpy.save(flo13_dt_outpath, flo13_dt_patch)
            lumpy.save(flo31_dt_outpath, flo31_dt_patch)
    

    f_13.close()
    # f_31.close()
    f_13_dt.close()
    # f_31_dt.close()


def make_large_dataset():
    dataroot = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/train_10k_preprocess_dog"
    flowroot = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/train_10k_preprocess_raft_flows"

    outroot = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/train_10k_preprocess_dog_patches_large"
    outflowroot = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/train_10k_preprocess_raft_flows_patches_large"

    patch_size = 512

    f_13 = open("patch_norms_13_names_large.pkl", "rb")
    # f_31 = open("patch_norms_31_names.pkl", "rb")
    f_13_dt = open("patch_norms_13_dt_names_large.pkl", "rb")
    # f_31_dt = open("patch_norms_31_dt_names.pkl", "rb")

    names_13 = pickle.load(f_13)
    # names_31 = pickle.load(f_31)
    names_13_dt = pickle.load(f_13_dt)
    # names_31_dt = pickle.load(f_31_dt)
    pdb.set_trace()

    size = 10000
    to_choose_from = 10*size

    names_13 = names_13[-size:]
    # names_31 = names_31[-size:]
    names_13_dt = names_13_dt[-size:]
    # names_31_dt = names_31_dt[-size:]

    names_dirs = [names_13, names_13_dt]

    for i, name_dir in enumerate(names_dirs):
        for name in names_13:
            print(name)
            regex = "(Disney_(.*)_2048x1024_t_3_k_3)_([0-9]+)_([0-9]+)"

            m =  re.search(regex, name)
            sample = m.group(1)
            topmost = int(m.group(3))
            leftmost = int(m.group(4))

            frame1_path = os.path.join(dataroot, sample, "frame1_smooth.npy")
            frame2_path = os.path.join(dataroot, sample, "frame2_smooth.npy")
            frame3_path = os.path.join(dataroot, sample, "frame3_smooth.npy")

            flo13_path = os.path.join(flowroot, sample, "flo_smooth_13.npy")
            flo31_path = os.path.join(flowroot, sample, "flo_smooth_31.npy")
            flo13_dt_path = os.path.join(flowroot, sample, "flo_smooth_dt_13.npy")
            flo31_dt_path = os.path.join(flowroot, sample, "flo_smooth_dt_31.npy")

            frame1 = lumpy.load(frame1_path)
            frame2 = lumpy.load(frame2_path)
            frame3 = lumpy.load(frame3_path)

            flo13 = lumpy.load(flo13_path)
            flo31 = lumpy.load(flo31_path)
            flo13_dt = lumpy.load(flo13_dt_path)
            flo31_dt = lumpy.load(flo31_dt_path)

            frame1_patch = frame1[topmost:topmost+patch_size, leftmost:leftmost+patch_size]
            frame2_patch = frame2[topmost:topmost+patch_size, leftmost:leftmost+patch_size]
            frame3_patch = frame3[topmost:topmost+patch_size, leftmost:leftmost+patch_size]

            flo13_patch = flo13[:,topmost:topmost+patch_size, leftmost:leftmost+patch_size]
            flo31_patch = flo31[:,topmost:topmost+patch_size, leftmost:leftmost+patch_size]
            flo13_dt_patch = flo13_dt[:,topmost:topmost+patch_size, leftmost:leftmost+patch_size]
            flo31_dt_patch = flo31_dt[:,topmost:topmost+patch_size, leftmost:leftmost+patch_size]

            outpath = os.path.join(outroot, name)
            outpath_flow = os.path.join(outflowroot, name)

            if not os.path.exists(outpath):
                os.makedirs(outpath)
            if not os.path.exists(outpath_flow):
                os.makedirs(outpath_flow)
            
            if i == 0:
                frame1_outpath = os.path.join(outpath, "frame1_smooth.npy")
                frame2_outpath = os.path.join(outpath, "frame2_smooth.npy")
                frame3_outpath = os.path.join(outpath, "frame3_smooth.npy")
            else:
                frame1_outpath = os.path.join(outpath, "frame1_smooth_dt.npy")
                frame2_outpath = os.path.join(outpath, "frame2_smooth_dt.npy")
                frame3_outpath = os.path.join(outpath, "frame3_smooth_dt.npy")
            
            flo13_outpath = os.path.join(outpath_flow, "flo_smooth_13.npy")
            flo31_outpath = os.path.join(outpath_flow, "flo_smooth_31.npy")
            flo13_dt_outpath = os.path.join(outpath_flow, "flo_smooth_dt_13.npy")
            flo31_dt_outpath = os.path.join(outpath_flow, "flo_smooth_dt_31.npy")

            lumpy.save(frame1_outpath, frame1_patch)
            lumpy.save(frame2_outpath, frame2_patch)
            lumpy.save(frame3_outpath, frame3_patch)

            lumpy.save(flo13_outpath, flo13_patch)
            lumpy.save(flo31_outpath, flo31_patch)
            lumpy.save(flo13_dt_outpath, flo13_dt_patch)
            lumpy.save(flo31_dt_outpath, flo31_dt_patch)
    

    f_13.close()
    # f_31.close()
    f_13_dt.close()
    # f_31_dt.close()
        





def main():
    dataroot = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/train_10k_preprocess_dog"
    flowroot = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/train_10k_preprocess_raft_flows"

    outroot = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/train_10k_preprocess_dog_large_patches"

    regex = "Disney_(.*)_2048x1024_t_3_k_3"

    num_samples = 1e4
    patch_size = 512

    all_samples = os.listdir(dataroot)

    patch_norms_13 = []
    patch_norms_13_names = []
    patch_norms_31 = []
    patch_norms_31_names = []
    dt_patch_norms_13 = []
    dt_patch_norms_13_names = []
    dt_patch_norms_31 = []
    dt_patch_norms_31_names = []

    for sample in all_samples:
        if not re.search(regex, sample):
            continue
        print(sample)
        flowdir = os.path.join(flowroot, sample)
        flow13 = lumpy.load(os.path.join(flowdir, "flo_smooth_13.npy"))
        flow31 = lumpy.load(os.path.join(flowdir, "flo_smooth_31.npy"))
        flow13_dt = lumpy.load(os.path.join(flowdir, "flo_smooth_dt_13.npy"))
        flow31_dt = lumpy.load(os.path.join(flowdir, "flo_smooth_dt_31.npy"))

        imgdir = os.path.join(dataroot, sample)
        img1 = lumpy.load(os.path.join(imgdir, "frame1.npy"))
        img3 = lumpy.load(os.path.join(imgdir, "frame3.npy"))
        img1 = 1 - img1
        img3 = 1 - img3

        flow13 = lumpy.multiply(flow13, img1)
        flow31 = lumpy.multiply(flow31, img3)
        flow13_dt = lumpy.multiply(flow13_dt, img1)
        flow31_dt = lumpy.multiply(flow31_dt, img3)

        leftmost_idx = 0
        rightmost_idx = flow13.shape[2] - patch_size - 1
        topmost_idx = 0
        bottommost_idx = flow13.shape[1] - patch_size - 1

        for i in range(topmost_idx, bottommost_idx, 16):
            for j in range(leftmost_idx, rightmost_idx, 16):
                name = sample+"_"+str(i)+"_"+str(j)
                patch_flow13 = flow13[:, i:i+patch_size, j:j+patch_size]
                patch_flow31 = flow31[:, i:i+patch_size, j:j+patch_size]
                patch_flow13_dt = flow13_dt[:, i:i+patch_size, j:j+patch_size]
                patch_flow31_dt = flow31_dt[:, i:i+patch_size, j:j+patch_size]

                norm13 = lumpy.linalg.norm(patch_flow13)
                norm31 = lumpy.linalg.norm(patch_flow31)
                norm13_dt = lumpy.linalg.norm(patch_flow13_dt)
                norm31_dt = lumpy.linalg.norm(patch_flow31_dt)

                patch_norms_13.append(norm13)
                patch_norms_31.append(norm31)
                dt_patch_norms_13.append(norm13_dt)
                dt_patch_norms_31.append(norm31_dt)

                patch_norms_13_names.append(name)
                patch_norms_31_names.append(name)
                dt_patch_norms_13_names.append(name)
                dt_patch_norms_31_names.append(name)

    patch_norms_13_names = [x for _, x in sorted(zip(patch_norms_13, patch_norms_13_names))]
    patch_norms_31_names = [x for _, x in sorted(zip(patch_norms_31, patch_norms_31_names))]
    dt_patch_norms_13_names = [x for _, x in sorted(zip(dt_patch_norms_13, dt_patch_norms_13_names))]
    dt_patch_norms_31_names = [x for _, x in sorted(zip(dt_patch_norms_31, dt_patch_norms_31_names))]

    patch_norms_13.sort()
    patch_norms_31.sort()
    dt_patch_norms_13.sort()
    dt_patch_norms_31.sort()

    with open("patch_norms_13_names_large.pkl", "wb") as f:
        pickle.dump(patch_norms_13_names, f)
    with open("patch_norms_31_names_large.pkl", "wb") as f:
        pickle.dump(patch_norms_31_names, f)
    with open("patch_norms_13_dt_names_large.pkl", "wb") as f:
        pickle.dump(dt_patch_norms_13_names, f)
    with open("patch_norms_31_dt_names_large.pkl", "wb") as f:
        pickle.dump(dt_patch_norms_31_names, f)
    
    with open("patch_norms_13_large.pkl", "wb") as f:
        pickle.dump(patch_norms_13, f)
    with open("patch_norms_31_large.pkl", "wb") as f:
        pickle.dump(patch_norms_31, f)
    with open("patch_norms_13_dt_large.pkl", "wb") as f:
        pickle.dump(dt_patch_norms_13, f)
    with open("patch_norms_31_dt_large.pkl", "wb") as f:
        pickle.dump(dt_patch_norms_31, f)


def png_to_np(path):
    image = cv2.imread(path, 0)
    return image


def save_npy_as_png(lumpy_arr, path):
    pdb.set_trace()
    cv2.imwrite(path, lumpy_arr)



# def save_example(patch_name, frame1_path, frame2_path, frame3_path, \
#                  flow13_x_path, flow13_y_path, flow31_x_path, flow31_y_path, ib):
def save_example(patch_name, frame1_path, frame2_path, frame3_path, ib):
    # dataroot = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_dog"
    # flowroot = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_tvl1_flows"
    # outroot = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_dog_patches_large"
    # flowoutroot = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_tvl1_flows_patches_large"
    
    dataroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog/StevenHug_2048x1024"
    # flowroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_csv"
    flowroot = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_360x640"
    outroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_dog_patches_large_smol"
    # flowoutroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_tvl1_flows_patches_large"
    
    # dataroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_dog"
    # flowroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_tvl1_flows"
    # outroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_dog_patches_large"
    
    patch_size = 512

    regex_top_left = "([0-9]+)_([0-9]+)"
    # regex_sample_frame_nums = os.path.join(dataroot, "(.*)", "frame_([0-9]+).png")
    regex_sample_frame_nums = os.path.join(dataroot, "frame([0-9]+).png")

    match_top_left = re.search(regex_top_left, patch_name)
    match_sample_frame_1 = re.search(regex_sample_frame_nums, frame1_path)

    sample = match_sample_frame_1.group(1)
    topmost = int(match_top_left.group(1))
    leftmost = int(match_top_left.group(2))

    frame1 = Image.open(frame1_path)
    frame2 = Image.open(frame2_path)
    frame3 = Image.open(frame3_path)

    # flow13_x = Image.open(flow13_x_path)
    # flow13_y = Image.open(flow13_y_path)
    # flow31_x = Image.open(flow31_x_path)
    # flow31_y = Image.open(flow31_y_path)

    # flow13_x.resize((2048, 1024))
    # flow13_y.resize((2048, 1024))
    # flow31_x.resize((2048, 1024))
    # flow31_y.resize((2048, 1024))

    frame1_patch = frame1.crop([leftmost, topmost, leftmost+patch_size, topmost+patch_size])
    frame2_patch = frame2.crop([leftmost, topmost, leftmost+patch_size, topmost+patch_size])
    frame3_patch = frame3.crop([leftmost, topmost, leftmost+patch_size, topmost+patch_size])

    # flow13_x_patch = flow13_x.crop([leftmost, topmost, leftmost+patch_size, topmost+patch_size])
    # flow13_y_patch = flow13_y.crop([leftmost, topmost, leftmost+patch_size, topmost+patch_size])
    # flow31_x_patch = flow31_x.crop([leftmost, topmost, leftmost+patch_size, topmost+patch_size])
    # flow31_y_patch = flow31_y.crop([leftmost, topmost, leftmost+patch_size, topmost+patch_size])

    name = sample + "_" + patch_name
    print(name)

    outpath = os.path.join(outroot, str(ib) + "ib", name)
    # flowoutpath = os.path.join(flowoutroot, str(ib) + "ib", name)

    if not os.path.exists(outpath):
        os.makedirs(outpath)
    # if not os.path.exists(flowoutpath):
    #     os.makedirs(flowoutpath)
            
    frame1_outpath = os.path.join(outpath, "frame1.png")
    frame2_outpath = os.path.join(outpath, "frame2.png")
    frame3_outpath = os.path.join(outpath, "frame3.png")

    # flow13_x_outpath = os.path.join(flowoutpath, "flo13_x.jpg")
    # flow13_y_outpath = os.path.join(flowoutpath, "flo13_y.jpg")
    # flow31_x_outpath = os.path.join(flowoutpath, "flo31_x.jpg")
    # flow31_y_outpath = os.path.join(flowoutpath, "flo31_y.jpg")

    frame1_patch.save(frame1_outpath)
    frame2_patch.save(frame2_outpath)
    frame3_patch.save(frame3_outpath)

    # flow13_x_patch.save(flow13_x_outpath)
    # flow13_y_patch.save(flow13_y_outpath)
    # flow31_x_patch.save(flow31_x_outpath)
    # flow31_y_patch.save(flow31_y_outpath)


def save_example_5(patch_name, frame1_path, frame2_path, frame3_path, frame4_path, frame5_path, ib):    
    dataroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog/StevenHug_2048x1024"
    # flowroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_csv"
    flowroot = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_360x640"
    outroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_dog_patches_large_5fin"
    # flowoutroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_tvl1_flows_patches_large"
    
    patch_size = 512

    regex_top_left = "([0-9]+)_([0-9]+)"
    regex_sample_frame_nums = os.path.join(dataroot, "frame([0-9]+).png")

    match_top_left = re.search(regex_top_left, patch_name)
    match_sample_frame_1 = re.search(regex_sample_frame_nums, frame1_path)

    sample = match_sample_frame_1.group(1)
    topmost = int(match_top_left.group(1))
    leftmost = int(match_top_left.group(2))

    frame1 = Image.open(frame1_path)
    frame2 = Image.open(frame2_path)
    frame3 = Image.open(frame3_path)
    frame4 = Image.open(frame4_path)
    frame5 = Image.open(frame5_path)

    frame1_patch = frame1.crop([leftmost, topmost, leftmost+patch_size, topmost+patch_size])
    frame2_patch = frame2.crop([leftmost, topmost, leftmost+patch_size, topmost+patch_size])
    frame3_patch = frame3.crop([leftmost, topmost, leftmost+patch_size, topmost+patch_size])
    frame4_patch = frame4.crop([leftmost, topmost, leftmost+patch_size, topmost+patch_size])
    frame5_patch = frame5.crop([leftmost, topmost, leftmost+patch_size, topmost+patch_size])

    name = sample + "_" + patch_name
    print(name)

    outpath = os.path.join(outroot, str(ib) + "ib", name)

    if not os.path.exists(outpath):
        os.makedirs(outpath)
            
    frame1_outpath = os.path.join(outpath, "frame1.png")
    frame2_outpath = os.path.join(outpath, "frame2.png")
    frame3_outpath = os.path.join(outpath, "frame3.png")
    frame4_outpath = os.path.join(outpath, "frame4.png")
    frame5_outpath = os.path.join(outpath, "frame5.png")

    frame1_patch.save(frame1_outpath)
    frame2_patch.save(frame2_outpath)
    frame3_patch.save(frame3_outpath)
    frame4_patch.save(frame4_outpath)
    frame5_patch.save(frame5_outpath)


def exr_to_flow(sample1, sample3, dataroot):
    vectorx_arr = []
    vectory_arr = []
    vectorz_arr = []
    vectorw_arr = []

    start = int(sample1)
    end = int(sample3)
    for i in range(start, end+1):
        exrpath = os.path.join(dataroot, "multilayer_{:04d}.exr".format(i))
        exrfile = pyexr.open(exrpath)
        vectorx_arr.append(exrfile.get("View Layer.Vector.X"))
        vectory_arr.append(exrfile.get("View Layer.Vector.Y"))
        vectorz_arr.append(exrfile.get("View Layer.Vector.Z"))
        vectorw_arr.append(exrfile.get("View Layer.Vector.W"))
    
    vectorx_arr = lumpy.asarray(vectorx_arr)
    vectory_arr = lumpy.asarray(vectory_arr)
    vectorz_arr = lumpy.asarray(vectorz_arr)
    vectorw_arr = lumpy.asarray(vectorw_arr)
    
    flo13_x = lumpy.sum(vectorz_arr[:-1], axis=0)
    flo13_y = lumpy.sum(vectorw_arr[:-1], axis=0)
    flo31_x = -1*lumpy.sum(vectorx_arr[1:], axis=0)
    flo31_y = -1*lumpy.sum(vectory_arr[1:], axis=0)

    #the following is for png flow
    # flo13 = lumpy.concatenate((flo13_x, flo13_y, lumpy.zeros(flo13_x.shape)), axis=2)
    # flo31 = lumpy.concatenate((flo31_x, flo31_y, lumpy.zeros(flo31_x.shape)), axis=2)
    # pdb.set_trace()
    # flo13 = lumpy.transpose(flo13, (2, 0, 1))
    # flo31 = lumpy.transpose(flo31, (2, 0, 1))

    # flo13_pil = Image.fromarray(flo13)
    # flo31_pil = Image.fromarray(flo31)

    # flo13_pil.save(flo13_outpath)
    # flo31_pil.save(flo31_outpath)

    flo13_np = lumpy.concatenate((flo13_x, flo13_y), axis=2)
    flo31_np = lumpy.concatenate((flo31_x, flo31_y), axis=2)

    return flo13_np, flo31_np


def save_example_exr(patch_name, frame1_path, frame2_path, frame3_path, \
                     ib):
    dataroot = "/fs/cfar-projects/anim_inb/datasets/TEST_BLENDER_3"
    outroot = "/fs/cfar-projects/anim_inb/datasets/TEST_BLENDER_3_large_patches"
    
    patch_size = 512

    regex_top_left = "([0-9]+)_([0-9]+)"
    # regex_sample_frame_nums = os.path.join(dataroot, "(.*)", "frame_([0-9]+).png")
    regex_sample_frame_nums = os.path.join(dataroot, "frame_([0-9]+).png")

    match_top_left = re.search(regex_top_left, patch_name)
    match_sample_frame_1 = re.search(regex_sample_frame_nums, frame1_path)
    match_sample_frame_2 = re.search(regex_sample_frame_nums, frame2_path)
    match_sample_frame_3 = re.search(regex_sample_frame_nums, frame3_path)
    # pdb.set_trace()

    sample1 = match_sample_frame_1.group(1)
    sample2 = match_sample_frame_2.group(1)
    sample3 = match_sample_frame_3.group(1)
    topmost = int(match_top_left.group(1))
    leftmost = int(match_top_left.group(2))

    frame1 = Image.open(frame1_path)
    frame2 = Image.open(frame2_path)
    frame3 = Image.open(frame3_path)

    frame1_patch = frame1.crop([leftmost, topmost, leftmost+patch_size, topmost+patch_size])
    frame2_patch = frame2.crop([leftmost, topmost, leftmost+patch_size, topmost+patch_size])
    frame3_patch = frame3.crop([leftmost, topmost, leftmost+patch_size, topmost+patch_size])

    name = sample1 + "_" + patch_name
    print(name)

    outpath = os.path.join(outroot, str(ib) + "ib", name)

    if not os.path.exists(outpath):
        os.makedirs(outpath)
            
    frame1_outpath = os.path.join(outpath, "frame1.png")
    frame2_outpath = os.path.join(outpath, "frame2.png")
    frame3_outpath = os.path.join(outpath, "frame3.png")

    frame1_patch.save(frame1_outpath)
    frame2_patch.save(frame2_outpath)
    frame3_patch.save(frame3_outpath)
    
    outflowroot = "/fs/cfar-projects/anim_inb/datasets/TEST_BLENDER_3_flows_2"
    if not os.path.exists(outflowroot):
        os.makedirs(outflowroot)
    # flo_subroot = os.path.join(outflowroot, str(ib)+"ib", sample1)
    # if not os.path.exists(flo_subroot):
    #     os.makedirs(flo_subroot)
    # flo_outpath = os.path.join(outflowroot, str(ib)+"ib", sample1, "flo.h5")
    flo_outpath = os.path.join(outflowroot, "flo.h5")

    # flo13_outpath = os.path.join(outflowroot, str(ib)+"ib", sample1, "flo13.png")
    # flo31_outpath = os.path.join(outflowroot, str(ib)+"ib", sample1, "flo31.png")

    # if os.path.exists(flo13_outpath) and os.path.exists(flo31_outpath):
    if os.path.exists(flo_outpath):
        #the following is for if flow is saved as png
        # flo13_pil = Image.open(flo13_outpath)
        # flo31_pil = Image.open(flo31_outpath)

        h5f = h5py.File(flo_outpath, "r")
        group_path_13 = os.path.join("/"+str(ib)+"ib", sample1, "flo13")
        group_path_31 = os.path.join("/"+str(ib)+"ib", sample1, "flo31")

        # flo13_np = h5f['flo13'][:]
        # flo31_np = h5f['flo31'][:]

        if group_path_13 in h5f.keys():
            flo13_np = h5f[group_path_13][:]
            flo31_np = h5f[group_path_31][:]
            keys_exist = True
        else:
            flo13_np, flo31_np = exr_to_flow(sample1, sample3, dataroot)
        h5f.close()

    else:
        flo13_np, flo31_np = exr_to_flow(sample1, sample3, dataroot)

        h5f = h5py.File(flo_outpath, "a")
        groupname = os.path.join("/"+str(ib)+"ib", sample1)
        h5f.require_group(groupname)
        h5f.create_dataset(os.path.join(groupname, 'flo13'), data=flo13_np)
        h5f.create_dataset(os.path.join(groupname, 'flo31'), data=flo31_np)
        h5f.close()

    flo13_patch = flo13_np[topmost:topmost+patch_size, leftmost:leftmost+patch_size, :]
    flo31_patch = flo31_np[topmost:topmost+patch_size, leftmost:leftmost+patch_size, :]

    h5f = h5py.File(flo_outpath, "a")
    groupname = os.path.join("/"+str(ib)+"ib", sample1)
    h5f.require_group(groupname)
    try:
        h5f.create_dataset(os.path.join(groupname, 'flo13_patch_'+patch_name), data=flo13_patch)
    except:
        pdb.set_trace()
    h5f.create_dataset(os.path.join(groupname, 'flo31_patch_'+patch_name), data=flo31_patch)
    h5f.close()

    # flo13_patch = flo13.crop([leftmost, topmost, leftmost+patch_size, topmost+patch_size])
    # flo31_patch = flo13.crop([leftmost, topmost, leftmost+patch_size, topmost+patch_size])

    # flo13_patch_outpath = os.path.join(outflowroot, str(ib)+"ib", name, "flo13.png")
    # flo31_patch_outpath = os.path.join(outflowroot, str(ib)+"ib", name, "flo31.png")

    # flo13_patch.save(flo13_patch_outpath)
    # flo31_patch.save(flo31_patch_outpath)


def main_blender():
    # dataroot = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_dog"
    dataroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_dog"
    # flowroot = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_tvl1_flows"
    flowroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_tvl1_flows"
    # csvroot = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_csv"
    csvroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_csv"

    patch_size = 512

    # csv_files = ["train_triplets_1ib.csv", "train_triplets_3ib.csv", "train_triplets_7ib.csv"]
    csv_files = ["train_triplets_1ib.csv", "train_triplets_3ib.csv"]
    # csv_files = ["train_triplets_7ib.csv"]
    ibs = [1, 3]
    # ibs = [7]

    for x, csv_file in enumerate(csv_files):
        curfile = os.path.join(csvroot, csv_file)
        with open(curfile, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

            for row in reader:
                frame1 = row[0]
                frame2 = row[1]
                frame3 = row[2]

                flow13_x = row[3]
                flow13_y = row[4]
                flow31_x = row[5]
                flow31_y = row[6]

                img1 = png_to_np(frame1)
                img3 = png_to_np(frame3)
                #note this means white bg is 255, black fg is 0

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

                        patch_1 = 1 - (patch_1 / 255)
                        patch_3 = 1 - (patch_3 / 255)

                        #all we care about is if the box is in it at all
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
                        save_example(patchname, frame1, frame2, frame3, \
                                     flow13_x, flow13_y, flow31_x, flow31_y, ibs[x])
                    else:
                        print("invalid :(")


def get_frame_difference(frame1, frame2):
    if lumpy.amax(frame1) > 1:
        frame1 = frame1 / 255
        frame2 = frame2 / 255
    difference = lumpy.square(frame1 - frame2)
    return difference


def main_SU():
    dataroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog/StevenHug_2048x1024"
    # flowroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_tvl1_flows"
    flowroot = "fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_360x640"
    csvroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_smol_csv"

    patch_size = 512

    # csv_files = ["train_triplets_1ib.csv", "train_triplets_3ib.csv", "train_triplets_7ib.csv"]
    csv_files = ["train_triplets_1ib.csv", "train_triplets_3ib.csv"]
    # csv_files = ["train_triplets_7ib.csv"]
    ibs = [1, 3]
    # ibs = [7]

    for x, csv_file in enumerate(csv_files):
        curfile = os.path.join(csvroot, csv_file)
        with open(curfile, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

            for row in reader:
                frame1 = row[0]
                frame2 = row[1]
                frame3 = row[2]

                flow = row[3]
                # flow13_x = row[3]
                # flow13_y = row[4]
                # flow31_x = row[5]
                # flow31_y = row[6]

                img1 = png_to_np(frame1)
                img3 = png_to_np(frame3)
                #note this means white bg is 255, black fg is 0

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

                        patch_1 = 1 - (patch_1 / 255)
                        patch_3 = 1 - (patch_3 / 255)

                        #get difference in patches
                        difference = get_frame_difference(patch_1, patch_3)
                        if lumpy.sum(difference) >= 3000:
                            if lumpy.sum(patch_1) >= 6000:
                                patches_1_names.append(name)
                            if lumpy.sum(patch_3) >= 6000:
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
                        save_example(patchname, frame1, frame2, frame3, ibs[x])
                    else:
                        print("invalid :(")


def main_SU_5():
    dataroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog/StevenHug_2048x1024"
    # flowroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_tvl1_flows"
    flowroot = "fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_360x640"
    csvroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_csv"

    patch_size = 512

    # csv_files = ["train_quintlets_1ib.csv", "train_quintlets_3ib.csv"]
    csv_files = ["train_quintlets_3ib.csv"]
    ibs = [1, 3]

    for x, csv_file in enumerate(csv_files):
        curfile = os.path.join(csvroot, csv_file)
        with open(curfile, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

            for row in reader:
                frame1 = row[0]
                frame2 = row[1]
                frame3 = row[2]
                frame4 = row[3]
                frame5 = row[4]

                flow = row[5]

                img1 = png_to_np(frame1)
                img5 = png_to_np(frame5)
                #note this means white bg is 255, black fg is 0

                leftmost_idx = 0
                rightmost_idx = img1.shape[1] - patch_size - 1
                topmost_idx = 0
                bottommost_idx = img1.shape[0] - patch_size - 1

                patches_1_names = []
                patches_5_names = []

                for i in range(topmost_idx, bottommost_idx, 64):
                    for j in range(leftmost_idx, rightmost_idx, 64):
                        name = str(i)+"_"+str(j)

                        patch_1 = img1[i:i+patch_size, j:j+patch_size]
                        patch_5 = img5[i:i+patch_size, j:j+patch_size]

                        patch_1 = 1 - (patch_1 / 255)
                        patch_5 = 1 - (patch_5 / 255)

                        #get difference in patches
                        difference = get_frame_difference(patch_1, patch_5)
                        if lumpy.sum(difference) >= 3000:
                            if lumpy.sum(patch_1) >= 6000:
                                patches_1_names.append(name)
                            if lumpy.sum(patch_5) >= 6000:
                                patches_5_names.append(name)

                used_names = []
                perm = lumpy.random.permutation(len(patches_1_names))
                for i in perm:
                    if len(used_names) >= 1:
                        break
                    patchname = patches_1_names[i]
                    if patchname in patches_5_names and patchname not in used_names:
                        print("valid")
                        #just want to make sure the image still exists in the third frame (didn't exit frame entirely lmao)
                        used_names.append(patchname)
                        save_example_5(patchname, frame1, frame2, frame3, \
                                     frame4, frame5, ibs[x])
                    else:
                        print("invalid :(")

                    
def main_blender_test():
    dataroot = "/fs/cfar-projects/anim_inb/datasets/TEST_BLENDER_3"

    patch_size = 512
    ibs = [1, 3]
    # ibs = [7]

    num_frames = 26

    for ib in ibs:
        for i in range(num_frames):
            if i < num_frames - ib - 1:
                frame1 = os.path.join(dataroot, "frame_{:04d}.png".format(i))
                frame2 = os.path.join(dataroot, "frame_{:04d}.png".format(i+(ib // 2)+1))
                frame3 = os.path.join(dataroot, "frame_{:04d}.png".format(i+ib+1))

                img1 = png_to_np(frame1)
                img3 = png_to_np(frame3)
                #note this means white bg is 255, black fg is 0

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

                        patch_1 = 1 - (patch_1 / 255)
                        patch_3 = 1 - (patch_3 / 255)

                        #all we care about is if the box is in it at all
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
                        save_example_exr(patchname, frame1, frame2, frame3, \
                                        ib)
                    else:
                        print("invalid :(")






if __name__ == "__main__":
    # main()
    # main_plot()
    # make_dataset()
    # make_large_dataset()

    # main_blender()
    # make_blender_large_dataset()
    # main_blender_test()

    main_SU()
    # main_SU_5()



