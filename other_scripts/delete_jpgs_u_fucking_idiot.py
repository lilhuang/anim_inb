import os
import glob
import re
import argparse
import numpy as lumpy
from PIL import Image

import pdb


def check_for_npy():
    # root = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_dog"
    root = "/fs/cfar-projects/anim_inb/datasets/train_10k_preprocess_dog/"
    # root = "/fs/cfar-projects/anim_inb/datasets/test_2k_original_preprocess_dog/"
    alldirs = os.listdir(root)
    regex = "(.*).npy"
    has_npy = False
    npy_files = []
    for dir_ in alldirs:
        files = os.listdir(os.path.join(root, dir_))
        for file in files:
            if re.search(regex, file):
                has_npy = True
                npy_files.append(os.path.join(dir_, file))
    
    pdb.set_trace()



def npy_to_png_atd12k():
    parser = argparse.ArgumentParser(description='directory root name')
    parser.add_argument('--root', type=str, help='directory root name')

    args = parser.parse_args()

    if args.root == "train":
        root = "/fs/cfar-projects/anim_inb/datasets/train_10k_preprocess_dog"
    else: #if root == "test"
        root = "/fs/cfar-projects/anim_inb/datasets/test_2k_original_preprocess_dog"

    alldirs = os.listdir(root)
    regex = "(.*).npy"
    for dir_ in alldirs:
        filelist = os.listdir(os.path.join(root, dir_))
        # pdb.set_trace()
        for file in filelist:
            name = re.search(regex, file)
            if not name == None:
                name = re.search(regex, file).group(1)
                arr = lumpy.load(os.path.join(root, dir_, file))
                pil = Image.fromarray(arr, mode="L")
                pil.save(os.path.join(root, dir_, name+".png"))
                os.remove(os.path.join(root, dir_, file))
                # pdb.set_trace()



def npy_to_png():
    parser = argparse.ArgumentParser(description='directory root name')
    parser.add_argument('--name', type=str, help='directory root name')

    args = parser.parse_args()


    root = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_dog"
    for i in range(10):
        dir_ = os.path.join(root, args.name+str(i)+"_png_2048x1024")
        filelist = os.listdir(os.path.join(root, dir_))
        for file in filelist:
            regex = "(.*).npy"
            name = re.search(regex, file)
            if not name == None:
                name = name.group(1)
                # pdb.set_trace()
                arr = lumpy.load(os.path.join(root, dir_, file))
                pil = Image.fromarray(arr, mode="L")
                pil.save(os.path.join(root, dir_, name+".png"))
                os.remove(os.path.join(root, dir_, file))
                # print(os.path.join(root, dir_, name+".png"))
                # pdb.set_trace()



def npy_to_png_flow():
    parser = argparse.ArgumentParser(description='directory root name')
    parser.add_argument('--name', type=str, help='directory root name')

    args = parser.parse_args()

    root = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_raft_flows"
    regex = "(.*).npz"

    if args.name == "t_straight_s_random_r_none_":
        val = 4
    else:
        val = 10

    for i in range(val):
        dir_ = os.path.join(root, args.name+str(i)+"_png_2048x1024")        
        filelist = os.listdir(os.path.join(root, dir_))
        for file in filelist:
            name = re.search(regex, file)
            if not name == None:
                name = name.group(1)
                try:
                    arr = lumpy.load(os.path.join(root, dir_, file))['arr_0']
                except:
                    pdb.set_trace()
                print(os.path.join(root, dir_, name))
                pilx = Image.fromarray(arr[0], mode="L")
                pily = Image.fromarray(arr[1], mode="L")
                pilx.save(os.path.join(root, dir_, name+"_x.png"))
                pily.save(os.path.join(root, dir_, name+"_y.png"))
                os.remove(os.path.join(root, dir_, file))


def main_delete_npzs():
    root = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_raft_flows"

    all_dirs = os.listdir(root)
    for dir_ in all_dirs:
        filelist = glob.glob(os.path.join(root, dir_, "*.npz"))
        for file in filelist:
            try:
                os.remove(file)
                print("taetae loves u")
            except:
                print("error in removing", file)



def main():
    root = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_dog"

    all_dirs = os.listdir(root)
    for dir_ in all_dirs:
        filelist = glob.glob(os.path.join(root, dir_, "*.jpg"))

        for file in filelist:
            try:
                os.remove(file)
                print("taetae is proud of u")
            except:
                print("error in removing", file)



if __name__ == "__main__":
    # main()
    # npy_to_png()
    # check_for_npy()
    # npy_to_png_atd12k()
    # npy_to_png_flow()
    main_delete_npzs()


