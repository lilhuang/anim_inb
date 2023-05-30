import os
import shutil
import glob

import pdb


def main():
    # root = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_raft_flows"
    root = "/fs/cfar-projects/anim_inb/datasets/test_2k_preprocess_raft_flows"

    all_dirs = os.listdir(root)

    for dir_ in all_dirs:
        fulldir = os.path.join(root, dir_)
        filelist = glob.glob(os.path.join(root, dir_, "*.jpg"))

        for file in filelist:
            try:
                os.remove(file)
                print("removed", file)
            except:
                print("cannot remove", file)


def main_others():
    roots = ["/fs/cfar-projects/anim_inb/datasets/test_2k_original_preprocess_dog",
            "/fs/cfar-projects/anim_inb/datasets/train_10k_preprocess_dog"]
    
    for root in roots:
        all_dirs = os.listdir(root)
        dirlist = glob.glob(os.path.join(root, "*_256x128*"))

        for dir_ in dirlist:
            try:
                print("removing", dir_)
                shutil.rmtree(dir_)
            except:
                print("cannot remove", dir_)


if __name__ == "__main__":
    main()
    # main_others()
