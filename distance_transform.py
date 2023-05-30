import os
import re
import numpy as lumpy
import cv2
from PIL import Image
from scipy.ndimage import distance_transform_edt

import pdb


def main():
    #first use these; later will suppress background
    regex = "2048x1024_t_3_k_3"
    roots = ["/fs/vulcan-projects/anim_inb_lilhuang/datasets/train_10k_preprocess_dog", \
            "/fs/vulcan-projects/anim_inb_lilhuang/datasets/test_2k_original_preprocess_dog"]

    tao = 15

    for root in roots:
        print(root)
        all_dirs = os.listdir(root)
        for dir_ in all_dirs:
            if re.search(regex, dir_) != None:
                print(dir_)
                frame1 = lumpy.asarray(Image.open(os.path.join(root, dir_, "frame1_smooth.jpg"))) / 255
                frame3 = lumpy.asarray(Image.open(os.path.join(root, dir_, "frame3_smooth.jpg"))) / 255

                transf_frame1 = (1 - lumpy.exp(-1*distance_transform_edt(frame1)/tao))*255
                transf_frame3 = (1 - lumpy.exp(-1*distance_transform_edt(frame3)/tao))*255

                frame1_final = Image.fromarray(transf_frame1).convert('L')
                frame3_final = Image.fromarray(transf_frame3).convert('L')

                frame1_final.save(os.path.join(root, dir_, "frame1_smooth_dt.jpg"))
                frame3_final.save(os.path.join(root, dir_, "frame3_smooth_dt.jpg"))
                frame1_final.save("aigoo.jpg")
                frame3_final.save("aigooo.jpg")



def main_blender():
    roots = ["/fs/vulcan-projects/anim_inb_lilhuang/datasets/Blender_cubes_dog"]

    total_num_frames_per_example = 242

    tao = 15

    for root in roots:
        print(root)
        all_dirs = os.listdir(root)
        for dir_ in all_dirs:
            print(dir_)
            for i in range(total_num_frames_per_example):
                curframe = lumpy.load(os.path.join(root, dir_, "frame_{:04d}.npy".format(i)))

                transf_curframe = (1 - lumpy.exp(-1*distance_transform_edt(curframe)/tao))*255

                curframe_final = Image.fromarray(transf_curframe).convert("L")

                curframe_final.save(os.path.join(root, dir_, "frame_{:04d}_dt.jpg".format(i)))
                curframe_final.save("agustd.jpg")


def main_SU():
    root = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_dog_patches_large"
    ibs = ["1ib", "3ib"]

    tao = 15

    # for ib in ibs:
    #     print(ib)
    #     all_dirs = os.listdir(os.path.join(root, ib))
    #     for dir_ in all_dirs:
    #         print(dir_)
    #         for i in range(1, 4):
    #             curframe = (cv2.imread(os.path.join(root, ib, dir_, "frame"+str(i)+".png"), 0) / 255).astype("uint8")

    #             transf_curframe = (1 - lumpy.exp(-1*distance_transform_edt(curframe)/tao))*255

    #             curframe_final = Image.fromarray(transf_curframe).convert("L")

    #             curframe_final.save(os.path.join(root, ib, dir_, "frame"+str(i)+"_dt.png"))
    #             # curframe_final.save("agustd.jpg")

    testroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog/StevenHug_2048x1024"

    all_frames = os.listdir(testroot)
    regex = ".png"
    for frame in all_frames:
        if re.search(regex, frame):
            print(frame)
            curframe = (cv2.imread(os.path.join(testroot, frame), 0) / 255).astype("uint8")
            transf_curframe = (1 - lumpy.exp(-1*distance_transform_edt(curframe)/tao))*255
            curframe_final = Image.fromarray(transf_curframe).convert("L")
            curframe_final.save(os.path.join(testroot, "dt_"+frame))



if __name__ == "__main__":
    # main()
    # main_blender()
    main_SU()



