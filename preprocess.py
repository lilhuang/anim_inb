import os
import re
import numpy as lumpy
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision as tv
import kornia

import pdb


def save_mask_to_img(mask, name):
    # pdb.set_trace()
    image_np = 255*lumpy.ones((mask.shape[0], mask.shape[1], 3))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 0:
                image_np[i][j] = [0,0,0]
    image = Image.fromarray(image_np.astype("uint8"))
    image.save(name)


# input: (bs,rgb,h,w) or (bs,1,h,w)
# returns: (bs,1,h,w)
def batch_dog(img, t=1.0, sigma=1.0, k=1.6, epsilon=0.01, kernel_factor=4, clip=True):
    # to grayscale if needed
    if len(img.shape) == 3:
        img = torch.unsqueeze(img, 0)
    bs,ch,h,w = img.shape
    if ch==3:
        img = kornia.color.rgb_to_grayscale(img)
    else:
        assert ch==1

    # calculate dog
    kern0 = max(2*int(sigma*kernel_factor)+1, 3)
    kern1 = max(2*int(sigma*k*kernel_factor)+1, 3)
    g0 = kornia.filters.gaussian_blur2d(
        img, (kern0,kern0), (sigma,sigma), border_type='replicate',
    )
    g1 = kornia.filters.gaussian_blur2d(
        img, (kern1,kern1), (sigma*k,sigma*k), border_type='replicate',
    )
    ans = 0.5 + t*(g1-g0) - epsilon
    ans = ans.clip(0,1) if clip else ans
    return ans



def main_atd12k():
    # root = "/vulcanscratch/lilhuang/Blender/datasets"
    # data_root = os.path.join(root, "data")
    # preprocess_root = os.path.join(root, "preprocess_dog")

    root = "/fs/vulcan-projects/anim_inb_lilhuang/datasets"
    # data_root = os.path.join(root, "train_10k")
    # preprocess_root = os.path.join(root, "train_10k_preprocess_dog")

    data_root = os.path.join(root, "test_2k_original")
    preprocess_root = os.path.join(root, "test_2k_original_preprocess_dog")

    all_triplet_dirs = os.listdir(data_root)
    # all_triplet_dirs = ["Disney_v4_0_000716_s1"]

    # resize_size = (240, 136)
    # resize_size = (256, 128)
    # resize_sizes = [(2048, 1024), (1024, 512), (512, 256), (256, 128)]
    # resize_sizes = [(256, 128)]
    resize_sizes = [(2048, 1024)]

    # count = 0
    for i, dir in enumerate(all_triplet_dirs):
        # if count > 0:
        #     break
        print(dir)
        full_dir = os.path.join(data_root, dir)
        images = os.listdir(full_dir)
        truth = True
        for i, im in enumerate(images):
            name_regex = "(.*).jpg"
            frame_num = re.search(name_regex, im).group(1)

            cur_img_path = os.path.join(full_dir, im)
            cur_img = Image.open(cur_img_path).convert("RGB")

            for resize_size in resize_sizes:
                target_dir = os.path.join(preprocess_root, dir+"_"+str(resize_size[0])+"x"+str(resize_size[1])+"_t_3_k_3")
                # target_dir = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/ATD_TEST_PREPROCESS_3_k_2-5"
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

                target_jpg = os.path.join(target_dir, frame_num+".jpg")
                target_npy = os.path.join(target_dir, frame_num+".npy")
                
                if not os.path.exists(target_jpg) or not os.path.exists(target_npy) or truth:
                    timg = tv.transforms.functional.to_tensor(cur_img)[None,]
                    out = batch_dog(timg, t=3, k=3)
                    result = tv.transforms.functional.to_pil_image((out[0]<0.5).float())
                    # result.save("CHECK.jpg")
                    result_resized = result.resize(resize_size)
                    result_resized.save(target_jpg)
                    # result_resized.save("CHECK_RESIZED.jpg")

                    result_np = (1/255)*lumpy.array(result_resized)
                    result_np_1 = result_np.astype(int)
                    result_np = lumpy.where(result_np > 0.5, 1, 0)
                    result_np = result_np.astype(int)
                    # pdb.set_trace()
                    # save_mask_to_img(result_np_1, "first_ex.jpg")
                    # save_mask_to_img(result_np, "second_ex.jpg")
                    # count += 1
                    lumpy.save(target_npy, result_np)


def main_nexus_SU():
    root = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/"
    data_root = os.path.join(root, "SU_24fps", "StevenHug")
    preprocess_root = os.path.join(root, "preprocess_dog")

    all_frames = os.listdir(data_root)

    resize_sizes = [(2048, 1024)]

    for i, im in enumerate(all_frames):
        print(im)
        name_regex = "(.*).png"
        # pdb.set_trace()
        frame_num = re.search(name_regex, im).group(1)

        cur_img_path = os.path.join(data_root, im)
        cur_img = Image.open(cur_img_path).convert("L")

        for resize_size in resize_sizes:
            cur_img_resized = cur_img.resize(resize_size)
            target_dir = os.path.join(preprocess_root, "StevenHug_"+str(resize_size[0])+"x"+str(resize_size[1]))
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            target_png = os.path.join(target_dir, frame_num+".png")
            
            if not os.path.exists(target_png):
                timg = tv.transforms.functional.to_tensor(cur_img)[None,]
                out = batch_dog(timg, t=2.0, k=1.5)
                result = tv.transforms.functional.to_pil_image((out[0]<0.5).float())
                result_resized = result.resize(resize_size)
                result_resized.save(target_png)


def main_nexus_Blender():
    root = "/fs/cfar-projects/anim_inb/datasets/"
    # data_root = os.path.join(root, "Blender_cubes_raw")
    # preprocess_root = os.path.join(root, "Blender_cubes_dog")
    data_root = os.path.join(root, "Suzanne_raw")
    preprocess_root = os.path.join(root, "Suzanne_dog")


    all_dirs = os.listdir(data_root)
    # all_dirs = ['Disney_v4_0_000479_s2']

    resize_sizes = [(2048, 1024)]

    for i, dir_ in enumerate(all_dirs):
        regex = "_png$"
        # # regex2 = "_jpg$"
        # # if not (re.search(regex, dir) or re.search(regex2, dir)):
        if not re.search(regex, dir_):
            continue
        print(dir_)
        full_dir = os.path.join(data_root, dir_)
        # smol_im_dir = os.path.join(data_root, dir+"_smol_2")

        # images = os.listdir(smol_im_dir)
        images = os.listdir(full_dir)
        # pdb.set_trace()
        for i, im in enumerate(images):
            # if re.search(regex, full_dir):
            #     name_regex = "(.*).png"
            # else:
            #     name_regex = "(.*).jpg"
            name_regex = "(.*).png"
            # pdb.set_trace()
            frame_num = re.search(name_regex, im).group(1)

            cur_img_path = os.path.join(full_dir, im)
            cur_img = Image.open(cur_img_path).convert("L")

            for resize_size in resize_sizes:
                cur_img_resized = cur_img.resize(resize_size)
                target_dir = os.path.join(preprocess_root, dir_+"_"+str(resize_size[0])+"x"+str(resize_size[1]))
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

                # cur_img_resized.save(os.path.join(target_dir, frame_num+"_ORIGINAL.jpg"))

                target_png = os.path.join(target_dir, frame_num+".png")
                # pdb.set_trace()
                # target_png = "minyoongi.png"
                # target_npy = os.path.join(target_dir, frame_num+".npy")
                
                if not os.path.exists(target_png):
                    # print(target_png)
                    timg = tv.transforms.functional.to_tensor(cur_img_resized)[None,]
                    out = batch_dog(timg, t=3.0, k=3.0)
                    result = tv.transforms.functional.to_pil_image((out[0]<0.5).float())
                    result.save(target_png)


                    # result_np = (1/255)*lumpy.array(result_resized)
                    # result_np = lumpy.where(result_np >= 0.5, 1, 0)
                    # result_np = result_np.astype(int)
                    # # pdb.set_trace()
                    # lumpy.save(target_npy, result_np)


def main_nexus_pencil_tests():
    root = "/fs/cfar-projects/anim_inb/datasets/vid_pngs/"
    other_root = "/fs/cfar-projects/anim_inb/datasets/vid_pngs_dog"

    resize_size = (2048, 1024)

    all_dirs = os.listdir(root)
    name_regex = "(.*).png"
    for dir_ in all_dirs:
        print(dir_)
        full_root = os.path.join(root, dir_)
        out_root = os.path.join(other_root, dir_)
        if not os.path.exists(out_root):
            os.makedirs(out_root)
        all_frames = os.listdir(full_root)
        for i, im in enumerate(all_frames):
            if im != ".DS_Store":
                frame_name = re.search(name_regex, im).group(1)
                cur_img_path = os.path.join(full_root, im)
                cur_img = Image.open(cur_img_path).convert("L")
                cur_img_resized = cur_img.resize(resize_size)
                target_png = os.path.join(out_root, frame_name+".png")
                if not os.path.exists(target_png):
                    timg = tv.transforms.functional.to_tensor(cur_img)[None,]
                    out = batch_dog(timg, t=1.8, k=1.3)
                    result = tv.transforms.functional.to_pil_image((out[0]<0.5).float())
                    result_resized = result.resize(resize_size)
                    result_resized.save(target_png)
    

    


if __name__ == "__main__":
    # main_atd12k()
    # main()
    # main_nexus_Blender()
    # main_nexus_SU()
    main_nexus_pencil_tests()



