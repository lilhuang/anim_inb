import numpy as lumpy
import os
import re
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image

import pdb


def make_dataset(data_root, root_hist_mask):
    framesPath = []
    maskPath = []
    framesFolder = []

    sample_dirs = os.listdir(data_root)
    for dir in sample_dirs:
        correct_regex = "256x128_t_3_k_3$"
        if not re.search(correct_regex, dir):
            continue
        full_path = os.path.join(data_root, dir)
        all_imgs = os.listdir(full_path)
        all_imgs.sort()

        for i, file in enumerate(all_imgs):
            full_filename = os.path.join(full_path, file)
            if i % 2 == 0:   
                framesPath.append(full_filename)
            else:
                maskPath.append(full_filename)
        
        framesFolder += [dir, dir, dir]
    
    return framesPath, maskPath, framesFolder


def filename_to_pil(filename, resize_size=(256, 128)):
    image = Image.open(filename).convert("L")
    resized_img = image.resize(resize_size, Image.ANTIALIAS)
    return resized_img



class MaskToImageLoader(data.Dataset):
    def __init__(self, data_root, root_hist_mask, resizeSize=(256, 128)):
        self.data_root = data_root
        self.root_hist_mask = root_hist_mask
        self.resize_size = resizeSize

        self.framesPath, \
            self.maskPath, \
            self.framesFolder = make_dataset(self.data_root, self.root_hist_mask)

    
    def __getitem__(self, index):
        image_filename = self.framesPath[index]
        image_sample = filename_to_pil(image_filename, resize_size=self.resize_size)
        image_sample = transforms.ToTensor()(image_sample)

        mask_filename = self.maskPath[index]
        mask = lumpy.load(mask_filename)

        folder_name = self.framesFolder[index]

        return image_sample, mask, folder_name
    

    def __len__(self):
        return len(self.framesPath)
    

    def __repr__(self):
        #tbh no idea what this is but ok?
        return 0



