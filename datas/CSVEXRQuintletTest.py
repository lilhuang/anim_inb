# dataloader for multi frames (acceleration), modified from superslomo

import torch.utils.data as data
from PIL import Image
import os
import re
import os.path
import random
import sys
import numpy as np
import torch
import cv2
import csv
import pyexr
import h5py
import torch.nn.functional as F
import torchvision.transforms as transforms

import pdb


def _make_dataset(data_root, num_inbs, overfit=False):
    dataPath = []
    framesFolder = []
    framesIndex = []

    if overfit:
        # fix this later if necessary
        # folder = "t_bezier_s_facing_bezier_r_none_0_exr"
        folder = "t_bezier_s_facing_bezier_r_none_6_exr"
        # out_foldername = folder + "_0019"
        out_foldername = folder + "_0118_448_640"
        # data_root_3 = os.path.join(data_root, "test", "1ib", folder, "example_0019")
        # flow_file = os.path.join(flow_root, "test", "1ib", folder, "flows_0019.npz")
        data_root_3 = os.path.join(data_root, "train", "1ib", folder, "example_0118_448_640")
        flow_file = os.path.join(flow_root, "train", "1ib", folder, "flows_0118_448_640.npz")

        srcframe = os.path.join(data_root_3, "frame_0.png")
        inbframe = os.path.join(data_root_3, "frame_1.png")
        trgframe = os.path.join(data_root_3, "frame_2.png")

        dataPath.append([srcframe, inbframe, trgframe])
        flowPath.append(flow_file)
        framesFolder.append([out_foldername, out_foldername, out_foldername])
        framesIndex.append(['0', '1', '2'])
        return dataPath, flowPath, framesFolder, framesIndex
    
    split = "test"
    
    data_root_2 = os.path.join(data_root, split, str(num_inbs)+"ib")
    
    all_folders = os.listdir(data_root_2)
    regex = "(t_.*_s_.*_r_.*_[0-9])_exr"

    regex_png = "example_([0-9]+)"

    count = 0
    for folder in all_folders:
        if not re.search(regex, folder):
            continue
        data_root_3 = os.path.join(data_root_2, folder)
        all_examples = os.listdir(data_root_3)
        all_examples.sort()
        for i, example in enumerate(all_examples):
            if count % 5 != 0:
                count += 1
                continue
            if not re.search(regex_png, example):
                continue
            if i >= len(all_examples) - 2:
                break
            png_match = re.search(regex_png, example)
            ex_num = png_match.group(1)
            ex_num_int = int(ex_num)
            ex_num_next = "{:04d}".format(ex_num_int+1)
            ex_num_next2 = "{:04d}".format(ex_num_int+2)

            next_example = "example_"+ex_num_next
            next2_example = "example_"+ex_num_next2
            
            srcframe = os.path.join(data_root_3, example, "frame_0.png")
            inbframe = os.path.join(data_root_3, example, "frame_1.png")
            trgframe = os.path.join(data_root_3, example, "frame_2.png")
            inb2frame = os.path.join(data_root_3, next_example, "frame_2.png")
            trg2frame = os.path.join(data_root_3, next2_example, "frame_2.png")

            out_foldername = folder+"_"+ex_num
            
            dataPath.append([srcframe, inbframe, trgframe, inb2frame, trg2frame])
            framesFolder.append([out_foldername, out_foldername, out_foldername, out_foldername, out_foldername])
            framesIndex.append(['0', '1', '2', '3', '4'])
            
            count += 1

    return dataPath, framesFolder, framesIndex


def _get_discrim_crop(patch_location, image_size, d_crop_size):
    if d_crop_size != None:
        if patch_location == "random":
            top = np.random.randint(0, img_size[1]-d_crop_size)
            bottom = top+d_crop_size
            left = np.random.randint(0, img_size[0]-d_crop_size)
            right = left+d_crop_size
        elif patch_location == "center":
            top = (img_size[1]-d_crop_size) // 2
            bottom = top+d_crop_size
            left = (img_size[0]-d_crop_size) // 2
            right = left+d_crop_size
        elif patch_location == "centerish":
            if d_crop_size <= 256:
                topmost = d_crop_size
                bottommost_of_top = img_size[1]-2*d_crop_size
                leftmost = 2 * d_crop_size
                rightmost_of_left = img_size[0]-3*d_crop_size

                top = np.random.randint(topmost, bottommost_of_top)
                bottom = top+d_crop_size
            else: #if it's 512
                top = d_crop_size // 2
                bottom = top+d_crop_size
                leftmost = d_crop_size
                rightmost_of_left = img_size[0]-2*d_crop_size

            left = np.random.randint(leftmost, rightmost_of_left)
            right = left+d_crop_size

        discrim_crop = (left, top, right, bottom)
    else:
        discrim_crop = None
    
    return discrim_crop


def get_frame_range(random_reverse):
    reverse = 0
    if (random_reverse):
        ### Data Augmentation ###
        reverse = random.randint(0, 1)
        if reverse:
            frameRange = [4, 3, 2, 1, 0]
            inter = 1
        else:
            frameRange = [0, 1, 2, 3, 4]
            inter = 1
    else:
        frameRange = [0, 1, 2, 3, 4]
        inter = 1
    
    return frameRange, inter


def random_flip(sample, masks, flow):
    flip = random.randint(0, 1)
    if flip:
        which_flip = random.randint(0, 1)
        if which_flip:
            #horiz flip
            for i, image in enumerate(sample):
                newimage = transforms.functional.hflip(image)
                sample[i] = newimage
            for i, mask in enumerate(masks):
                newmask = np.flip(mask, axis=1).copy()
                masks[i] = newmask
            for i, ind_flow in enumerate(flow):
                newflow = transforms.functional.hflip(ind_flow)
                flow[i] = newflow
        else:
            #vert flip
            for i, image in enumerate(sample):
                newimage = transforms.functional.vflip(image)
                sample[i] = newimage
            for i, mask in enumerate(masks):
                newmask = np.flip(mask, axis=0).copy()
                masks[i] = newmask
            for i, ind_flow in enumerate(flow):
                newflow = transforms.functional.vflip(ind_flow)
                flow[i] = newflow
    return sample, masks, flow


def _flow_loader_npz(filename):
    flows = np.load(filename)
    flo13 = flows["flo13"]
    flo31 = flows["flo31"]
    return flo13, flo31


def _img_loader_png(path):
    img = Image.open(path)
    # img = img.resize((256, 256))
    # img = img.crop((0, 256, 256, 512))
    img_rgb = img.convert("RGB")

    return img, img_rgb


    
class CSVEXRQuintletTest(data.Dataset):
    def __init__(self, data_root, num_ib_frames, img_size=(2048,1024), patch_size=512,\
                 discrim_crop_size=None, random_flip=False, random_reverse=False, \
                 dt=False, patch_location=None, overfit=False):
        # Populate the list with image paths for all the
        # frame in `root`.
        dataPath, \
            framesFolder, \
            framesIndex = _make_dataset(data_root, num_ib_frames, overfit=overfit)
        # Raise error if no images found in root.
        if len(dataPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + data_root + "\n"))
        
        self.num_ib_frames  = num_ib_frames
        self.img_size       = img_size
        self.dataPath       = dataPath
        self.framesFolder   = framesFolder
        self.framesIndex    = framesIndex
        self.d_crop_size    = discrim_crop_size
        self.dt             = dt
        self.random_flip    = random_flip
        self.random_reverse = random_reverse
        self.patch_location = patch_location
        self.patch_size     = patch_size
        self.data_root      = data_root

        self.to_tensor      = transforms.PILToTensor()

    def __getitem__(self, index):
        sample = []
        masks = []

        folders = []
        indices = []
        
        discrim_crop = _get_discrim_crop(self.patch_location, self.img_size, self.d_crop_size)
        frameRange, inter = get_frame_range(self.random_reverse)
    
        # Loop over for all frames corresponding to the `index`.
        for frameIndex in frameRange:
            image, image_rgb = _img_loader_png(self.dataPath[index][frameIndex])
            #seg networks should be normalized 0 to 1 (or...-1 to 1?)
            mask = (1 - (np.array(image) / 255)).astype("uint8")
            mask = np.expand_dims(mask, axis=0)
            frame = self.to_tensor(image_rgb)
            mask = torch.tensor(mask)
            sample.append(frame)
            masks.append(mask)

            folder = self.framesFolder[index][frameIndex]
            iindex = self.framesIndex[index][frameIndex]
            folders.append(folder)
            indices.append(iindex)

        if self.random_flip:
            sample, masks, flow = random_flip(sample, masks, flow)
    
        return sample, masks, folders, indices

    def __len__(self):
        return len(self.dataPath)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
