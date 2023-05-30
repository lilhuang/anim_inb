############ HI YIANNI THIS LINE IS IMPORTANT #############
import torch.utils.data as data
#########################################################
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
import pickle
import torch.nn.functional as F
import torchvision.transforms as transforms

import pdb


def _make_csv_dataset(data_root, csv_root, num_ib_frames, data_source, csv_filename_in=None, \
                      flow_type="pips", flow_root=None):
    framesPath = []
    framesFolder = []
    framesIndex = []
    flowPath = []
    
    csv_filename = "train"+csv_filename_in
    csv_file = os.path.join(csv_root, csv_filename)
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            src_name = row[0]
            inb_name = row[1]
            trg_name = row[2]
            framesPath.append([src_name, inb_name, trg_name])
            
            regex = os.path.join(data_root, "(.*)", "frame_([0-9]+).png")
            
            match = re.search(regex, src_name)
            match2 = re.search(regex, trg_name)
            src_num = match.group(2)
            trg_num = match2.group(2)
            folder = match.group(1)+"_"+src_num+"_to_"+trg_num

            framesFolder.append([folder, folder, folder])
            framesIndex.append(['0', '1', '2'])
            if flow_type == "pips":
                flow04 = row[3]
                flow02 = row[4]
                flow24 = row[5]
                flowPath.append([flow04, flow02, flow24])

        return framesPath, framesFolder, framesIndex, flowPath


def _flow_loader_npz(filename):
    flows = np.load(filename)
    flo13 = flows["flo13"]
    flo31 = flows["flo31"]
    return flo13, flo31


def _img_loader(path, img_size=None):
    img = Image.open(path)
    if img_size != None:
        img = img.resize(img_size)
    img_rgb = img.convert("RGB")

    return img, img_rgb

    
class SampleDataloader(data.Dataset):
    def __init__(self, args, data_root, ibs, data_source, csv_filename=None, \
                 img_size=(2048,1024), flow_type="pips", flow_root=None, csv_root=None, \
                 csv=False):
        ########### HI YIANNI OVER HERE A SMOL README FOR YOU
        # Populate the list with image paths for all images in the dataset
        # note that in my situation, each data sample contains 3 images and optical
        # flow. This next function call basically takes in a csv file which has already
        # written out the paths of sample image triplets and their corresponding flows 
        # and populates lists with all the paths corresponding to a single sample.
        # Later, these lists are referenced at training/testing time with an index
        # (see the __getitem__ function below).

        # btw, the csv thing is something i did for my situation/dataset, but you 
        # can populate a list how you like.
        # essentially what you want to end up with is a list of either images or paths
        # or arrays or whatever data format you're using. That way, __getitem__ 
        # can reference this list at runtime.
        ######################################################

        framesPath, \
            framesFolder, \
            framesIndex, \
            flowPath = _make_csv_dataset(data_root, csv_root, ibs, data_source, \
                                         csv_filename_in=csv_filename, flow_type=flow_type, flow_root=flow_root)

        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))
                
        self.root           = data_root
        self.framesPath     = framesPath
        self.framesFolder   = framesFolder
        self.framesIndex    = framesIndex
        self.flowPath       = flowPath
        self.random_reverse = random_reverse
        self.overfit        = overfit
        self.img_size       = img_size
        self.flow_type      = flow_type
        self.ibs            = ibs


    def __getitem__(self, index):
        sample = []
        rgb_sample = []
        masks = []

        folders = []
        indices = []

        flow = []

        to_tensor = transforms.PILToTensor()

        frameRange = [0, 1, 2]
        if self.ibs == 7:
            flowRange = [0, 2]
        else:
            flowRange = [0, 1]

    
        # Loop over for all frames corresponding to the `index`.
        for frameIndex in frameRange:
            # Open image using pil
            image, image_rgb = _img_loader(self.framesPath[index][frameIndex], self.img_size)
            image = to_tensor(image).float()
            image_rgb = to_tensor(image_rgb).float()

            sample.append(image)
            rgb_sample.append(image_rgb)

            mask_np = np.asarray(image.cpu())
            mask_np = mask_np / 255
            mask_np = mask_np.astype("uint8")
            mask_pos_fg = 1 - mask_np

            masks.append(mask_pos_fg)

            folder = self.framesFolder[index][frameIndex]
            iindex = self.framesIndex[index][frameIndex]
            folders.append(folder)
            indices.append(iindex)

        if self.flow_type == "pips":
            flo15, flo51 = _flow_loader_npz(self.flowPath[index][0])
            flo13, flo31 = _flow_loader_npz(self.flowPath[index][1])
            flo35, flo53 = _flow_loader_npz(self.flowPath[index][2])
            flow.append([flo15, flo51, flo13, flo31, flo35, flo53])

        return sample, rgb_sample, folders, indices, masks, flow




    def __len__(self):
        return len(self.framesPath)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
