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
import h5py
import torch.nn.functional as F
import torchvision.transforms as transforms

import pdb


def _make_dataset(data_root, flow_root, train=True):
    dataPath = []
    flowPath = []
    framesFolder = []
    framesIndex = []
    
    if train:
        split = "train"
    else:
        split = "test"
    
    data_root_split = os.path.join(data_root, split)
    flow_root_split = os.path.join(flow_root, split)
    
    all_folders = os.listdir(data_root_split)

    count = 0
    for folder in all_folders:
        if count % 20 != 0:
            count += 1
            continue
        
        srcframe = os.path.join(data_root_split, folder, "frame_0.png")
        inbframe = os.path.join(data_root_split, folder, "frame_1.png")
        trgframe = os.path.join(data_root_split, folder, "frame_2.png")

        flow_filename = os.path.join(flow_root_split, folder+".npz")
        
        dataPath.append([srcframe, inbframe, trgframe])
        flowPath.append(flow_filename)
        framesFolder.append([folder, folder, folder])
        framesIndex.append(['0', '1', '2'])
        
        count += 1

    return dataPath, flowPath, framesFolder, framesIndex


def get_frame_range(random_reverse):
    reverse = 0
    if (random_reverse):
        ### Data Augmentation ###
        reverse = random.randint(0, 1)
        if reverse:
            frameRange = [2, 1, 0]
            inter = 1
        else:
            frameRange = [0, 1, 2]
            inter = 1
    else:
        frameRange = [0, 1, 2]
        inter = 1
    
    return frameRange, inter


def _flow_loader_npz(filename):
    flows = np.load(filename)
    flo13 = flows["flo13"]
    flo31 = flows["flo31"]
    flo12 = flows["flo12"]
    flo21 = flows["flo21"]
    flo23 = flows["flo23"]
    flo32 = flows["flo32"]
    return flo13, flo31, flo12, flo21, flo23, flo32


def _img_loader_png(path):
    img = Image.open(path)
    img_rgb = img.convert("RGB")

    return img, img_rgb


    
class AnimTripletWFlow(data.Dataset):
    def __init__(self, data_root, flow_root, train=True, img_size=(2048,1024), random_reverse=False):
        # Populate the list with image paths for all the
        # frame in `root`.
        dataPath, flowPath, \
            framesFolder, framesIndex = _make_dataset(data_root, flow_root, train=train)
        # Raise error if no images found in root.
        if len(dataPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + data_root + "\n"))
        
        self.img_size       = img_size
        self.dataPath       = dataPath
        self.framesFolder   = framesFolder
        self.framesIndex    = framesIndex
        self.flowPath       = flowPath
        self.random_reverse = random_reverse
        self.train          = train
        self.data_root      = data_root

        self.to_tensor      = transforms.PILToTensor()

    def __getitem__(self, index):
        sample = []
        flow = []
        masks = []

        folders = []
        indices = []
        
        frameRange, inter = get_frame_range(self.random_reverse)
    
        # Loop over for all frames corresponding to the `index`.
        for frameIndex in frameRange:
            image, image_rgb = _img_loader_png(self.dataPath[index][frameIndex])
            #seg networks should be normalized 0 to 1
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

        flow13, flow31, \
            flow12, flow21, \
            flow23, flow32 = _flow_loader_npz(self.flowPath[index])

        flow13 = torch.tensor(flow13)
        flow31 = torch.tensor(flow31)
        flow12 = torch.tensor(flow12)
        flow21 = torch.tensor(flow21)
        flow23 = torch.tensor(flow23)
        flow32 = torch.tensor(flow32)
        flow.append([flow13, flow31, flow12, flow21, flow23, flow32])
    
        return sample, masks, flow, folders, indices

    def __len__(self):
        return len(self.dataPath)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
