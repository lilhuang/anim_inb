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
import torch.nn.functional as F

import pdb

def _make_csv_dataset(data_root, root_flow, csv_root, root_hist_mask, dt, raft, dataset_filepath):    
    framesPath_file = dataset_filepath+"_framesPath.pkl"
    flowPath_file = dataset_filepath+"_raft_flowPath.pkl"
    framesFolder_file = dataset_filepath+"_framesFolder.pkl"
    framesIndex_file = dataset_filepath+"_framesIndex.pkl"

    if os.path.exists(framesPath_file) and os.path.exists(flowPath_file) \
            and os.path.exists(framesFolder_file) and os.path.exists(framesIndex_file):
        print("test shit was saved, unpickling now! taetae :D boxy smile!")
        with open(framesPath_file, 'rb') as f:
            framesPath = pickle.load(f)
        with open(flowPath_file, 'rb') as f:
            flowPath = pickle.load(f)
        with open(framesFolder_file, 'rb') as f:
            framesFolder = pickle.load(f)
        with open(framesIndex_file, 'rb') as f:
            framesIndex = pickle.load(f)
    else:
        print("train shit was not saved :( yoongi ._____. face")
        framesPath = []
        flowPath = []
        framesFolder = []
        framesIndex = []
        maskPath = []

        csv_filename = "train_triplets_" + str(num_ib_frames) + "ib.csv"
        csv_file = os.path.join(csv_root, csv_filename)
        with open(csv_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                src_name = row[0]
                inb_name = row[1]
                trg_name = row[2]
                framesPath.append([src_name, inb_name, trg_name])

                #flo13, flo31, flo13_dt, flo31_dt
                flowPath.append([row[3], row[4], row[5], row[6]])

                regex = os.path.join(dataroot, "([.*]+)", "flow_([.*]+).npy")
                match = re.search(regex, row[3])
                folder = match.group(1)+"_"+match.group(2)
                pdb.set_trace()

                framesFolder.append([folder, folder, folder])
                framesIndex.append(['0', '1', '2'])
        
        with open(framesPath_file, 'wb') as f:
            pickle.dump(framesPath, f)
        with open(flowPath_file, 'wb') as f:
            pickle.dump(flowPath, f)
        with open(framesFolder_file, 'wb') as f:
            pickle.dump(framesFolder, f)
        with open(framesIndex_file, 'wb') as f:
            pickle.dump(framesIndex, f)

    return framesPath, flowPath, framesFolder, framesIndex


def _flow_loader(path, cropArea=None, resizeDim=None):
    flow_np = np.load(path)
    flow = torch.from_numpy(flow_np)
    # print(flow.size())
    # pdb.set_trace()

    if resizeDim is not None:
        flow = F.interpolate(flow[None], size=(resizeDim[1],resizeDim[0]))[0]

        factor0 = float(resizeDim[0]) / flow_np.shape[2]
        factor1 = float(resizeDim[1]) / flow_np.shape[1]

        flow[0, :, :] *= factor0
        flow[1, :, :] *= factor1
    
    if discrim_crop is not None:
        flow = flow[:, discrim_crop[1]:discrim_crop[3], discrim_crop[0]:discrim_crop[2]]

    return flow.clone().detach()


def _flow_loader_jpg(pathx, pathy, resizeDim=None, discrim_crop=None):
    flow_x_jpg = Image.open(pathx)
    flow_y_jpg = Image.open(pathy)

    flow_x_arr = np.asarray(flow_x_jpg)
    flow_y_arr = np.asarray(flow_y_jpg)

    flow_np = np.stack([flow_x_arr, flow_y_arr])

    B = 100.0
    flow_np = ((flow_np * (2 * B)) / 255) - B

    flow = torch.from_numpy(flow_np) 
    # print(flow.size())
    flow = F.interpolate(flow[None], size=(resizeDim[1],resizeDim[0]))[0]

    if resizeDim is not None:
        factor0 = float(resizeDim[0]) / flow_np.shape[2]
        factor1 = float(resizeDim[1]) / flow_np.shape[1]

        flow[0, :, :] *= factor0
        flow[1, :, :] *= factor1
    
    if discrim_crop != None:
        flow = flow[discrim_crop[1]:discrim_crop[3], discrim_crop[0]:discrim_crop[2]]

    return flow.clone().detach()


    
class CSVAniTripletWithSGMFlow(data.Dataset):
    def __init__(self, data_root, csv_root, num_ib_frames, root_flow, \
                 dataset_filepath, root_hist_mask, transform=None, img_size=(2048,1024), \
                 discrim_crop_size=None, highmag_flow=False, random_flip=False, \
                 random_reverse=False, dt=False, raft=False, patch_location=None):
        # Populate the list with image paths for all the
        # frame in `root`.
        framesPath, flowPath, \
            framesFolder, framesIndex = _make_csv_dataset(data_root, root_flow, \
                                                 csv_root, root_hist_mask, dataset_filepath)
        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))
                
        self.root           = data_root
        self.transform      = transform
        self.img_size       = img_size
        self.framesPath     = framesPath
        self.flowPath       = flowPath
        self.framesFolder   = framesFolder
        self.framesIndex    = framesIndex
        self.maskPath       = maskPath
        self.d_crop_size    = discrim_crop_size
        self.highmag_flow   = highmag_flow
        self.dt             = dt
        self.raft           = raft
        self.random_flip    = random_flip
        self.random_reverse = random_reverse
        self.patch_location = patch_location

    def __getitem__(self, index):
        sample = []
        flow = []
        cropArea = []

        folders = []
        indices = []
        
        if self.d_crop_size != None:
            if self.patch_location == "random":
                top = np.random.randint(0, self.img_size[1]-self.d_crop_size)
                bottom = top+self.d_crop_size
                left = np.random.randint(0, self.img_size[0]-self.d_crop_size)
                right = left+self.d_crop_size
            elif self.patch_location == "center":
                top = (self.img_size[1]-self.d_crop_size) // 2
                bottom = top+self.d_crop_size
                left = (self.img_size[0]-self.d_crop_size) // 2
                right = left+self.d_crop_size
            elif self.patch_location == "centerish":
                if self.d_crop_size <= 256:
                    topmost = self.d_crop_size
                    bottommost_of_top = self.img_size[1]-2*self.d_crop_size
                    leftmost = 2 * self.d_crop_size
                    rightmost_of_left = self.img_size[0]-3*self.d_crop_size

                    top = np.random.randint(topmost, bottommost_of_top)
                    bottom = top+self.d_crop_size
                else: #if it's 512
                    top = self.d_crop_size // 2
                    bottom = top+self.d_crop_size
                    leftmost = self.d_crop_size
                    rightmost_of_left = self.img_size[0]-2*self.d_crop_size

                left = np.random.randint(leftmost, rightmost_of_left)
                right = left+self.d_crop_size

            discrim_crop = (left, top, right, bottom)
        else:
            discrim_crop = None

        reverse = 0
        if (self.random_reverse):
            ### Data Augmentation ###
            reverse = random.randint(0, 1)
            if reverse:
                frameRange = [2, 1, 0]
                if self.raft:
                    flowRange = [1, 0]
                else:
                    flowRange = [2, 0]
                inter = 1
            else:
                frameRange = [0, 1, 2]
                if self.raft:
                    flowRange = [0, 1]
                else:
                    flowRange = [0, 2]
                inter = 1
        else:
            frameRange = [0, 1, 2]
            if self.raft:
                flowRange = [0, 1]
            else:
                flowRange = [0, 2]
            inter = 1
    
        # Loop over for all frames corresponding to the `index`.
        for frameIndex in frameRange:
            mask_np = np.load(self.maskPath[index][frameIndex])
            if np.max(mask_np) > 1:
                mask_np = mask_np / 255
                mask_np = mask_np.astype("uint8")
            mask_pos_fg = 1 - mask_np

            if discrim_crop != None:
                mask_pos_fg = mask_pos_fg[discrim_crop[1]:discrim_crop[3], discrim_crop[0]:discrim_crop[2]]

            sample.append(mask_pos_fg)

            folder = self.framesFolder[index][frameIndex]
            iindex = self.framesIndex[index][frameIndex]
            folders.append(folder)
            indices.append(iindex)

        for flowIndex in flowRange:
            if self.raft:
                flowa = _flow_loader(self.flowPath[index][flowIndex], discrim_crop=discrim_crop)
            else:
                flowa = _flow_loader_jpg(self.flowPath[index][flowIndex], self.flowPath[index][flowIndex+1], discrim_crop=discrim_crop)
            flow.append(flowa)

        if self.random_flip:
            newsample = []
            newmasks = []
            newflow = []
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

        return sample, flow, folders, indices

    def __len__(self):
        return len(self.framesPath)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
