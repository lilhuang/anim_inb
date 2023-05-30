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
import pickle
import torch.nn.functional as F
import torchvision.transforms as transforms

import pdb

def _make_dataset(data_root, root_flow, root_hist_mask, dt, raft, smooth, img_size, \
                    dataset_filepath, discrim_crop_size=None, overfit=False):
    if overfit:
        folder = "Disney_0_05215_s2_256x128_t_3_k_3"
        framesFolder = [[folder, folder, folder]]
        framesIndex = [['0', '1', '2']]

        full_path = os.path.join(data_root, folder)
        all_imgs = os.listdir(full_path)
        all_imgs.sort()

        src_name = os.path.join(full_path, "frame1.jpg")
        inb_name = os.path.join(full_path, "frame2.jpg")
        trg_name = os.path.join(full_path, "frame3.jpg")
        framesPath = [[src_name, inb_name, trg_name]]

        mask_src = os.path.join(full_path, "frame1.npy")
        mask_inb = os.path.join(full_path, "frame2.npy")
        mask_trg = os.path.join(full_path, "frame3.npy")
        maskPath = [[mask_src, mask_inb, mask_trg]]

        full_flow_dir = os.path.join(root_flow, folder)
        all_flows = os.listdir(full_flow_dir)
        all_flows.sort()

        flow_x13 = os.path.join(full_flow_dir, all_flows[0])
        flow_x31 = os.path.join(full_flow_dir, all_flows[1])
        flow_y13 = os.path.join(full_flow_dir, all_flows[2])
        flow_y31 = os.path.join(full_flow_dir, all_flows[3])
        flowPath = [[flow_x13, flow_y13, flow_x31, flow_y31]]

    else:
        if smooth:
            framesPath_file = dataset_filepath+"_smooth_framesPath.pkl"
            if dt:
                flowPath_file = dataset_filepath+"_dt_raft_smooth_flowPath.pkl"
            elif raft:
                flowPath_file = dataset_filepath+"_raft_flowPath.pkl"
            else:
                flowPath_file = dataset_filepath+"_smooth_flowPath.pkl"
            maskPath_file = dataset_filepath+"_smooth_maskPath.pkl"
        else:
            framesPath_file = dataset_filepath+"_framesPath.pkl"
            if dt:
                flowPath_file = dataset_filepath+"_dt_raft_flowPath.pkl"
            elif raft:
                flowPath_file = dataset_filepath+"_raft_flowPath.pkl"
            else:
                flowPath_file = dataset_filepath+"_flowPath.pkl"
            maskPath_file = dataset_filepath+"_maskPath.pkl"
        framesFolder_file = dataset_filepath+"_framesFolder.pkl"
        framesIndex_file = dataset_filepath+"_framesIndex.pkl"

        if os.path.exists(framesPath_file) and os.path.exists(flowPath_file) \
                and os.path.exists(framesFolder_file) and os.path.exists(framesIndex_file) \
                and os.path.exists(maskPath_file):
            print("shit was saved, unpickling now!")
            with open(framesPath_file, 'rb') as f:
                framesPath = pickle.load(f)
            with open(flowPath_file, 'rb') as f:
                flowPath = pickle.load(f)
            with open(framesFolder_file, 'rb') as f:
                framesFolder = pickle.load(f)
            with open(framesIndex_file, 'rb') as f:
                framesIndex = pickle.load(f)
            with open(maskPath_file, 'rb') as f:
                maskPath = pickle.load(f)
        
        else:
            print("shit was not saved :(")
            framesPath = []
            flowPath = []
            framesFolder = []
            framesIndex = []
            maskPath = []
            correct_regex = "Disney_(.*)"+str(img_size[0])+"x"+str(img_size[1])+"_t_3_k_3"

            sample_dirs = os.listdir(data_root)
            for dir in sample_dirs:
                if not re.search(correct_regex, dir):
                    continue
                full_path = os.path.join(data_root, dir)
                all_imgs = os.listdir(full_path)

                full_flow_dir = os.path.join(root_flow, dir)
                all_flows = os.listdir(full_flow_dir)

                if smooth:
                    src_name = os.path.join(full_path, "frame1_smooth.jpg")
                    inb_name = os.path.join(full_path, "frame2_smooth.jpg")
                    trg_name = os.path.join(full_path, "frame3_smooth.jpg")
                    framesPath.append([src_name, inb_name, trg_name])

                    mask_src = os.path.join(full_path, "frame1_smooth.npy")
                    mask_inb = os.path.join(full_path, "frame2_smooth.npy")
                    mask_trg = os.path.join(full_path, "frame3_smooth.npy")
                    maskPath.append([mask_src, mask_inb, mask_trg])
                    
                    if raft:
                        if dt:
                            flow_13 = os.path.join(full_flow_dir, "flo_smooth_dt_13.npy")
                            flow_31 = os.path.join(full_flow_dir, "flo_smooth_dt_31.npy")
                        else:
                            flow_13 = os.path.join(full_flow_dir, "flo_smooth_13.npy")
                            flow_31 = os.path.join(full_flow_dir, "flo_smooth_31.npy")
                        flowPath.append([flow_13, flow_31])
                    else:
                        all_flows.sort()
                        flow_x13 = os.path.join(full_flow_dir, all_flows[0])
                        flow_x31 = os.path.join(full_flow_dir, all_flows[1])
                        flow_y13 = os.path.join(full_flow_dir, all_flows[2])
                        flow_y31 = os.path.join(full_flow_dir, all_flows[3])
                        flowPath.append([flow_x13, flow_y13, flow_x31, flow_y31])

                else:
                    src_name = os.path.join(full_path, "frame1.jpg")
                    inb_name = os.path.join(full_path, "frame2.jpg")
                    trg_name = os.path.join(full_path, "frame3.jpg")
                    framesPath.append([src_name, inb_name, trg_name])

                    mask_src = os.path.join(full_path, "frame1.npy")
                    mask_inb = os.path.join(full_path, "frame2.npy")
                    mask_trg = os.path.join(full_path, "frame3.npy")
                    maskPath.append([mask_src, mask_inb, mask_trg])

                    if raft:
                        if dt:
                            flow_13 = os.path.join(full_flow_dir, "flo_dt_13.npy")
                            flow_31 = os.path.join(full_flow_dir, "flo_dt_31.npy")
                        else:
                            flow_13 = os.path.join(full_flow_dir, "flo_13.npy")
                            flow_31 = os.path.join(full_flow_dir, "flo_31.npy")
                        flowPath.append([flow_13, flow_31])
                    else:
                        all_flows.sort()
                        flow_x13 = os.path.join(full_flow_dir, all_flows[0])
                        flow_x31 = os.path.join(full_flow_dir, all_flows[1])
                        flow_y13 = os.path.join(full_flow_dir, all_flows[2])
                        flow_y31 = os.path.join(full_flow_dir, all_flows[3])
                        flowPath.append([flow_x13, flow_y13, flow_x31, flow_y31])

                framesFolder.append([dir, dir, dir])
                framesIndex.append(['0', '1', '2'])
            
            with open(framesPath_file, 'wb') as f:
                pickle.dump(framesPath, f)
            with open(flowPath_file, 'wb') as f:
                pickle.dump(flowPath, f)
            with open(framesFolder_file, 'wb') as f:
                pickle.dump(framesFolder, f)
            with open(framesIndex_file, 'wb') as f:
                pickle.dump(framesIndex, f)
            with open(maskPath_file, 'wb') as f:
                pickle.dump(maskPath, f)

    return framesPath, flowPath, framesFolder, framesIndex, maskPath


def save_mask_to_img(mask, name):
    image_np = 255*np.ones((mask.shape[0], mask.shape[1], 3))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 0:
                image_np[i][j] = [0,0,0]
    image = Image.fromarray(image_np.astype("uint8"))
    image.save(name)


def _img_loader(path, discrim_crop=None):
    img = np.expand_dims(cv2.imread(path, 0), 0)

    if discrim_crop != None:
        cropped_img = img[:, discrim_crop[1]:discrim_crop[3], discrim_crop[0]:discrim_crop[2]]
    else:
        cropped_img = img

    return cropped_img


def _flow_loader(path, resizeDim=None, discrim_crop=None):
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

    if resizeDim is not None:
        flow = F.interpolate(flow[None], size=(resizeDim[1],resizeDim[0]))[0]

        factor0 = float(resizeDim[0]) / flow_np.shape[2]
        factor1 = float(resizeDim[1]) / flow_np.shape[1]

        flow[0, :, :] *= factor0
        flow[1, :, :] *= factor1

    if discrim_crop != None:
        flow = flow[:, discrim_crop[1]:discrim_crop[3], discrim_crop[0]:discrim_crop[2]]
    
    return flow.clone().detach()


    
class AniTripletWithSGMFlow(data.Dataset):
    def __init__(self, data_root, \
                 root_flow, root_hist_mask, dataset_filepath, transform=None, transform_rgb=None, \
                 dt=False, raft=False, smooth=False, img_size=(2048,1024), discrim_crop_size=None, \
                 patch_location=None, random_flip=False, random_reverse=False, overfit=False, \
                 highmag_flow=False):
        # Populate the list with image paths for all there
        # frame in `root`.
        framesPath, flowPath, \
            framesFolder, framesIndex, maskPath = _make_dataset(data_root, root_flow, \
                                                root_hist_mask, dt, raft, smooth, img_size,\
                                                dataset_filepath, \
                                                discrim_crop_size=discrim_crop_size, overfit=overfit)

        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))
                
        self.root           = data_root
        self.transform      = transform
        self.transform_rgb  = transform_rgb
        self.framesPath     = framesPath
        self.flowPath       = flowPath
        self.framesFolder   = framesFolder
        self.framesIndex    = framesIndex
        self.maskPath       = maskPath
        self.d_crop_size    = discrim_crop_size
        self.patch_location = patch_location
        self.random_flip    = random_flip
        self.random_reverse = random_reverse
        self.overfit        = overfit
        self.img_size       = img_size
        self.dt             = dt
        self.raft           = raft
        self.smooth         = smooth
        self.highmag_flow   = highmag_flow

    def __getitem__(self, index):
        sample = []
        flow = []
        masks = []

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
            # Open image using pil and augment the image.
            if not self.highmag_flow:
                image = torch.tensor(_img_loader(self.framesPath[index][frameIndex], discrim_crop=discrim_crop)).float()
                image_np = image.cpu().numpy()

                if self.transform is not None:
                    image = self.transform(image)
                sample.append(image)

            mask_np = np.load(self.maskPath[index][frameIndex])
            if np.max(mask_np) > 1:
                mask_np = mask_np / 255
                mask_np = mask_np.astype("uint8")
            mask_pos_fg = 1 - mask_np

            if discrim_crop != None:
                mask_pos_fg = mask_pos_fg[discrim_crop[1]:discrim_crop[3], discrim_crop[0]:discrim_crop[2]]

            masks.append(mask_pos_fg)

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

        return sample, flow, folders, indices, masks

    def __len__(self):
        return len(self.framesPath)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
