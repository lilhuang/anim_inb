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

import pdb

def _make_dataset(data_root, root_flow, root_hist_mask, dt, raft, smooth, \
                    im_size, dataset_filepath):
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
        print("test shit was saved, unpickling now!")
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
        print("test shit was not saved :(")
        framesPath = []
        flowPath = []
        framesFolder = []
        framesIndex = []
        maskPath = []
        correct_regex = "Disney_(.*)"+str(im_size[0])+"x"+str(im_size[1])+"_t_3_k_3$"

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


def _img_loader(path, resizeDim=None):
    img = np.expand_dims(cv2.imread(path, 0), 0)
    if resizeDim != None:
        resized_img = cv2.resize(img, dsize=resizeDim, interpolation=cv2.INTER_CUBIC)
    else:
        resized_img = img

    return resized_img


def _flow_loader(path, resizeSize=None):
    flow_np = np.load(path)
    flow = torch.from_numpy(flow_np)
    # print(flow.size())
    # pdb.set_trace()

    if resizeSize is not None:
        flow = F.interpolate(flow[None], size=(resizeDim[1],resizeDim[0]))[0]
        factor0 = float(resizeDim[0]) / flow_np.shape[2]
        factor1 = float(resizeDim[1]) / flow_np.shape[1]

        flow[0, :, :] *= factor0
        flow[1, :, :] *= factor1

    return flow.clone().detach()


def _flow_loader_jpg(pathx, pathy, resizeDim=None):
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

    return flow.clone().detach()


    
class AniTripletWithSGMFlowTest(data.Dataset):
    def __init__(self, data_root, root_flow, root_hist_mask, dataset_filepath, \
                 transform=None, transform_rgb=None, dt=False, raft=False, smooth=False,\
                 img_size=(2048,1024), resize=(2048,1024)):
        # Populate the list with image paths for all the
        # frame in `root`.
        framesPath, flowPath, \
            framesFolder, framesIndex, maskPath = _make_dataset(data_root, root_flow, \
                                                 root_hist_mask, dt, raft, smooth, img_size, \
                                                 dataset_filepath)
        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))
                
        self.root           = data_root
        self.transform      = transform
        self.transform_rgb  = transform_rgb
        self.img_size       = img_size
        self.resize         = resize
        self.framesPath     = framesPath
        self.flowPath       = flowPath
        self.framesFolder   = framesFolder
        self.framesIndex    = framesIndex
        self.maskPath       = maskPath
        self.raft           = raft
        self.dt             = dt
        self.smooth         = smooth

    def __getitem__(self, index):
        sample = []
        flow = []
        masks = []

        folders = []
        indices = []
        
        frameRange = [0, 1, 2]
        if self.raft:
            flowRange = [0, 1]
        else:
            flowRange = [0, 2]
        inter = 1
        
        # Loop over for all frames corresponding to the `index`.
        for frameIndex in frameRange:
            # Open image using pil and augment the image.
            image = torch.tensor(_img_loader(self.framesPath[index][frameIndex], resizeDim=self.resize)).float()
            # image.save(str(frameIndex) + '.jpg')

            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)

            mask_np = np.load(self.maskPath[index][frameIndex])
            if np.max(mask_np) > 1:
                mask_np = mask_np / 255
                mask_np = mask_np.astype("uint8")
            mask_pos_fg = 1 - mask_np

            if self.resize != None and self.img_size != self.resize:
                mask_pos_fg = cv2.resize(mask_pos_fg, dsize=img_size, interpolation=cv2.INTER_CUBIC)

            masks.append(mask_pos_fg)

            folder = self.framesFolder[index][frameIndex]
            iindex = self.framesIndex[index][frameIndex]
            folders.append(folder)
            indices.append(iindex)

        for flowIndex in flowRange:
            if self.raft:
                flowa = _flow_loader(self.flowPath[index][flowIndex], resizeSize=self.resize)
            else:
                flowa = _flow_loader_jpg(self.flowPath[index][flowIndex], self.flowPath[index][flowIndex+1], \
                                        resizeDim=self.resize)
            flow.append(flowa)

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
    
