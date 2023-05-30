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

def _make_dataset_exr(data_root, csv_root, flow_root, num_inbs, train=True, overfit=False):
    exrPath = []
    allPatches = []
    framesFolder = []
    framesIndex = []
    flowFilenames = []
    flowDatanames = []

    if train:
        split = "train"
    else:
        split = "test"

    csv_root_2 = os.path.join(csv_root, split, str(num_inbs)+"ib")
    flow_root_2 = os.path.join(flow_root, split, str(num_inbs)+"ib")

    #so far we can only do this for 1ib --> 3ib bc we don't have 5ib
    bigflow_ib = (num_inbs * 2) + 1
    big_flow_root = os.pat.join(flow_root, split, str(bigflow_ib)+ib)
    
    all_csvs = os.listdir(csv_root_2)
    regex = "(t_.*_s_.*_r_.*_[0-9])_exr.csv"
    regex_exr = os.path.join(data_root, "(t_.*_s_.*_r_.*_[0-9]_exr)", "multilayer_([0-9]+).exr")
    for csv_filename in all_csvs:
        if not re.search(regex, csv_filename):
            continue
        folder = re.search(regex, csv_filename).group(1)
        csv_file = os.path.join(csv_root_2, csv_filename)
        with open(csv_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                src_exr = row[0]
                inb_exr = row[1]
                trg_exr = row[2]

                src_num = re.search(regex_exr, src_exr).group(2)
                foldername = re.search(regex_exr, src_exr).group(1)
                flow_filename = os.path.join(flow_root_2, foldername, "flo.h5")
                
                if train:
                    for i in range(3, len(row)):
                        exrPath.append([src_exr, inb_exr, trg_exr])
                        allPatches.append(row[i])
                        framesFolder.append([folder, folder, folder])
                        framesIndex.append(['0', '1', '2'])
                        flowFilenames.append(flow_filename)
                        flowDatanames.append(src_num)
                else:
                    exrPath.append([src_exr, inb_exr, trg_exr])
                    framesFolder.append([folder, folder, folder])
                    framesIndex.append(['0', '1', '2'])
                    flowFilenames.append(flow_filename)
                    flowDatanames.append(src_num)

    return exrPath, allPatches, framesFolder, framesIndex, flowFilenames, flowDatanames


def _make_dataset(data_root, csv_root, flow_root, num_inbs, train=True, overfit=False):
    dataPath = []
    flowPath = []
    framesFolder = []
    framesIndex = []

    if overfit:
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
    
    if train:
        split = "train"
    else:
        split = "test"
    
    data_root_2 = os.path.join(data_root, split, str(num_inbs)+"ib")
    flow_root_2 = os.path.join(flow_root, split, str(num_inbs)+"ib")
    
    all_folders = os.listdir(data_root_2)
    regex = "(t_.*_s_.*_r_.*_[0-9])_exr"

    if train:
        regex_png = "example_([0-9]+)_([0-9]+)_([0-9]+)"
    else:
        regex_png = "example_([0-9]+)"

    count = 0
    for folder in all_folders:
        if not re.search(regex, folder):
            continue
        data_root_3 = os.path.join(data_root_2, folder)
        all_examples = os.listdir(data_root_3)
        for example in all_examples:
            if count % 20 != 0:
                count += 1
                continue
            if not re.search(regex_png, example):
                continue
            png_match = re.search(regex_png, example)
            ex_num = png_match.group(1)
            next_num = "{:04d}".format(int(ex_num)+1+num_inbs)

            if train:
                patchname_0 = png_match.group(2)
                patchname_1 = png_match.group(3)
            
            srcframe = os.path.join(data_root_3, example, "frame_0.png")
            inbframe = os.path.join(data_root_3, example, "frame_1.png")
            trgframe = os.path.join(data_root_3, example, "frame_2.png")

            if train:
                flow_filename = os.path.join(flow_root_2, folder, \
                                            "flows_"+ex_num+"_to_"+\
                                            next_num+"_"+patchname_0+\
                                            "_"+patchname_1+".npz")
                out_foldername = folder+"_"+ex_num+"_to_"+next_num+\
                                "_"+patchname_0+"_"+patchname_1
            else:
                flow_filename = os.path.join(flow_root_2, folder, \
                                            "flows_"+ex_num+"_to_"+next_num+".npz")
                out_foldername = folder+"_"+ex_num
            
            dataPath.append([srcframe, inbframe, trgframe])
            flowPath.append(flow_filename)
            framesFolder.append([out_foldername, out_foldername, out_foldername])
            framesIndex.append(['0', '1', '2'])
            
            count += 1

    return dataPath, flowPath, framesFolder, framesIndex


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
            frameRange = [2, 1, 0]
            inter = 1
        else:
            frameRange = [0, 1, 2]
            inter = 1
    else:
        frameRange = [0, 1, 2]
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


def _flow_loader_exr(path1, path3, dataroot, patchname, img_size, patch_size):
    vectorx_arr = []
    vectory_arr = []
    vectorz_arr = []
    vectorw_arr = []

    regex = os.path.join(dataroot, "(t_.*_s_.*_r_.*_[0-9]_exr)", "multilayer_([0-9]+).exr")
    foldername = re.search(regex, path1).group(1)
    start = int(re.search(regex, path1).group(2))
    end = int(re.search(regex, path3).group(2))

    for i in range(start, end+1):
        exrpath = os.path.join(dataroot, foldername, "multilayer_{:04d}.exr".format(i))
        exrfile = pyexr.open(exrpath)
        vectorx_arr.append(exrfile.get("View Layer.Vector.X"))
        vectory_arr.append(exrfile.get("View Layer.Vector.Y"))
        vectorz_arr.append(exrfile.get("View Layer.Vector.Z"))
        vectorw_arr.append(exrfile.get("View Layer.Vector.W"))
    
    vectorx_arr = np.asarray(vectorx_arr)
    vectory_arr = np.asarray(vectory_arr)
    vectorz_arr = np.asarray(vectorz_arr)
    vectorw_arr = np.asarray(vectorw_arr)
    
    flo13_x = np.sum(vectorz_arr[:-1], axis=0)
    flo13_y = np.sum(vectorw_arr[:-1], axis=0)
    flo31_x = -1*np.sum(vectorx_arr[1:], axis=0)
    flo31_y = -1*np.sum(vectory_arr[1:], axis=0)

    flo13_np = np.concatenate((flo13_x, flo13_y), axis=2)
    flo31_np = np.concatenate((flo31_x, flo31_y), axis=2)

    flo13_np_resized = cv2.resize(flo13_np, img_size)
    flo31_np_resized = cv2.resize(flo31_np, img_size)

    if patchname != None:
        patch_stuff = patchname.split("_")
        top = int(patch_stuff[0])
        left = int(patch_stuff[1])
        flo13_np_resized = flo13_np_resized[top:top+patch_size, left:left+patch_size]
        flo31_np_resized = flo31_np_resized[top:top+patch_size, left:left+patch_size]

    flo13_np_resized = np.transpose(flo13_np_resized, (2, 0, 1))
    flo31_np_resized = np.transpose(flo31_np_resized, (2, 0, 1))
    return flo13_np_resized, flo31_np_resized


def _flow_loader_h5(filename, dataname, patchname, img_size, patch_size):
    try:
        h5f = h5py.File(filename, "r")
    except:
        pdb.set_trace()

    flo13_np = h5f[dataname]["flo13"][:]
    flo31_np = h5f[dataname]["flo31"][:]
    h5f.close()

    flo13_np_resized = cv2.resize(flo13_np, img_size)
    flo31_np_resized = cv2.resize(flo31_np, img_size)

    if patchname != None:
        patch_stuff = patchname.split("_")
        top = int(patch_stuff[0])
        left = int(patch_stuff[1])
        flo13_np_resized = flo13_np_resized[top:top+patch_size, left:left+patch_size]
        flo31_np_resized = flo31_np_resized[top:top+patch_size, left:left+patch_size]

    flo13_np_resized = np.transpose(flo13_np_resized, (2, 0, 1))
    flo31_np_resized = np.transpose(flo31_np_resized, (2, 0, 1))

    return flo13_np_resized, flo31_np_resized


def _flow_loader_npz(filename):
    flows = np.load(filename)
    flo13 = flows["flo13"]
    flo31 = flows["flo31"]
    flo12 = flows["flo12"]
    flo21 = flows["flo21"]
    flo23 = flows["flo23"]
    flo32 = flows["flo32"]
    return flo13, flo31, flo12, flo21, flo23, flo32


def _img_loader_exr(exrfile, patchname, img_size, patch_size):
    frame_R = exrfile.get("Composite.Combined.R")
    frame_G = exrfile.get("Composite.Combined.G")
    frame_B = exrfile.get("Composite.Combined.B")

    frame = np.concatenate((frame_R, frame_G, frame_B), axis=2)
    frame_resized = cv2.resize(frame, img_size)

    if patchname != None:
        patch_stuff = patchname.split("_")
        top = int(patch_stuff[0])
        left = int(patch_stuff[1])
        frame_resized = frame_resized[top:top+patch_size, left:left+patch_size]
    mask = 1 - cv2.cvtColor(frame_resized.astype("uint8"), cv2.COLOR_BGR2GRAY)
    frame_resized = np.transpose(frame_resized, (2, 0, 1))
    mask = np.where(mask > 0.5, 1, 0)
    mask = np.expand_dims(mask, axis=0)

    return frame_resized*255, mask


def _img_loader_png(path):
    img = Image.open(path)
    # img = img.resize((256, 256))
    # img = img.crop((0, 256, 256, 512))
    img_rgb = img.convert("RGB")

    return img, img_rgb


    
class CSVEXRTriplet(data.Dataset):
    def __init__(self, csv_root, data_root, flow_root, num_ib_frames, train=True, \
                 transform=None, img_size=(2048,1024), patch_size=512,\
                 discrim_crop_size=None, random_flip=False, \
                 random_reverse=False, dt=False, patch_location=None, exr=False, \
                 overfit=False):
        # Populate the list with image paths for all the
        # frame in `root`.
        if exr:
            exrPath, allPatches, \
                framesFolder, framesIndex, \
                flowFilenames, flowDatanames = _make_dataset_exr(data_root, csv_root, \
                                                            flow_root, num_ib_frames, \
                                                            train=train, overfit=overfit)
            dataPath = None
            flowPath = None
            # Raise error if no images found in root.
            if len(exrPath) == 0:
                raise(RuntimeError("Found 0 files in subfolders of: " + data_root + "\n"))
        else:
            dataPath, flowPath, \
                framesFolder, framesIndex = _make_dataset(data_root, csv_root, flow_root, \
                                                          num_ib_frames, train=train, \
                                                          overfit=overfit)
            exrPath = None
            allPatches = None
            flowFilenames = None
            flowDatanames = None
            # Raise error if no images found in root.
            if len(dataPath) == 0:
                raise(RuntimeError("Found 0 files in subfolders of: " + data_root + "\n"))
        
        self.num_ib_frames  = num_ib_frames
        self.img_size       = img_size
        self.exrPath        = exrPath
        self.dataPath       = dataPath
        self.framesFolder   = framesFolder
        self.framesIndex    = framesIndex
        self.allPatches     = allPatches
        self.flowPath       = flowPath
        self.flowFilenames  = flowFilenames
        self.flowDatanames  = flowDatanames
        self.d_crop_size    = discrim_crop_size
        self.dt             = dt
        self.transform      = transform
        self.random_flip    = random_flip
        self.random_reverse = random_reverse
        self.patch_location = patch_location
        self.patch_size     = patch_size
        self.train          = train
        self.exr            = exr
        self.csv_root       = csv_root
        self.data_root      = data_root

        self.to_tensor      = transforms.PILToTensor()

    def __getitem__(self, index):
        sample = []
        flow = []
        masks = []

        folders = []
        indices = []
        
        discrim_crop = _get_discrim_crop(self.patch_location, self.img_size, self.d_crop_size)
        frameRange, inter = get_frame_range(self.random_reverse)

        if self.train and self.exr:
            patchname = self.allPatches[index]
        else:
            patchname = None
    
        # Loop over for all frames corresponding to the `index`.
        for frameIndex in frameRange:
            if self.exr:
                exr_filename = self.exrPath[index][frameIndex]
                exrfile = pyexr.open(exr_filename)

                frame, mask = _img_loader_exr(exrfile, patchname, \
                                        self.img_size, self.patch_size)
                frame = torch.tensor(frame)
            else:
                image, image_rgb = _img_loader_png(self.dataPath[index][frameIndex])
                #seg networks should be normalized 0 to 1 (or...-1 to 1?)
                mask = (1 - (np.array(image) / 255)).astype("uint8")
                mask = np.expand_dims(mask, axis=0)
                #is image a mask or grayscale still?
                frame = self.to_tensor(image_rgb)
            mask = torch.tensor(mask)
            sample.append(frame)
            masks.append(mask)

            folder = self.framesFolder[index][frameIndex]
            iindex = self.framesIndex[index][frameIndex]
            folders.append(folder)
            indices.append(iindex)

        if self.exr:
            flow13, flow31 = _flow_loader_h5(self.flowFilenames[index], \
                                         self.flowDatanames[index], \
                                         patchname, self.img_size, \
                                         self.patch_size)
        else:
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

        if self.random_flip:
            sample, masks, flow = random_flip(sample, masks, flow)
    
        return sample, masks, flow, folders, indices

    def __len__(self):
        if self.exr:
            return len(self.exrPath)
        else:
            return len(self.dataPath)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
