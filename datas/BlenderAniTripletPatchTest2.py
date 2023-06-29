# dataloader for multi frames (acceleration), modified from superslomo

import torch.utils.data as data
from PIL import Image
import os
import os.path
import random
import sys
import re
import csv
import numpy as np
import torch
import cv2
import pickle
import torch.nn.functional as F
from utils.config import Config
import torchvision.transforms as transforms

import pdb

def _make_csv_dataset(data_root, csv_root, num_ib_frames, data_source, csv_filename_in=None, \
                      flow_type="tvl1", flow_root=None, small_dataset=False, dt=False):    
    print("test shit was not saved :( yoongi ._____. face")
    framesPath = []
    framesFolder = []
    framesIndex = []
    flowPath = []

    regex_data_source = "pt"
    if data_source == "all" or data_source == "moving_gif":
        csv_filename = "test_triplets.csv"
    elif re.search(regex_data_source, data_source):
        csv_filename = "test"+csv_filename_in
    else:
        # csv_filename = "test_triplets_2_" + str(num_ib_frames) + "ib.csv"
        csv_filename = "test_triplets_" + str(num_ib_frames) + "ib.csv"
    csv_file = os.path.join(csv_root, csv_filename)
    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        i = 0
        for row in reader:
            if (not small_dataset) or (small_dataset and i % 10 == 0):
                src_name = row[0]
                inb_name = row[1]
                trg_name = row[2]

                # if data_source == "SU":
                #     regex = os.path.join(data_root, "(.*)", "frame([0-9]+).png")
                # else:
                #     regex = os.path.join(data_root, "(.*)", "frame_([0-9]+).png")

                if data_source == "moving_gif":
                    this_data_root = "/fs/cfar-projects/anim_inb/new_datasets/moving_gif_png_dog/test/"
                    regex = os.path.join(this_data_root, "(.*)", "frame_([0-9]+).png")
                elif re.search("SU_24fps", src_name):
                    this_data_root = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog"
                    regex = os.path.join(this_data_root, "(.*)", "frame([0-9]+).png")
                else:
                    this_data_root = "/fs/cfar-projects/anim_inb/datasets/vid_pngs_dog"
                    regex = os.path.join(this_data_root, "(.*)", "frame_([0-9]+).png")
                
                match = re.search(regex, src_name)
                match2 = re.search(regex, trg_name)
                src_num = match.group(2)
                trg_num = match2.group(2)
                if data_source == "all" or data_source == "SU" or re.search(regex_data_source, data_source):
                    folder = match.group(1)+"_"+src_num+"_to_"+trg_num
                else:
                    folder = match.group(1)

                if flow_type == "tvl1":
                    if num_ib_frames == 7:
                        flow13_x = row[3]
                        flow13_y = row[4]
                        flow31_x = row[5]
                        flow31_y = row[6]
                        flowPath.append([flow13_x, flow13_y, flow31_x, flow31_y])
                    else:
                        if data_source == "SU":
                            if not dt:
                                flow13 = os.path.join(flow_root, str(num_ib_frames)+"ib", "flo_"+src_num+"_to_"+trg_num+".png")
                                flow31 = os.path.join(flow_root, str(num_ib_frames)+"ib", "flo_"+trg_num+"_to_"+src_num+".png")
                            else:
                                flow13 = os.path.join(flow_root, str(num_ib_frames)+"ib", "flo_dt_"+src_num+"_to_"+trg_num+".png")
                                flow31 = os.path.join(flow_root, str(num_ib_frames)+"ib", "flo_dt_"+trg_num+"_to_"+src_num+".png")
                        else:
                            flow13 = os.path.join(flow_root, str(num_ib_frames)+"ib", folder, "flo_"+src_num+"_to_"+trg_num+".png")
                            flow31 = os.path.join(flow_root, str(num_ib_frames)+"ib", folder, "flo_"+trg_num+"_to_"+src_num+".png")
                        flowPath.append([flow13, flow31])
                elif flow_type == "pips":
                    flow04 = row[3]
                    flow02 = row[4]
                    flow24 = row[5]
                    if os.path.exists(flow04) and os.path.exists(flow02) and os.path.exists(flow24):
                        framesPath.append([src_name, inb_name, trg_name])
                        framesFolder.append([folder, folder, folder])
                        framesIndex.append(['0', '1', '2'])
                        flowPath.append([flow04, flow02, flow24])
            i += 1

    if flow_type != None:
        return framesPath, framesFolder, framesIndex, flowPath
    else:
        return framesPath, framesFolder, framesIndex, []


def _flow_loader_png(path, resizeDim=None, discrim_crop=None):
    flow = cv2.imread(path)
    flow = np.transpose(flow, (2, 0, 1))
    flow = flow[:2] #third channel was just zeros to let it save as png
    #add in resizeDim and discrim_crop if they are ever relevant again lol
    return torch.from_numpy(flow).float().clone().detach()


def _flow_loader_jpg(pathx, pathy, resizeDim=None, discrim_crop=None):
    flow_x_jpg = Image.open(pathx)
    flow_y_jpg = Image.open(pathy)

    flow_x_arr = np.asarray(flow_x_jpg)
    flow_y_arr = np.asarray(flow_y_jpg)

    flow_np = np.stack([flow_x_arr, flow_y_arr])

    B = 100.0
    flow_np = ((flow_np * (2 * B)) / 255) - B

    flow = torch.from_numpy(flow_np).float()
    # print(flow.size())

    if resizeDim is not None:
        flow = F.interpolate(flow[None], size=(resizeDim[1],resizeDim[0]))[0]
        factor0 = float(resizeDim[0]) / flow_np.shape[2]
        factor1 = float(resizeDim[1]) / flow_np.shape[1]

        flow[0, :, :] *= factor0
        flow[1, :, :] *= factor1
    
    if discrim_crop != None:
        flow = flow[discrim_crop[1]:discrim_crop[3], discrim_crop[0]:discrim_crop[2]]

    return flow.clone().detach()


def _flow_loader_npz(filename):
    flows = np.load(filename)
    flo13 = flows["flo13"]
    flo31 = flows["flo31"]
    if re.search("moving_gif", filename):
        flo13 = flo13[:,:,140:-140]
        flo31 = flo31[:,:,140:-140]
    return flo13, flo31


def _img_loader(path, resize=None):
    img = Image.open(path)
    if resize != None:
        img = img.resize(resize)
    img_rgb = img.convert("RGB")

    return img, img_rgb


def _get_global_max_flow(flowPath):
    _max = 0.0
    for paths in flowPath:
        flo13_path, flo31_path = paths
        flo13 = cv2.imread(flo13_path)
        flo31 = cv2.imread(flo31_path)
        thismax = np.amax(np.stack((flo13, flo31)))
        if thismax > _max:
            _max = thismax
    return _max


class BlenderAniTripletPatchTest2(data.Dataset):
    def __init__(self, args, data_root, csv_root, num_ib_frames, data_source, csv_filename=None, \
                 img_size=(2048,1024), resize=None, flow_type="tvl1", flow_root=None, small_dataset=False, \
                 dt=False):
        # Populate the list with image paths for all the
        # frame in `root`.
        framesPath, \
            framesFolder, \
            framesIndex, \
            flowPath = _make_csv_dataset(data_root, csv_root, num_ib_frames, data_source, \
                                         csv_filename_in=csv_filename, flow_type=flow_type, flow_root=flow_root, \
                                         small_dataset=small_dataset, dt=dt)

        #LILTHING: SO FAR THE FOLLOWING FUNCTION ONLY WORKS FOR PNG TVL1 JUST SAYING
        # maxflow = _get_global_max_flow(flowPath)

        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))
                
        self.root           = data_root
        self.img_size       = img_size
        self.framesPath     = framesPath
        self.framesFolder   = framesFolder
        self.framesIndex    = framesIndex
        self.flowPath       = flowPath
        self.resize         = resize
        self.flow_type      = flow_type
        self.num_ib_frames  = num_ib_frames
        # self.maxflow        = maxflow


    def __getitem__(self, index):
        sample = []
        rgb_sample = []
        masks = []

        folders = []
        indices = []

        flow = []

        to_tensor = transforms.PILToTensor()

        frameRange = [0, 1, 2]
        if self.num_ib_frames == 7:
            flowRange = [0, 2]
        else:
            flowRange = [0, 1]
    
        # Loop over for all frames corresponding to the `index`.
        for frameIndex in frameRange:
            image, image_rgb = _img_loader(self.framesPath[index][frameIndex], resize=self.resize)
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
        
        if self.flow_type == "tvl1":
            for flowIndex in flowRange:
                if self.num_ib_frames == 7:
                    flow_out = _flow_loader_jpg(self.flowPath[index][flowIndex], \
                                                self.flowPath[index][flowIndex+1], \
                                                resizeDim=self.img_size)
                else:
                    flow_out = _flow_loader_png(self.flowPath[index][flowIndex])
                flow.append(flow_out)
        elif self.flow_type == "pips":
            print(self.flowPath[index][0])
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
    




    



