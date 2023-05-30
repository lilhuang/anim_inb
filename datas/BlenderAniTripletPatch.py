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


def _make_dataset(data_root, ibs, img_size, dataset_filepath, flow_type="tvl1", \
                  flow_root=None, overfit=False, small_dataset=False, dt=False):
    if overfit:
        # LILTHING: note this only supports blender cube 1 example overfitting
        # will need to write one for SU/suzanne if necessary
        dir_ = "t_bezier_s_facing_bezier_r_none_1_png_2048x1024_0_960"
        full_path = os.path.join(data_root, str(ibs)+"ib", dir_)

        src_name = os.path.join(full_path, "frame1.png")
        inb_name = os.path.join(full_path, "frame2.png")
        trg_name = os.path.join(full_path, "frame3.png")
        framesPath = [[src_name, inb_name, trg_name]]
        framesFolder = [[dir_, dir_, dir_]]
        framesIndex = [['0', '1', '2']]
        if flow_type == "tvl1":
            flow_path = os.path.join(flow_root, str(ibs)+"ib", dir_)
            if ibs == 7:
                flow13_x = os.path.join(flow_path, "flo13_x.jpg")
                flow13_y = os.path.join(flow_path, "flo13_y.jpg")
                flow31_x = os.path.join(flow_path, "flo31_x.jpg")
                flow31_y = os.path.join(flow_path, "flo31_y.jpg")

                flowPath = [[flow13_x, flow13_y, flow31_x, flow31_y]]
            else:
                flow13 = os.path.join(flow_path, "flo13.png")
                flow31 = os.path.join(flow_path, "flo31.png")
                flowPath = [[flow13, flow31]]
    else:
        framesPath = []
        framesFolder = []
        framesIndex = []
        flowPath = []

        sample_dirs = os.listdir(os.path.join(data_root, str(ibs)+"ib"))
        for i, dir_ in enumerate(sample_dirs):
            if (not small_dataset) or (small_dataset and i % 10 == 0):
                full_path = os.path.join(data_root, str(ibs)+"ib", dir_)
                    
                src_name = os.path.join(full_path, "frame1.png")
                inb_name = os.path.join(full_path, "frame2.png")
                trg_name = os.path.join(full_path, "frame3.png")
                framesPath.append([src_name, inb_name, trg_name])

                framesFolder.append([dir_, dir_, dir_])
                framesIndex.append(['0', '1', '2'])

                if flow_type == "tvl1":
                    flow_path = os.path.join(flow_root, str(ibs)+"ib", dir_)                            
                    if ibs == 7:
                        flow13_x = os.path.join(flow_path, "flo13_x.jpg")
                        flow13_y = os.path.join(flow_path, "flo13_y.jpg")
                        flow31_x = os.path.join(flow_path, "flo31_x.jpg")
                        flow31_y = os.path.join(flow_path, "flo31_y.jpg")

                        flowPath.append([flow13_x, flow13_y, flow31_x, flow31_y])
                    else:
                        if dt:
                            flow13 = os.path.join(flow_path, "flow13_dt.png")
                            flow31 = os.path.join(flow_path, "flow31_dt.png")
                        else:
                            flow13 = os.path.join(flow_path, "flo13.png")
                            flow31 = os.path.join(flow_path, "flo31.png")
                        flowPath.append([flow13, flow31])
    if flow_type != None:
        return framesPath, framesFolder, framesIndex, flowPath
    else:
        return framesPath, framesFolder, framesIndex, []


def _make_csv_dataset(data_root, csv_root, num_ib_frames, data_source, \
                      flow_type="tvl1", flow_root=None, small_dataset=False, dt=False):
    if small_dataset:
        framesPath_file = dataset_filepath+"_small_framesPath.pkl"
        framesFolder_file = dataset_filepath+"_small_framesFolder.pkl"
        framesIndex_file = dataset_filepath+"_small_framesIndex.pkl"
        if not dt:
            flowPath_file = dataset_filepath+"_small_flowPath.pkl"
        else:
            flowPath_file = dataset_filepath+"_small_dt_flowPath.pkl"
    else:
        framesPath_file = dataset_filepath+"_framesPath.pkl"
        framesFolder_file = dataset_filepath+"_framesFolder.pkl"
        framesIndex_file = dataset_filepath+"_framesIndex.pkl"
        if not dt:
            flowPath_file = dataset_filepath+"_flowPath.pkl"
        else:
            flowPath_file = dataset_filepath+"_dt_flowPath.pkl"
    
    if os.path.exists(framesPath_file) and os.path.exists(framesFolder_file) and \
            os.path.exists(framesIndex_file):
        if small_dataset:
            print("small test shit was saved, unpickling now! taetae :D boxy smile!")
        else:
            print("test shit was saved, unpickling now! taetae :D boxy smile!")
        with open(framesPath_file, 'rb') as f:
            framesPath = pickle.load(f)
        with open(framesFolder_file, 'rb') as f:
            framesFolder = pickle.load(f)
        with open(framesIndex_file, 'rb') as f:
            framesIndex = pickle.load(f)
        if flow_type != None and os.path.exists(flowPath_file):
            with open(flowPath_file, 'rb') as f:
                flowPath = pickle.load(f)
    else:
        if small_dataset:
            print("small test shit was not saved :( yoongi ._____. face")
        else:
            print("test shit was not saved :( yoongi ._____. face")
        framesPath = []
        framesFolder = []
        framesIndex = []
        flowPath = []

        csv_filename = "train_triplets_" + str(num_ib_frames) + "ib.csv"
        csv_file = os.path.join(csv_root, csv_filename)
        with open(csv_file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            i = 0
            for row in reader:
                if (not small_dataset) or (small_dataset and i % 10 == 0):
                    src_name = row[0]
                    inb_name = row[1]
                    trg_name = row[2]
                    framesPath.append([src_name, inb_name, trg_name])

                    if data_source == "SU":
                        regex = os.path.join(data_root, "(.*)", "frame([0-9]+).png")
                    else:
                        regex = os.path.join(data_root, "(.*)", "frame_([0-9]+).png")
                    
                    match = re.search(regex, src_name)
                    match2 = re.search(regex, trg_name)
                    src_num = match.group(2)
                    trg_num = match2.group(2)
                    if data_source == "SU":
                        folder = match.group(1)+"_"+src_num+"_to_"+trg_num
                    else:
                        folder = match.group(1)

                    framesFolder.append([folder, folder, folder])
                    framesIndex.append(['0', '1', '2'])
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
                        flow = row[3]
                        flowPath.append(flow)
                i += 1

        with open(framesPath_file, 'wb') as f:
            pickle.dump(framesPath, f)
        with open(framesFolder_file, 'wb') as f:
            pickle.dump(framesFolder, f)
        with open(framesIndex_file, 'wb') as f:
            pickle.dump(framesIndex, f)
        if flow_type != None:
            with open(flowPath_file, 'wb') as f:
                pickle.dump(flowPath, f)

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
    return flo13, flo31


def _img_loader(path, img_size=None):
    img = Image.open(path)
    if img_size != None:
        img = img.resize(img_size)
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

    
class BlenderAniTripletPatch(data.Dataset):
    def __init__(self, args, data_root, ibs, data_source, \
                 img_size=(2048,1024), flow_type="tvl1", flow_root=None, csv_root=None, \
                 random_reverse=False, overfit=False, small_dataset=False, dt=False, csv=False):
        # Populate the list with image paths for all there
        # frame in `root`.
        if not csv:
            framesPath, \
                framesFolder, \
                framesIndex, \
                flowPath = _make_dataset(data_root, ibs, img_size, \
                                        flow_type=flow_type, flow_root=flow_root, \
                                        overfit=overfit, small_dataset=small_dataset, \
                                        dt=dt)
        else:
            framesPath, \
            framesFolder, \
            framesIndex, \
            flowPath = _make_csv_dataset(data_root, csv_root, ibs, \
                                         data_source, flow_type=flow_type, flow_root=flow_root, \
                                         small_dataset=small_dataset, dt=dt)
        
        #LILTHING: SO FAR THE FOLLOWING FUNCTION ONLY WORKS FOR PNG TVL1 JUST SAYING
        # maxflow = _get_global_max_flow(flowPath)

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
        # self.maxflow        = maxflow


    def __getitem__(self, index):
        sample = []
        rgb_sample = []
        masks = []

        folders = []
        indices = []

        flow = []

        to_tensor = transforms.PILToTensor()

        reverse = 0
        if (self.random_reverse):
            ### Data Augmentation ###
            reverse = random.randint(0, 1)
            if reverse:
                frameRange = [2, 1, 0]
                if self.ibs == 7:
                    flowRange = [2, 0]
                else:
                    flowRange = [1, 0]
            else:
                frameRange = [0, 1, 2]
                if self.ibs == 7:
                    flowRange = [0, 2]
                else:
                    flowRange = [0, 1]
        else:
            frameRange = [0, 1, 2]
            if self.ibs == 7:
                flowRange = [0, 2]
            else:
                flowRange = [0, 1]

    
        # Loop over for all frames corresponding to the `index`.
        for frameIndex in frameRange:
            # Open image using pil and augment the image.
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

        if self.flow_type == "tvl1":
            for flowIndex in flowRange:
                if self.ibs == 7:
                    flow_out = _flow_loader_jpg(self.flowPath[index][flowIndex], \
                                                self.flowPath[index][flowIndex+1])
                else:
                    flow_out = _flow_loader_png(self.flowPath[index][flowIndex])
                flow.append(flow_out)
        elif self.flow_type == "pips":
            flo13, flo31 = _flow_loader_npz(self.flowPath[index])
            flow.append(flo13)
            flow.append(flo31)

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
    
