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
import torch.nn.functional as F
import argparse
from utils.config import Config
import torchvision.transforms as TF

import pdb

def _make_csv_dataset(data_root, root_flow, csv_root, num_ib_frames, dataset_filepath):
    framesPath_file = dataset_filepath+"_framesPath.pkl"
    flowPath_file = dataset_filepath+"_raft_flowPath.pkl"
    framesFolder_file = dataset_filepath+"_framesFolder.pkl"
    framesIndex_file = dataset_filepath+"_framesIndex.pkl"

    if os.path.exists(framesPath_file) and os.path.exists(flowPath_file) \
            and os.path.exists(framesFolder_file) and os.path.exists(framesIndex_file):
        print("train shit was saved, unpickling now! taetae :D boxy smile!")
        with open(framesPath_file, 'rb') as f:
            framesPath = pickle.load(f)
        with open(flowPath_file, 'rb') as f:
            flowPath = pickle.load(f)
        with open(framesFolder_file, 'rb') as f:
            framesFolder = pickle.load(f)
        with open(framesIndex_file, 'rb') as f:
            framesIndex = pickle.load(f)
    else:
        print("test shit was not saved :( yoongi ._____. face")
    
        framesPath = []
        flowPath = []
        framesFolder = []
        framesIndex = []
        maskPath = []

        csv_filename = "test_triplets_" + str(num_ib_frames) + "ib.csv"
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
                
                regex = os.path.join(data_root, "([.*]+)", "flow_([.*]+).npy")
                match = re.search(regex, row[3])
                folder = match.group(1)+"_"+match.group(2)

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


def _flow_loader_jpg(pathx, pathy, cropArea=None, resizeDim=None, shiftX=0, shiftY=0):
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
    
    # if cropArea is not None:
    #     flow = flow[:, cropArea[1]:cropArea[3], cropArea[0]:cropArea[2]]

    flow[0] -= shiftX
    flow[1] -= shiftY

    return flow.clone().detach()


class CSVAniTripletWithSGMFlowTest(data.Dataset):
    def __init__(self, data_root, csv_root, num_ib_frames, root_flow, \
                 dataset_filepath, transform=None, img_size=(2048,1024), \
                 dt=False, raft=False, resize=(2048,1024)):
        # Populate the list with image paths for all the
        # frame in `root`.dataset_root_filepath_test
        framesPath, flowPath, \
            framesFolder, framesIndex = _make_csv_dataset(data_root, root_flow, \
                                                 csv_root, num_ib_frames, dataset_filepath)
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
        self.dt             = dt
        self.raft           = raft
        self.resize         = resize

    def __getitem__(self, index):
        sample = []
        flow = []

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
    


if __name__ == "__main__":
    # pdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    config = Config.from_file(args.config)
    normalize1 = TF.Normalize(config.mean, [1.0, 1.0, 1.0])
    normalize2 = TF.Normalize([0, 0, 0], config.std)
    trans = TF.Compose([TF.ToTensor(), normalize1, normalize2, ])

    revmean = [-x for x in config.mean]
    revstd = [1.0 / x for x in config.std]
    revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
    revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
    revNormalize = TF.Compose([revnormalize1, revnormalize2])
   
    testset = CSVAniTripletWithSGMFlowTest(config.trainset_root, config.csv_root, \
                                              config.num_ib_frames, config.data_source, \
                                              config.train_flow_root, trans, config.test_size, \
                                              config.test_crop_size, train=False)
    test_sampler = torch.utils.data.SequentialSampler(testset)
    testloader = torch.utils.data.DataLoader(testset, sampler=test_sampler, batch_size=16, shuffle=False, num_workers=0)

    



