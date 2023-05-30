import models
import datas
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import shutil
import subprocess
from math import log10
import numpy as np
import datetime
from utils.config import Config
from collections import OrderedDict
import sys
import cv2
from utils.vis_flow import flow_to_color
import json
from tqdm import tqdm
from piqa import PSNR, SSIM
from sklearn.metrics import precision_score, accuracy_score, recall_score
from PIL import Image
import matplotlib.pyplot as plt
from models.segmentation_models_pytorch.unet import Unet
from models.segmentation_models_pytorch.unet_rrdb import Unet_RRDB
from models.discriminator_model.discriminator import Discriminator_square, \
                                                    Discriminator_non_square, \
                                                    Discriminator_square_dumb
from models.discriminator_model.patch_discriminators import Discriminator_1x1_output, \
                                                        Discriminator_256x256_output, \
                                                        Discriminator_30x30_output, \
                                                        Discriminator_252x252_output, \
                                                        Discriminator_patch, \
                                                        dcgan_weights_init
from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import upflow2
import preprocess

import pdb


def save_flow_to_img(flow, des, epoch):
    f = flow[0].data.cpu().numpy().transpose([1, 2, 0])
    fcopy = f.copy()
    fcopy[:, :, 0] = f[:, :, 1]
    fcopy[:, :, 1] = f[:, :, 0]
    cf = flow_to_color(-fcopy)
    cv2.imwrite(des + '_epoch_'+str(epoch)+'.jpg', cf)


def save_mask_to_img(mask, name):
    #note mask should be np array
    if mask.shape[0] == 1:
        mask = np.transpose(mask, (1, 2, 0))
    cv2.imwrite(name, (1 - mask)*255)


def plot_loss(config, epoch_arr, loss_arr):
    if not os.path.exists(config.metrics_dir):
        os.makedirs(config.metrics_dir)

    plt.plot(epoch_arr, loss_arr, 'blue')
    filename = os.path.join(config.metrics_dir, config.loss_img_path)
    plt.savefig(filename)
    plt.close('all')


def save_metrics_to_arr(config, epoch_arr, loss_arr):
    if not os.path.exists(config.metrics_dir):
        os.makedirs(config.metrics_dir)
    epoch_arr_path = os.path.join(config.metrics_dir, "epoch_arr.npy")
    np.save(epoch_arr_path, epoch_arr)
    loss_arr_path = os.path.join(config.metrics_dir, "loss_arr.npy")
    np.save(loss_arr_path, loss_arr)


def save_psnr_ssim_to_txt(config, cur_psnr, cur_ssim, epoch):
    if not os.path.exists(config.metrics_dir):
        os.makedirs(config.metrics_dir)
    textpath = os.path.join(config.metrics_dir, "psnr_ssim_epoch_"+str(epoch)+".txt")
    with open(textpath, 'w') as f:
        f.write("psnr "+str(cur_psnr)+" ssim "+str(cur_ssim))



def training_loop(epoch, model, trainloader, trainset, optimizer, scheduler, \
                    config, loss_arr, to_img):
    #  start training...
    model.train()
    running_loss = 0.0

    real_label = 1.
    fake_label = 0.

    for trainIndex, trainData in enumerate(trainloader):
        if trainIndex % 10 == 0:
            print('Training {}/{}-th group...'.format(trainIndex, len(trainloader)))
        # sys.stdout.flush()
        # model.zero_grad()
        optimizer.zero_grad()

        rgb_sample, masks, flow, folder, index = trainData

        #get flow
        flow_up_13 = flow[0]
        flow_up_31 = flow[1]
        
        F12i = flow_up_13.float().cuda()
        F21i = flow_up_31.float().cuda()
        F12i_black = torch.zeros(F12i.shape)
        F21i_black = torch.zeros(F21i.shape)

        # Format batch
        srcframe = rgb_sample[0]
        # ibframe = rgb_sample[1]
        trgframe = rgb_sample[2]
        # It_truth = ibframe.cuda().float()
        I1 = srcframe.cuda().float()
        I2 = trgframe.cuda().float()

        I1_black = torch.zeros(I1.shape)
        I2_black = torch.zeros(I2.shape)

        srcmask = masks[0].cuda().float()
        ibmask = masks[1].cuda().float()
        trgmask = masks[2].cuda().float()

        srcmask_blank = torch.ones(srcmask.shape)
        trgmask_blank = torch.ones(trgmask.shape)

        num_bg = np.sum(np.where(ibmask.cpu().detach().numpy()==0, 1, 0))
        num_fg = np.sum(np.where(ibmask.cpu().detach().numpy()==1, 1, 0))
        p_bg = num_bg/(num_bg+num_fg)
        # if num_fg > 0:
        #     pos_weight_in = torch.FloatTensor([num_bg/num_fg]).cuda()
        # else:
        #     pos_weight_in = torch.FloatTensor([1.]).cuda()
        # pos_weight_in = torch.FloatTensor([10.]).cuda()

        loss_weights = (1-ibmask)*(2*p_bg - 1) + (1 - p_bg)

        if epoch % 10 == 0:
            outpath = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[0][0])
            if not os.path.exists(outpath):
                os.makedirs(outpath)

            # saves the input images in a folder
            srcmask_path = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[0][0], index[0][0]+".png")
            trgmask_path = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[-1][0], index[-1][0]+".png")
            save_mask_to_img(srcmask[0].cpu().detach().numpy(), srcmask_path)
            save_mask_to_img(trgmask[0].cpu().detach().numpy(), trgmask_path)
            # I1_black_pil = to_img(I1_black[0])
            # I2_black_pil = to_img(I2_black[0])
            # I1_black_pil.save(srcmask_path)
            # I2_black_pil.save(trgmask_path)
            save_flow_to_img(F12i.cpu(), os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F12"), epoch)
            save_flow_to_img(F21i.cpu(), os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F21"), epoch)
        
        # input_cat = torch.cat([I1, I2, F12i, F21i], 1)
        # pdb.set_trace()
        input_cat = torch.cat([srcmask, trgmask, F12i, F21i], 1)
        outputs = model(input_cat)
        # push values to be between 0 and 1
        outputs = torch.sigmoid(outputs)

        # loss = F.binary_cross_entropy_with_logits(outputs, ibmask, \
        #             pos_weight=pos_weight_in)
        loss = F.binary_cross_entropy(outputs, ibmask, weight=loss_weights)

        loss.backward()
        optimizer.step()
        running_loss += loss.detach().item()

        output_answer = torch.where(outputs > 0.5, 1., 0.)
        input_outfile = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_mask.png")
        output_outfile = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_est_mask.png")
        output_grayscale_outfile = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_grayscale.png")
        output_np = output_answer.cpu().detach().numpy()
        target_np = ibmask.cpu().detach().numpy()
        output_not_binarized_np = outputs.cpu().detach().numpy()
        if epoch % 10 == 0:
            save_mask_to_img(output_np[0], output_outfile)
            save_mask_to_img(target_np[0], input_outfile)
            save_mask_to_img(output_not_binarized_np[0], output_grayscale_outfile)

    scheduler.step()
    cur_loss = running_loss / (len(trainloader))

    print("epoch", epoch, "loss", cur_loss)
    loss_arr.append(cur_loss)

    return loss_arr, model



def main(config, args):
    print("tae is goin for it on ig")
    trainset = datas.CSVEXRTriplet(config.csv_root, config.trainset_root, config.flow_root, config.num_ib_frames, \
                                    train=True, img_size=config.test_size, patch_size=config.patch_size, \
                                    discrim_crop_size=None, random_flip=False, \
                                    random_reverse=config.random_reverse, dt=config.dt, \
                                    patch_location=None, overfit=config.overfit)

    sampler = torch.utils.data.RandomSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    to_img = transforms.ToPILImage()

    # create the generator
    model = Unet(
        encoder_name="resnet34",
        # encoder_weights="imagenet",
        encoder_weights=None,
        encoder_depth=6,
        decoder_channels=(512, 256, 128, 64, 32, 16),
        # in_channels=10,
        in_channels=6,
        classes=1
    )
    
    model = nn.DataParallel(model)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs-1)

    print('Everything prepared. Ready to train...')
    sys.stdout.flush()

    epoch_arr = []
    loss_arr = []

    for epoch in range(config.cur_epoch, config.num_epochs):
        print("######### EPOCH", epoch, "##########")
        epoch_arr.append(epoch)
        print("ilu yoongi")
        loss_arr, \
            model = training_loop(epoch, model, trainloader, trainset,\
                                    optimizer, scheduler, \
                                    config, loss_arr, to_img)

        if not os.path.exists(config.checkpoint_latest_dir):
            os.makedirs(config.checkpoint_latest_dir)
        checkpoint_path = os.path.join(config.checkpoint_latest_dir, config.checkpoint_latest_file+str(epoch)+".pth")
        torch.save(model.state_dict(), checkpoint_path)
        
        plot_loss(config, epoch_arr, loss_arr)
        save_metrics_to_arr(config, epoch_arr, loss_arr)

    return loss_arr


if __name__ == "__main__":

    # loading configures
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    #these are just for raft flow things
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    config = Config.from_file(args.config)

    loss_arr = main(config, args)

    print("")
    print("######### RESULTS ###########")
    print("LOSS:", loss_arr)
    print("#############################")
    print("\a"*2)

