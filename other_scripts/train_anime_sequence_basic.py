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
        sys.stdout.flush()

        sample, rgb_sample, folder, index, masks, flow = trainData

        #get flow
        flow_up_13 = flow[0]
        flow_up_31 = flow[1]

        F12i = flow_up_13.float().cuda()
        F21i = flow_up_31.float().cuda()
        maxval = torch.max(torch.max(F12i), torch.max(F21i)).cpu().detach().item()
        F12i = (1./maxval) * F12i
        F21i = (1./maxval) * F21i

        # Format batch
        # srcframe = sample[0]
        # ibframe = sample[1]
        # trgframe = sample[2]
        srcframe = rgb_sample[0]
        ibframe = rgb_sample[1]
        trgframe = rgb_sample[2]
        # It_truth = ibframe.cuda().float()
        I1 = srcframe.cuda().float()
        I2 = trgframe.cuda().float()

        srcmask = masks[0].cuda().float()
        ibmask = masks[1].cuda().float()
        trgmask = masks[2].cuda().float()
        # I1 = srcmask
        # I2 = trgmask

        num_bg = np.sum(np.where(ibmask.cpu().detach().numpy()==0, 1, 0))
        num_fg = np.sum(np.where(ibmask.cpu().detach().numpy()==1, 1, 0))
        if num_fg > 0:
            pos_weight_in = torch.FloatTensor([num_bg/num_fg]).cuda()
        else:
            pos_weight_in = torch.FloatTensor([1.]).cuda()

        if epoch % 10 == 0:
            if not os.path.exists(config.train_store_path + '/' + folder[0][0] + '/epoch_'+str(epoch)):
                os.makedirs(config.train_store_path + '/' + folder[0][0] + '/epoch_'+str(epoch))

        # saves the input images in a folder
            srcmask_path = os.path.join(config.train_store_path, folder[0][0], "epoch_"+str(epoch), index[0][0]+".png")
            trgmask_path = os.path.join(config.train_store_path, folder[-1][0], "epoch_"+str(epoch), index[-1][0]+".png")
            save_mask_to_img(srcmask[0].cpu().detach().numpy(), srcmask_path)
            save_mask_to_img(trgmask[0].cpu().detach().numpy(), trgmask_path)
            save_flow_to_img(F12i.cpu(), os.path.join(config.train_store_path, folder[1][0], "epoch_"+str(epoch), index[1][0]+"_F12"), epoch)
            save_flow_to_img(F21i.cpu(), os.path.join(config.train_store_path, folder[1][0], "epoch_"+str(epoch), index[1][0]+"_F21"), epoch)

        input_cat = torch.cat([I1, I2, F12i, F21i], 1)
        outputs_raw = model(input_cat)
        #push values to be between 0 and 1
        outputs = outputs_raw.sigmoid()
        output_answer = torch.where(outputs > 0.5, 1., 0.)

        input_outfile = os.path.join(config.train_store_path, folder[1][0], "epoch_"+str(epoch), index[1][0]+"_mask.png")
        output_outfile = os.path.join(config.train_store_path, folder[1][0], "epoch_"+str(epoch), index[1][0]+"_est_mask.png")
        output_grayscale_outfile = os.path.join(config.train_store_path, folder[1][0], "epoch_"+str(epoch), index[1][0]+"_grayscale.png")
        output_np = output_answer.cpu().detach().numpy()
        target_np = ibmask.cpu().detach().numpy()
        output_not_binarized_np = outputs.cpu().detach().numpy()
        
        if epoch % 10 == 0:
            save_mask_to_img(output_np[0], output_outfile)
            save_mask_to_img(target_np[0], input_outfile)
            save_mask_to_img(output_not_binarized_np[0], output_grayscale_outfile)

        loss = F.binary_cross_entropy_with_logits(outputs, ibmask, pos_weight=pos_weight_in)

        loss.backward()
        optimizer.step()
        running_loss += loss.detach().item()

    scheduler.step()
    cur_loss = running_loss / (len(trainloader))

    print("epoch", epoch, "loss", cur_loss)
    loss_arr.append(cur_loss)

    return loss_arr, model



def main(config, args):
    print("tae is goin for it on ig")

    trainset = datas.BlenderAniTripletPatch(args, config.trainset_root, \
                                            config.dataset_root_filepath_train, \
                                            config.num_ib_frames, \
                                            img_size=config.test_size, \
                                            flow_type=config.flow_type, \
                                            flow_root=config.trainflow_root, \
                                            random_reverse=config.random_reverse, \
                                            overfit=config.overfit)

    sampler = torch.utils.data.RandomSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    to_img = transforms.ToPILImage()
 
    sys.stdout.flush()

    # create the generator
    if config.model == "UNet_RRDB":
        model = Unet_RRDB(
            encoder_name="rrdb",
            kernel_size=1,
            # in_channels=6,
            in_channels=10,
            classes=1,
        )
        model = model.cuda()
    elif config.model == "UNet":
        model = Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=10,
            classes=1
        )
    else: #original model
        print("you're trying to use unet or a different model? go back to ye olde script")
        return
    
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100)

    print('Everything prepared. Ready to train...')
    sys.stdout.flush()

    epoch_arr = []
    loss_arr = []

    num_epochs = 501

    cur_epoch = config.cur_epoch
    for epoch in range(cur_epoch, num_epochs):
        print("######### EPOCH", epoch, "##########")
        epoch_arr.append(epoch)
        print("ilu yoongi")
        loss_arr, \
            model = training_loop(epoch, model, trainloader, trainset,\
                                    optimizer, scheduler, \
                                    config, loss_arr, \
                                    to_img)

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

