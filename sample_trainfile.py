import models
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import re
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
import preprocess
from pytorch3d.loss import chamfer_distance
from lpips_pytorch import lpips

###### HI YIANNI OVER HERE #############
from sample_dataloader import SampleDataloader
########################################

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


def plot_loss(config, epoch_arr, loss_arr, loss_d_arr, \
                loss_1_arr, loss_2_arr, test_epoch_arr, test_loss_arr):
    if not os.path.exists(config.metrics_dir):
        os.makedirs(config.metrics_dir)
    
    fig, axs = plt.subplots(3)
    axs[0].set_title('gen/disc (b/g)')
    axs[0].plot(epoch_arr, loss_arr, 'blue')
    axs[0].plot(epoch_arr, loss_d_arr, 'green')
    axs[1].set_title('1/2 (r/p)')
    axs[1].plot(epoch_arr, loss_1_arr, 'red')
    axs[1].plot(epoch_arr, loss_2_arr, 'purple')
    axs[2].set_title('test')
    axs[2].plot(test_epoch_arr, test_loss_arr, 'orange')

    filename = os.path.join(config.metrics_dir, config.loss_img_path)
    plt.savefig(filename)
    plt.close('all')


def save_psnr_ssim_to_txt(config, cur_psnr, cur_ssim, cur_cham, cur_lpips, epoch):
    if not os.path.exists(config.metrics_dir):
        os.makedirs(config.metrics_dir)
    textpath = os.path.join(config.metrics_dir, "psnr_ssim_epoch_"+str(epoch)+".txt")
    with open(textpath, 'w') as f:
        f.write("psnr "+str(cur_psnr)+" ssim "+str(cur_ssim)+" cham "+str(cur_cham)+" lpips "+str(cur_lpips))


def training_loop(epoch, model, trainloader, \
                    trainset, optimizer, scheduler, \
                    config, loss_arr, to_img):
    #  start training...
    model.train()
    running_loss = 0.0

    ######### HI YIANNI OVER HERE ####################
    for trainIndex, trainData in enumerate(trainloader):
        if trainIndex % 10 == 0:
            print('Training {}/{}-th group...'.format(trainIndex, len(trainloader)))
        sys.stdout.flush()

        sample, rgb_sample, folder, index, masks, flow = trainData
    #############################################################

        #get flow
        if config.flow_type != None:
            flow_15 = flow[0][0]
            flow_51 = flow[0][1]
            flow_13 = flow[0][2]
            flow_31 = flow[0][3]
            flow_35 = flow[0][4]
            flow_53 = flow[0][5]

            F15 = flow_15.float().cuda().to(memory_format=torch.channels_last)
            F51 = flow_51.float().cuda().to(memory_format=torch.channels_last)
            F13 = flow_13.float().cuda().to(memory_format=torch.channels_last)
            F31 = flow_31.float().cuda().to(memory_format=torch.channels_last)
            F35 = flow_35.float().cuda().to(memory_format=torch.channels_last)
            F53 = flow_53.float().cuda().to(memory_format=torch.channels_last)

        # Format batch
        srcmask = masks[0].cuda().float().to(memory_format=torch.channels_last)
        ibmask = masks[1].cuda().float().to(memory_format=torch.channels_last)
        trgmask = masks[2].cuda().float().to(memory_format=torch.channels_last)

        #weight bce loss by inverse distance from black pixel
        num_bg = np.sum(np.where(ibmask.cpu().detach().numpy()==0, 1, 0))
        num_fg = np.sum(np.where(ibmask.cpu().detach().numpy()==1, 1, 0))
        p_bg = num_bg/(num_bg+num_fg)
        loss_weights = (1-ibmask)*(2*p_bg - 1) + (1 - p_bg)
        loss_weights = loss_weights.to(memory_format=torch.channels_last)

        outpath = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[0][0])
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        
        if epoch % 10 == 0:
            save_flow_to_img(F15.cpu(), os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F15"), epoch)
            save_flow_to_img(F51.cpu(), os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F51"), epoch)
            save_flow_to_img(F13.cpu(), os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F13"), epoch)
            save_flow_to_img(F53.cpu(), os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F53"), epoch)
            save_flow_to_img(F13_output.cpu(), os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F13_est"), epoch)
            save_flow_to_img(F53_output.cpu(), os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F53_est"), epoch)
            # saves the input images in a folder
            srcmask_path = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[0][0], index[0][0]+".png")
            trgmask_path = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[-1][0], index[-1][0]+".png")
            save_mask_to_img(srcmask[0].cpu().detach().numpy(), srcmask_path)
            save_mask_to_img(trgmask[0].cpu().detach().numpy(), trgmask_path)
        
        input_cat = torch.cat([srcmask, trgmask], 1)
        input_cat = input_cat.to(memory_format=torch.channels_last)
        outputs, _ = model(input_cat)
        outputs = outputs.to(memory_format=torch.channels_last)
        output_answer = torch.where(outputs > 0.5, 1., 0.)

        model.zero_grad()
        optimizer.zero_grad()

        # calculate loss(es)
        loss = 0.0
        if config.recon_loss:
            if config.mask_loss:
                loss += torch.mean(torch.mul(F.binary_cross_entropy_with_logits(outputs, ibmask), loss_weights))
            elif config.l1_loss:
                loss += torch.mean(torch.mul(F.l1_loss(outputs, ibmask, reduction='none'), loss_weights))
            elif config.l2_loss:
                loss += torch.mean(torch.mul(F.mse_loss(outputs, ibmask, reduction='none'), loss_weights))

        loss.backward()
        optimizer.step()
        running_loss += loss.detach().item()

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


def testing_loop(epoch, model, model_1, model_2, testloader, testset, config, \
                    to_img, test_loss_arr):
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    running_cham = 0.0
    running_lpips = 0.0

    psnr = PSNR().cuda()
    ssim = SSIM(n_channels=1).cuda()

    #  start testing...
    with torch.no_grad():
        model.eval()

        ############# HI YIANNI OVER HERE ########################
        for validationIndex, validationData in enumerate(testloader):
            print('Testing {}/{}-th group...'.format(validationIndex, len(testloader)))
            sys.stdout.flush()
            
            sample, rgb_sample, folder, index, masks, flow = validationData
        ############################################################

            if config.flow_type != None:
                flow_15 = flow[0][0]
                flow_51 = flow[0][1]
                flow_13 = flow[0][2]
                flow_31 = flow[0][3]
                flow_35 = flow[0][4]
                flow_53 = flow[0][5]

                F15 = flow_15.float().cuda().to(memory_format=torch.channels_last)
                F51 = flow_51.float().cuda().to(memory_format=torch.channels_last)
                F13 = flow_13.float().cuda().to(memory_format=torch.channels_last)
                F31 = flow_31.float().cuda().to(memory_format=torch.channels_last)
                F35 = flow_35.float().cuda().to(memory_format=torch.channels_last)
                F53 = flow_53.float().cuda().to(memory_format=torch.channels_last)

            srcmask = masks[0].cuda().float().to(memory_format=torch.channels_last)
            ibmask = masks[1].cuda().float().to(memory_format=torch.channels_last)
            trgmask = masks[2].cuda().float().to(memory_format=torch.channels_last)

            #weight bce loss by inverse distance from black pixel
            num_bg = np.sum(np.where(ibmask.cpu().detach().numpy()==0, 1, 0))
            num_fg = np.sum(np.where(ibmask.cpu().detach().numpy()==1, 1, 0))
            p_bg = num_bg/(num_bg+num_fg)
            loss_weights = (1-ibmask)*(2*p_bg - 1) + (1 - p_bg)
            loss_weights = loss_weights.to(memory_format=torch.channels_last)

            outpath = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[0][0])
            if not os.path.exists(outpath):
                os.makedirs(outpath)

            srcmask_path = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[0][0], index[0][0]+".png")
            trgmask_path = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[-1][0], index[-1][0]+".png")
            save_mask_to_img(srcmask[0].cpu().detach().numpy(), srcmask_path)
            save_mask_to_img(trgmask[0].cpu().detach().numpy(), trgmask_path)
            if not config.flow_type == None:
                save_flow_to_img(F15.cpu(), os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F15"), epoch)
                save_flow_to_img(F51.cpu(), os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F51"), epoch)
                save_flow_to_img(F13.cpu(), os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F13"), epoch)
                save_flow_to_img(F53.cpu(), os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F53"), epoch)
                save_flow_to_img(F13_output.cpu(), os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F13_est"), epoch)
                save_flow_to_img(F53_output.cpu(), os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_F53_est"), epoch)
            
            input_cat = torch.cat([srcmask, trgmask], 1)
            outputs, _ = model(input_cat)
            outputs = outputs.to(memory_format=torch.channels_last)
            output_answer = torch.where(outputs > 0.5, 1., 0.)

            loss = torch.mean(torch.mul(F.binary_cross_entropy_with_logits(outputs, ibmask), loss_weights))
            running_loss += loss.detach().item()
            running_psnr += psnr(output_answer.detach(), ibmask.detach())
            running_ssim += ssim(output_answer.detach(), ibmask.detach())
            running_cham += chamfer_distance(output_answer.detach()[0], ibmask.detach()[0])[0].detach().cpu().item()
            running_lpips += lpips(output_answer, ibmask, net_type='vgg', version='0.1').detach().cpu().squeeze().item()

            input_outfile = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_mask.png")
            output_outfile = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_est_mask.png")
            output_grayscale_outfile = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_grayscale.png")
            output_np = output_answer.data.cpu().detach().numpy()
            target_np = ibmask.cpu().detach().numpy()
            output_not_binarized_np = outputs.cpu().detach().numpy()

            save_mask_to_img(output_np[0], output_outfile)
            save_mask_to_img(target_np[0], input_outfile)
            save_mask_to_img(output_not_binarized_np[0], output_grayscale_outfile)

        cur_loss = running_loss / len(testloader)
        cur_psnr = running_psnr / len(testloader)
        cur_ssim = running_ssim / len(testloader)
        cur_cham = running_cham / len(testloader)
        cur_lpips = running_lpips / len(testloader)

        print("epoch", epoch, "loss", cur_loss)
        test_loss_arr.append(cur_loss)

    return test_loss_arr, cur_psnr, cur_ssim, cur_cham, cur_lpips, model



def main(config, args):
    ##### HI YIANNI OVER HERE ###########
    trainset = SampleDataloader(args, "/path/to/training/data", \
                                            1, \
                                            "pt_JamesBaxterChel", \
                                            "/path/to/csv/filename", \
                                            flow_root="/path/to/flow/data", \
                                            csv_root="/path/to/csv/files", \
                                            csv=True, train=True)
    testset = SampleDataloader(args, "/path/to/test/data", \
                                            1, \
                                            "pt_JamesBaxterChel", \
                                            "/path/to/csv/filename", \
                                            flow_root="/path/to/flow/data", \
                                            csv_root="/path/to/csv/files", \
                                            csv=True, train=False)
    sampler = torch.utils.data.RandomSampler(trainset)
    test_sampler = torch.utils.data.SequentialSampler(testset)
    trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler, batch_size=16, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, sampler=test_sampler, batch_size=1, shuffle=False, num_workers=0)
    ###################################
    
    
    
    to_img = transforms.ToPILImage()
 
    sys.stdout.flush()

    # initialize models
    model = Unet(
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        encoder_depth=config.encoder_depth,
        decoder_channels=config.decoder_channels,
        in_channels=config.in_channels,
        classes=1
    )
    if config.checkpoint_in != None:
        dict1 = torch.load(config.checkpoint_in)
        newdict = OrderedDict()
        for key, value in dict1.items():
            name = key[7:]
            newdict[name] = value
        model.load_state_dict(newdict, strict=True)

        dict2 = torch.load(config.checkpoint_in_1)
        newdict2 = OrderedDict()
        for key, value in dict2.items():
            name = key[7:]
            newdict2[name] = value
        model_1.load_state_dict(newdict2, strict=True)

        dict3 = torch.load(config.checkpoint_in_2)
        newdict3 = OrderedDict()
        for key, value in dict3.items():
            name = key[7:]
            newdict3[name] = value
        model_2.load_state_dict(newdict3, strict=True)
    
    model = nn.DataParallel(model)
    model = model.cuda().to(memory_format=torch.channels_last)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    scheduler = optim.lr_scheduler.SequentialLR(
                                            optimizer,
                                            [
                                                optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=200),
                                                optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200),
                                            ],
                                            milestones=[200],
                                        )

    print('Everything prepared. Ready to train...')
    sys.stdout.flush()

    epoch_arr = []
    loss_arr = []

    test_epoch_arr = []
    test_loss_arr = []

    for epoch in range(config.cur_epoch, config.num_epochs):
        print("######### EPOCH", epoch, "##########")
        epoch_arr.append(epoch)
        loss_arr, \
            model = training_loop(epoch, model, trainloader, trainset,\
                                    optimizer, \
                                    scheduler, \
                                    config, loss_arr,\
                                    to_img)
        if epoch % 10 == 0:
            test_epoch_arr.append(epoch)
            test_loss_arr, \
                cur_psnr, \
                cur_ssim, \
                cur_cham, \
                cur_lpips, \
                model = testing_loop(epoch, model, testloader, testset, config, \
                                        to_img, test_loss_arr)
            save_psnr_ssim_to_txt(config, cur_psnr, cur_ssim, cur_cham, cur_lpips, epoch)

            if not os.path.exists(config.checkpoint_latest_dir):
                os.makedirs(config.checkpoint_latest_dir)
            checkpoint_path = os.path.join(config.checkpoint_latest_dir, config.checkpoint_latest_file+str(epoch)+".pth")
            torch.save(model.state_dict(), checkpoint_path)

    return loss_arr


if __name__ == "__main__":

    # loading configures
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    config = Config.from_file(args.config)

    loss_arr = main(config, args)

    print("")
    print("######### RESULTS ###########")
    print("LOSS:", loss_arr)
    print("#############################")
    print("\a"*2)

