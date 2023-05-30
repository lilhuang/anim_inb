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


def plot_loss(config, epoch_arr, loss_arr, loss_d_arr, test_epoch_arr, test_loss_arr):
    if not os.path.exists(config.metrics_dir):
        os.makedirs(config.metrics_dir)

    plt.plot(epoch_arr, loss_arr, 'blue')
    plt.plot(epoch_arr, loss_d_arr, 'green')
    plt.plot(test_epoch_arr, test_loss_arr, 'orange')
    filename = os.path.join(config.metrics_dir, config.loss_img_path)
    plt.savefig(filename)
    plt.close('all')


def save_metrics_to_arr(config, epoch_arr, loss_arr, loss_d_arr,\
                        test_epoch_arr, test_loss_arr):
    if not os.path.exists(config.metrics_dir):
        os.makedirs(config.metrics_dir)
    epoch_arr_path = os.path.join(config.metrics_dir, "epoch_arr.npy")
    np.save(epoch_arr_path, epoch_arr)
    loss_arr_path = os.path.join(config.metrics_dir, "loss_arr.npy")
    np.save(loss_arr_path, loss_arr)
    loss_arr_path = os.path.join(config.metrics_dir, "loss_d_arr.npy")
    np.save(loss_arr_path, loss_d_arr)
    
    test_epoch_arr_path = os.path.join(config.metrics_dir, "test_epoch_arr.npy")
    np.save(test_epoch_arr_path, test_epoch_arr)
    test_loss_arr_path = os.path.join(config.metrics_dir, "test_loss_arr.npy")
    np.save(test_loss_arr_path, test_loss_arr)



def save_progress_gif(results_root, epoch, folder):
    results_path = os.path.join(results_root, folder)
    
    gif_path_gt = os.path.join(results_path, "progress_gif_gt.gif")
    gif_path_gen = os.path.join(results_path, "progress_gif_gen.gif")
    gif_path_0 = os.path.join(results_path, "progress_gif_0_warp_from_inb.gif")
    gif_path_2 = os.path.join(results_path, "progress_gif_2_warp_from_inb.gif")
    
    workingdir_gt = os.path.join(results_path, "working_gif_dir_gt")
    workingdir_gen = os.path.join(results_path, "working_gif_dir_gen")
    workingdir_0 = os.path.join(results_path, "working_gif_dir_0_warp")
    workingdir_2 = os.path.join(results_path, "working_gif_dir_2_warp")
    
    if not os.path.exists(workingdir_gt):
        os.mkdir(workingdir_gt)
    if not os.path.exists(workingdir_gen):
        os.mkdir(workingdir_gen)
    if not os.path.exists(workingdir_2):
        os.mkdir(workingdir_2)
    if not os.path.exists(workingdir_0):
        os.mkdir(workingdir_0)
    
    cur_epoch = os.path.join(results_path, "epoch_{:03d}".format(epoch))
    if os.path.exists(cur_epoch):
        frame1_gt_path = os.path.join(cur_epoch, "1_mask.png")
        frame1_gen_path = os.path.join(cur_epoch, "1_est_mask.png")
        frame0_warp_path = os.path.join(cur_epoch, "0_warp_from_inb.png")
        frame2_warp_path = os.path.join(cur_epoch, "2_warp_from_inb.png")
        trg_frame1_gt_path = os.path.join(workingdir_gt, "epoch_{:03d}".format(epoch//10)+".png")
        trg_frame1_gen_path = os.path.join(workingdir_gen, "epoch_{:03d}".format(epoch//10)+".png")
        trg_frame0_warp_path = os.path.join(workingdir_0, "epoch_{:03d}".format(epoch//10)+".png")
        trg_frame2_warp_path = os.path.join(workingdir_2, "epoch_{:03d}".format(epoch//10)+".png")
        shutil.copy(frame1_gt_path, trg_frame1_gt_path)
        shutil.copy(frame1_gen_path, trg_frame1_gen_path)
        shutil.copy(frame0_warp_path, trg_frame0_warp_path)
        shutil.copy(frame2_warp_path, trg_frame2_warp_path)
    
    ########## I THINK THE PROBLEM IS IN THE FFMPEG COMMAND CHECK THIS!!!!!!!! ###################
    bashCommand_gen_gif = "ffmpeg -y -f image2 -framerate 1 -i "+workingdir_gen+"/epoch_%003d.png "+gif_path_gen
    bashCommand_gt_gif = "ffmpeg -y -f image2 -framerate 1 -i "+workingdir_gt+"/epoch_%003d.png "+gif_path_gt
    bashCommand_0_gif = "ffmpeg -y -f image2 -framerate 1 -i "+workingdir_0+"/epoch_%003d.png "+gif_path_0
    bashCommand_2_gif = "ffmpeg -y -f image2 -framerate 1 -i "+workingdir_2+"/epoch_%003d.png "+gif_path_2

    subprocess.run(bashCommand_gen_gif, shell=True)
    subprocess.run(bashCommand_gt_gif, shell=True)
    subprocess.run(bashCommand_0_gif, shell=True)
    subprocess.run(bashCommand_2_gif, shell=True)

    bashCommand_hstack = "ffmpeg -y -i "+gif_path_gt+" -i "+gif_path_gen+" -i "+gif_path_0+" -i "+gif_path_2+" -filter_complex \"[0:v][1:v]hstack=inputs=2[top]; [2:v][3:v]hstack=inputs=2[bottom]; [top][bottom]vstack=inputs=2[v]\" -map \"[v]\" "+results_path+"/progress.gif"
    subprocess.run(bashCommand_hstack, shell=True)


def save_psnr_ssim_to_txt(config, cur_psnr, cur_ssim, epoch):
    if not os.path.exists(config.metrics_dir):
        os.makedirs(config.metrics_dir)
    textpath = os.path.join(config.metrics_dir, "psnr_ssim_epoch_"+str(epoch)+".txt")
    with open(textpath, 'w') as f:
        f.write("psnr "+str(cur_psnr)+" ssim "+str(cur_ssim))


def backwarp(flow, image):
    W = image.shape[3]
    H = image.shape[2]
    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
    gridX = torch.Tensor(gridX).unsqueeze(0).expand_as(flow[:,0,:,:]).float().cuda()
    gridY = torch.Tensor(gridY).unsqueeze(0).expand_as(flow[:,1,:,:]).float().cuda()

    x = gridX + flow[:,0,:,:]
    y = gridY + flow[:,1,:,:]

    x = 2*(x/W - 0.5)
    y = 2*(y/H - 0.5)

    grid = torch.stack((x, y), dim=3)

    imgout = F.grid_sample(image, grid, mode="nearest")
    # warped = F.grid_sample(image, grid, mode="nearest")
    # both_0 = torch.bitwise_and(x==0, y==0)
    # imgout = torch.where(both_0, warped, image)
    
    return imgout



def training_loop(epoch, model, trainloader, trainset, optimizer, scheduler, \
                    config, loss_arr, loss_d_arr, to_img, netD, \
                    optimizerD, criterionD):
    #  start training...
    model.train()
    running_loss = 0.0
    running_d_loss = 0.0

    real_label = 1.
    fake_label = 0.

    for trainIndex, trainData in enumerate(trainloader):
        if trainIndex % 10 == 0:
            print('Training {}/{}-th group...'.format(trainIndex, len(trainloader)))
        sys.stdout.flush()

        rgb_sample, masks, folder, index = trainData

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        optimizerD.zero_grad()
        # Format batch
        srcframe = rgb_sample[0]
        # ibframe = rgb_sample[1]
        trgframe = rgb_sample[2]
        trg2frame = rgb_sample[4]
        # It_truth = ibframe.cuda().float()
        I1 = srcframe.cuda().float()
        I2 = trgframe.cuda().float()
        I3 = trg2frame.cuda().float()

        srcmask = masks[0].cuda().float()
        ibmask = masks[1].cuda().float()
        trgmask = masks[2].cuda().float()
        ibmask2 = masks[3].cuda().float()
        trg2mask = masks[4].cuda().float()

        #weight bce loss by inverse distance from black pixel
        num_bg = np.sum(np.where(ibmask.cpu().detach().numpy()==0, 1, 0))
        num_fg = np.sum(np.where(ibmask.cpu().detach().numpy()==1, 1, 0))
        p_bg = num_bg/(num_bg+num_fg)
        loss_weights = (1-ibmask)*(2*p_bg - 1) + (1 - p_bg)

        num_bg_2 = np.sum(np.where(ibmask2.cpu().detach().numpy()==0, 1, 0))
        num_fg_2 = np.sum(np.where(ibmask2.cpu().detach().numpy()==1, 1, 0))
        p_bg_2 = num_bg_2/(num_bg_2+num_fg_2)
        loss_weights_2 = (1-ibmask2)*(2*p_bg_2 - 1) + (1 - p_bg_2)
        
        if config.discrim == "patch":
            real_output1x1_1, real_output16x16_1, real_output70x70_1, real_output256x256_1 = netD(ibmask)
            real_output1x1_2, real_output16x16_2, real_output70x70_2, real_output256x256_2 = netD(ibmask2)
            real_outputs_1 = [real_output1x1_1, real_output16x16_1, real_output70x70_1, real_output256x256_1]
            real_outputs_2 = [real_output1x1_2, real_output16x16_2, real_output70x70_2, real_output256x256_2]
            label_1x1 = torch.full(real_output1x1_1.shape, real_label, dtype=torch.float).cuda()
            label_16x16 = torch.full(real_output16x16_1.shape, real_label, dtype=torch.float).cuda()
            label_70x70 = torch.full(real_output70x70_1.shape, real_label, dtype=torch.float).cuda()
            label_256x256 = torch.full(real_output256x256_1.shape, real_label, dtype=torch.float).cuda()
            labels = [label_1x1, label_16x16, label_70x70, label_256x256]
            errD_real = 0.
            for i in range(len(labels)):
                if i == 0:
                    continue
                errD_real_1 += criterionD(real_outputs_1[i], labels[i])
                errD_real_2 += criterionD(real_outputs_2[i], labels[i])
        else:
            real_output_1 = netD(ibmask)
            real_output_2 = netD(ibmask2)
            label = torch.full(real_output_1.shape, real_label, dtype=torch.float).cuda()
            errD_real_1 = criterionD(real_output_1, label)
            errD_real_2 = criterionD(real_output_2, label)
        
        errD_real = errD_real_1 + errD_real_2
        errD_real.backward()

        if epoch % 10 == 0:
            outpath = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[0][0])
            if not os.path.exists(outpath):
                os.makedirs(outpath)

        # saves the input images in a folder
            srcmask_path = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[0][0], index[0][0]+".png")
            trgmask_path = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[2][0], index[2][0]+".png")
            trg2mask_path = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[-1][0], index[-1][0]+".png")
            save_mask_to_img(srcmask[0].cpu().detach().numpy(), srcmask_path)
            save_mask_to_img(trgmask[0].cpu().detach().numpy(), trgmask_path)
            save_mask_to_img(trg2mask[0].cpu().detach().numpy(), trg2mask_path)
        input_cat_1 = torch.cat([srcmask, trgmask], 1)
        input_cat_2 = torch.cat([trgmask, trg2mask], 1)
        outputs_1 = model(input_cat_1)
        outputs_2 = model(input_cat_2)
        # push values to be between 0 and 1
        outputs_1 = torch.sigmoid(outputs_1)
        outputs_2 = torch.sigmoid(outputs_2)
        output_answer_1 = torch.where(outputs_1 > 0.5, 1., 0.)
        output_answer_2 = torch.where(outputs_2 > 0.5, 1., 0.)

        #LILTHING: gotta do this for both triplets
        ## Train discriminator with all-fake batch
        # Classify all fake batch with D
        if config.discrim == "patch":
            fake_output1x1_1, fake_output16x16_1, fake_output70x70_1, fake_output256x256_1 = netD(outputs_1.detach())
            fake_output1x1_2, fake_output16x16_2, fake_output70x70_2, fake_output256x256_2 = netD(outputs_2.detach())
            fake_outputs_1 = [fake_output1x1_1, fake_output16x16_1, fake_output70x70_1, fake_output256x256_1]
            fake_outputs_2 = [fake_output1x1_2, fake_output16x16_2, fake_output70x70_2, fake_output256x256_2]
            errD_fake_1 = 0.
            errD_fake_2 = 0.
            for i in range(len(fake_outputs_1)):
                if i == 0:
                    continue
                labels[i].fill_(fake_label)
                errD_fake_1 += criterionD(fake_outputs_1[i], labels[i])
                errD_fake_2 += criterionD(fake_outputs_2[i], labels[i])
        else:
            label.fill_(fake_label)
            fake_output_1 = netD(outputs_1.detach())
            fake_output_2 = netD(outputs_2.detach())
            # Calculate D's loss on the all-fake batch
            errD_fake_1 = criterionD(fake_output_1, label)
            errD_fake_2 = criterionD(fake_output_2, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake = errD_fake_1 + errD_fake_2
        errD_fake.backward()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        running_d_loss += errD.item()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        model.zero_grad()
        optimizer.zero_grad()

        if config.discrim == "patch":
            fake_output_again1x1_1, fake_output_again16x16_1, \
                fake_output_again70x70_1, fake_output_again256x256_1 = netD(outputs_1)
            fake_output_again1x1_2, fake_output_again16x16_2, \
                fake_output_again70x70_2, fake_output_again256x256_2 = netD(outputs_2)
            fake_outputs_again_1 = [fake_output_again1x1_1, fake_output_again16x16_1, fake_output_again70x70_1, fake_output_again256x256_1]
            fake_outputs_again_2 = [fake_output_again1x1_2, fake_output_again16x16_2, fake_output_again70x70_2, fake_output_again256x256_2]
            
            errG = 0.
            for i in range(len(fake_outputs_again_1)):
                if i == 0:
                    continue
                labels[i].fill_(real_label)
                errG += criterionD(fake_outputs_again_1[i], labels[i])
                errG += criterionD(fake_outputs_again_2[i], labels[i])
        else:
            label.fill_(real_label) # fake labels are real for generator cost
            fake_output_again_1 = netD(outputs_1)
            fake_output_again_2 = netD(outputs_2)
            # Calculate G's loss based on this output
            errG = criterionD(fake_output_again_1, label) + criterionD(fake_output_again_2, label)

        # calculate loss(es)
        loss = 0.0
        if config.recon_loss:
            loss += F.binary_cross_entropy(outputs_1, ibmask, \
                                        weight=loss_weights)
            loss += F.binary_cross_entropy(outputs_2, ibmask2, \
                                        weight=loss_weights_2)
        if config.gan_loss:
            loss += errG
        if config.warp_loss and epoch >= 10:
            input_cat_3 = torch.cat([outputs_1, outputs_2], 1)
            outputs_3 = model(input_cat_3)
            outputs_3 = torch.sigmoid(outputs_3)
            output_answer_3 = torch.where(outputs_3 > 0.5, 1., 0.)

            num_bg_trg = np.sum(np.where(trgmask.cpu().detach().numpy()==0, 1, 0))
            num_fg_trg = np.sum(np.where(trgmask.cpu().detach().numpy()==1, 1, 0))
            p_bg_trg = num_bg_trg/(num_bg_trg+num_fg_trg)
            loss_weights_trg = (1-trgmask)*(2*p_bg_trg - 1) + (1 - p_bg_trg)
            loss += (config.warp_weight*F.binary_cross_entropy(outputs_3, trgmask, \
                                            weight=loss_weights_trg))

        loss.backward()
        optimizer.step()
        running_loss += loss.detach().item()

        input_outfile = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_mask.png")
        output_outfile = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_est_mask.png")
        output_grayscale_outfile = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_grayscale.png")
        
        input2_outfile = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[3][0], index[3][0]+"_mask.png")
        output2_outfile = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[3][0], index[3][0]+"_est_mask.png")
        output2_grayscale_outfile = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[3][0], index[3][0]+"_grayscale.png")
        
        output_np = output_answer_1.cpu().detach().numpy()
        target_np = ibmask.cpu().detach().numpy()
        output_not_binarized_np = outputs_1.cpu().detach().numpy()
        output_np_2 = output_answer_2.cpu().detach().numpy()
        target_np_2 = ibmask2.cpu().detach().numpy()
        output_not_binarized_np_2 = outputs_2.cpu().detach().numpy()
        
        if epoch % 10 == 0:
            save_mask_to_img(output_np[0], output_outfile)
            save_mask_to_img(target_np[0], input_outfile)
            save_mask_to_img(output_not_binarized_np[0], output_grayscale_outfile)
            save_mask_to_img(output_np_2[0], output2_outfile)
            save_mask_to_img(target_np_2[0], input2_outfile)
            save_mask_to_img(output_not_binarized_np_2[0], output2_grayscale_outfile)
            if config.warp_loss and epoch >= 10:
                print("writing warp output")
                warp_outfile = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[2][0], index[2][0]+"_est_mask.png")
                grayscale_warp_outfile = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[2][0], index[2][0]+"_est_grayscale.png")
                output_3_np = output_answer_3.cpu().detach().numpy()
                output_3_not_binarized_np = outputs_3.cpu().detach().numpy()
                save_mask_to_img(output_3_np[0], warp_outfile)
                save_mask_to_img(output_3_not_binarized_np[0], grayscale_warp_outfile)

    scheduler.step()
    cur_loss = running_loss / (len(trainloader))
    cur_d_loss = running_d_loss / (len(trainloader))

    print("epoch", epoch, "loss", cur_loss, "discrim loss", cur_d_loss)
    loss_arr.append(cur_loss)
    loss_d_arr.append(cur_d_loss)

    return loss_arr, loss_d_arr, model


def testing_loop(epoch, model, testloader, testset, config, \
                    to_img, test_loss_arr):
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0

    psnr = PSNR().cuda()
    ssim = SSIM(n_channels=1).cuda()

    #  start testing...
    with torch.no_grad():
        model.eval()
        for validationIndex, validationData in enumerate(testloader):
            print('Testing {}/{}-th group...'.format(validationIndex, len(testloader)))
            sys.stdout.flush()
            rgb_sample, masks, folder, index = validationData

            frame1 = rgb_sample[0]
            frame2 = rgb_sample[2]
            frame3 = rgb_sample[4]
            I1 = frame1.cuda().float()
            I2 = frame2.cuda().float()
            I3 = frame3.cuda().float()

            srcmask = masks[0].cuda().float()
            ibmask = masks[1].cuda().float()
            trgmask = masks[2].cuda().float()
            ibmask2 = masks[3].cuda().float()
            trg2mask = masks[4].cuda().float()

            #weight bce loss by inverse distance from black pixel
            num_bg = np.sum(np.where(ibmask.cpu().detach().numpy()==0, 1, 0))
            num_fg = np.sum(np.where(ibmask.cpu().detach().numpy()==1, 1, 0))
            p_bg = num_bg/(num_bg+num_fg)
            loss_weights = (1-ibmask)*(2*p_bg - 1) + (1 - p_bg)

            num_bg_2 = np.sum(np.where(ibmask2.cpu().detach().numpy()==0, 1, 0))
            num_fg_2 = np.sum(np.where(ibmask2.cpu().detach().numpy()==1, 1, 0))
            p_bg_2 = num_bg_2/(num_bg_2+num_fg_2)
            loss_weights_2 = (1-ibmask2)*(2*p_bg_2 - 1) + (1 - p_bg_2)

            outpath = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[0][0])
            if not os.path.exists(outpath):
                os.makedirs(outpath)

            srcmask_path = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[0][0], index[0][0]+".png")
            trgmask_path = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[2][0], index[2][0]+".png")
            trg2mask_path = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[-1][0], index[-1][0]+".png")
            save_mask_to_img(srcmask[0].cpu().detach().numpy(), srcmask_path)
            save_mask_to_img(trgmask[0].cpu().detach().numpy(), trgmask_path)
            save_mask_to_img(trg2mask[0].cpu().detach().numpy(), trg2mask_path)
            
            input_cat_1 = torch.cat([srcmask, trgmask], 1)
            input_cat_2 = torch.cat([trgmask, trg2mask], 1)
            outputs_1 = model(input_cat_1)
            outputs_2 = model(input_cat_2)
            # push values to be between 0 and 1
            outputs_1 = torch.sigmoid(outputs_1)
            outputs_2 = torch.sigmoid(outputs_2)
            output_answer_1 = torch.where(outputs_1 > 0.5, 1., 0.)
            output_answer_2 = torch.where(outputs_2 > 0.5, 1., 0.)

            loss = F.binary_cross_entropy(outputs_1, ibmask, weight=loss_weights) + \
                                F.binary_cross_entropy(outputs_2, ibmask2, weight=loss_weights_2)
            running_loss += loss.detach().item()
            running_psnr += psnr(outputs_1.detach(), ibmask.detach())
            running_psnr += psnr(outputs_2.detach(), ibmask2.detach())
            running_ssim += ssim(outputs_1.detach(), ibmask.detach())
            running_ssim += ssim(outputs_2.detach(), ibmask2.detach())

            input_outfile = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_mask.png")
            output_outfile = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_est_mask.png")
            output_grayscale_outfile = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_grayscale.png")
            
            input2_outfile = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[3][0], index[3][0]+"_mask.png")
            output2_outfile = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[3][0], index[3][0]+"_est_mask.png")
            output2_grayscale_outfile = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[3][0], index[3][0]+"_grayscale.png")
            
            output_np = output_answer_1.cpu().detach().numpy()
            target_np = ibmask.cpu().detach().numpy()
            output_not_binarized_np = outputs_1.cpu().detach().numpy()
            output_np_2 = output_answer_2.cpu().detach().numpy()
            target_np_2 = ibmask2.cpu().detach().numpy()
            output_not_binarized_np_2 = outputs_2.cpu().detach().numpy()
        
            save_mask_to_img(output_np[0], output_outfile)
            save_mask_to_img(target_np[0], input_outfile)
            save_mask_to_img(output_not_binarized_np[0], output_grayscale_outfile)
            save_mask_to_img(output_np_2[0], output2_outfile)
            save_mask_to_img(target_np_2[0], input2_outfile)
            save_mask_to_img(output_not_binarized_np_2[0], output2_grayscale_outfile)

        cur_loss = running_loss / len(testloader)
        cur_psnr = running_psnr / len(testloader)
        cur_ssim = running_ssim / len(testloader)

        print("epoch", epoch, "loss", cur_loss)
        test_loss_arr.append(cur_loss)

    return test_loss_arr, cur_psnr, cur_ssim, model



def main(config, args):
    print("tae is goin for it on ig")

    if config.dataset == "atd12k":
        trainset = datas.AniTripletWithSGMFlow(config.trainset_root, \
                                                config.train_flow_root, config.hist_mask_root, \
                                                config.dataset_root_filepath_train, \
                                                dt=config.dt, raft=config.raft, \
                                                smooth=config.smooth, \
                                                img_size=config.test_size, \
                                                discrim_crop_size=config.discrim_crop_size, \
                                                patch_location=config.patch_location, \
                                                random_flip=config.random_flip, \
                                                random_reverse=config.random_reverse, \
                                                highmag_flow=config.highmag_flow)
        testset = datas.AniTripletWithSGMFlowTest(config.testset_root, \
                                                config.test_flow_root, config.hist_mask_test_root, \
                                                config.dataset_root_filepath_test, \
                                                dt=config.dt, raft=config.raft, \
                                                smooth=config.smooth, \
                                                img_size=config.test_size, \
                                                resize=config.test_resize)
    elif config.dataset == "blender_cubes" or config.dataset == "SU" or config.dataset == "suzanne":
        trainset = datas.BlenderAniTripletPatch(args, config.trainset_root, \
                                                config.dataset_root_filepath_train, \
                                                config.num_ib_frames, \
                                                img_size=config.test_size, \
                                                flow_type=config.flow_type, \
                                                flow_root=config.trainflow_root, \
                                                random_reverse=config.random_reverse, \
                                                overfit=config.overfit, \
                                                small_dataset=config.small_dataset, \
                                                dt=config.dt)
        testset = datas.BlenderAniTripletPatchTest(args, config.testset_root, \
                                                   config.csv_root, \
                                                   config.num_ib_frames, \
                                                   config.dataset_root_filepath_test, \
                                                   config.dataset, \
                                                   img_size=config.test_size, \
                                                   resize=config.test_resize, \
                                                   flow_type=config.flow_type, \
                                                   flow_root=config.testflow_root, \
                                                   small_dataset=config.small_dataset, \
                                                   dt=config.dt)
    elif config.dataset == "suzanne_exr":
        trainset = datas.CSVEXRQuintlet(config.trainset_root, config.num_ib_frames, \
                                       img_size=config.test_size, patch_size=config.patch_size, \
                                       discrim_crop_size=None, random_flip=False, \
                                       random_reverse=config.random_reverse, dt=config.dt, \
                                       patch_location=None, overfit=config.overfit)
        testset = datas.CSVEXRQuintletTest(config.testset_root, config.num_ib_frames, \
                                       img_size=config.test_size, patch_size=config.patch_size, \
                                       discrim_crop_size=None, random_flip=False, \
                                       random_reverse=config.random_reverse, dt=config.dt, \
                                       patch_location=None, overfit=config.overfit)
    else:
        print("something went wrong, check your dataset name")
        return
    sampler = torch.utils.data.RandomSampler(trainset)
    test_sampler = torch.utils.data.SequentialSampler(testset)
    trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    testloader = torch.utils.data.DataLoader(testset, sampler=test_sampler, batch_size=config.test_batch_size, shuffle=False, num_workers=config.num_workers)
    to_img = transforms.ToPILImage()
 
    sys.stdout.flush()

    # create the generator
    if config.model == "UNet_RRDB":
        model = Unet_RRDB(
            encoder_name="rrdb",
            kernel_size=1,
            in_channels=10,
            classes=1,
        )

        if config.checkpoint_in != None:
            dict1 = torch.load(config.checkpoint_in)
            newdict = OrderedDict()
            for key, value in dict1.items():
                name = key[7:]
                newdict[name] = value
            model.load_state_dict(newdict, strict=True)
    elif config.model == "UNet":
        model = Unet(
            encoder_name="resnet34",
            # encoder_weights="imagenet",
            encoder_weights=None,
            encoder_depth=6,
            decoder_channels=(512, 256, 128, 64, 32, 16),
            # in_channels=6,
            in_channels=2,
            classes=1
        )
    else: #original model
        print("you're trying to use a different model? go back to ye olde script")
        return
    
    model = nn.DataParallel(model)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs-1)

    # Create the Discriminator
    if config.discrim == "patch":
        netD = Discriminator_patch().cuda()
    else:
        netD = Discriminator_square().cuda()
    netD = nn.DataParallel(netD)
    netD = netD.cuda()
    netD.apply(dcgan_weights_init)
    optimizerD = optim.Adam(netD.parameters(), lr = config.lr_d, betas=(config.beta1, 0.999))
    criterionD = nn.BCEWithLogitsLoss()

    print('Everything prepared. Ready to train...')
    sys.stdout.flush()

    epoch_arr = []
    loss_arr = []
    loss_d_arr = []

    test_epoch_arr = []
    test_loss_arr = []

    if config.checkpoint_in != None:
        epoch_arr = np.load(os.path.join(config.metrics_dir, "epoch_arr.npy"))
        epoch_arr = list(epoch_arr)
        loss_arr = np.load(os.path.join(config.metrics_dir, "loss_arr.npy"))
        loss_arr = list(loss_arr)
        loss_d_arr = np.load(os.path.join(config.metrics_dir, "loss_d_arr.npy"))
        loss_d_arr = list(loss_d_arr)
        test_epoch_arr = np.load(os.path.join(config.metrics_dir, "test_epoch_arr.npy"))
        test_epoch_arr = list(test_epoch_arr)
        test_loss_arr = np.load(os.path.join(config.metrics_dir, "test_loss_arr.npy"))
        test_loss_arr = list(test_loss_arr)

    for epoch in range(config.cur_epoch, config.num_epochs):
        print("######### EPOCH", epoch, "##########")
        epoch_arr.append(epoch)
        print("ilu yoongi")
        loss_arr, \
            loss_d_arr, \
            model = training_loop(epoch, model, trainloader, trainset,\
                                    optimizer, scheduler, \
                                    config, loss_arr, loss_d_arr,\
                                    to_img, netD, optimizerD, criterionD)
        if epoch % 10 == 0:
            test_epoch_arr.append(epoch)
            test_loss_arr, \
                cur_psnr, \
                cur_ssim, \
                model = testing_loop(epoch, model, testloader, testset, config, \
                                        to_img, test_loss_arr)
            save_psnr_ssim_to_txt(config, cur_psnr, cur_ssim, epoch)

        if not os.path.exists(config.checkpoint_latest_dir):
            os.makedirs(config.checkpoint_latest_dir)
        checkpoint_path = os.path.join(config.checkpoint_latest_dir, config.checkpoint_latest_file+str(epoch)+".pth")
        torch.save(model.state_dict(), checkpoint_path)
        
        plot_loss(config, epoch_arr, loss_arr, loss_d_arr, test_epoch_arr, test_loss_arr)
        save_metrics_to_arr(config, epoch_arr, loss_arr, loss_d_arr,\
                            test_epoch_arr, test_loss_arr)

    return loss_arr, loss_d_arr


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

    loss_arr, loss_d_arr = main(config, args)

    print("")
    print("######### RESULTS ###########")
    print("LOSS:", loss_arr)
    print("DISCRIMINATOR LOSS:", loss_d_arr)
    print("#############################")
    print("\a"*2)

