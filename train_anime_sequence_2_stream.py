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
from torchmetrics.classification import BinaryPrecisionRecallCurve, BinaryAccuracy, \
                                        BinaryF1Score, BinaryRecall, BinaryPrecision, \
                                        BinaryAUROC
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
import preprocess
from pytorch3d.loss import chamfer_distance
from lpips_pytorch import lpips

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


def save_metrics_to_arr(config, epoch_arr, loss_arr, loss_d_arr,\
                        loss_1_arr, loss_2_arr, test_epoch_arr, test_loss_arr):
    if not os.path.exists(config.metrics_dir):
        os.makedirs(config.metrics_dir)
    epoch_arr_path = os.path.join(config.metrics_dir, "epoch_arr.npy")
    np.save(epoch_arr_path, epoch_arr)
    loss_arr_path = os.path.join(config.metrics_dir, "loss_arr.npy")
    np.save(loss_arr_path, loss_arr)
    loss_arr_path = os.path.join(config.metrics_dir, "loss_d_arr.npy")
    np.save(loss_arr_path, loss_d_arr)
    loss_1_path = os.path.join(config.metrics_dir, "loss_1_arr.npy")
    np.save(loss_arr_path, loss_1_arr)
    loss_2_path = os.path.join(config.metrics_dir, "loss_2_arr.npy")
    np.save(loss_arr_path, loss_2_arr)
    
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


def save_psnr_ssim_to_txt(config, cur_psnr, cur_ssim, cur_cham, cur_lpips, epoch):
    if not os.path.exists(config.metrics_dir):
        os.makedirs(config.metrics_dir)
    textpath = os.path.join(config.metrics_dir, "psnr_ssim_epoch_"+str(epoch)+".txt")
    with open(textpath, 'w') as f:
        f.write("psnr "+str(cur_psnr)+" ssim "+str(cur_ssim)+" cham "+str(cur_cham)+" lpips "+str(cur_lpips))


def save_classification_metrics_to_txt(config, cur_auroc, cur_accs, cur_precs, cur_recalls, cur_f1s, epoch):
    if not os.path.exists(config.metrics_dir):
        os.makedirs(config.metrics_dir)
    textpath = os.path.join(config.metrics_dir, "classification_metrics_epoch_"+str(epoch)+".txt")
    with open(textpath, 'w') as f:
        f.write("auroc "+str(cur_auroc)+"\n")
        f.write("accuracy "+str(cur_accs)+"\n")
        f.write("precision "+str(cur_precs)+"\n")
        f.write("recall "+str(cur_recalls)+"\n")
        f.write("f1 "+str(cur_f1s)+"\n")
    np.save(os.path.join(config.metrics_dir, "acc_arr_epoch_{}.npy".format(epoch)), cur_accs)
    np.save(os.path.join(config.metrics_dir, "prec_arr_epoch_{}.npy".format(epoch)), cur_precs)
    np.save(os.path.join(config.metrics_dir, "recall_arr_epoch_{}.npy".format(epoch)), cur_recalls)
    np.save(os.path.join(config.metrics_dir, "f1_arr_epoch_{}.npy".format(epoch)), cur_f1s)


def plot_roc_curve(config, cur_precs, cur_recalls, epoch):
    plt.plot(cur_recalls, cur_precs)
    plt.xlabel("recall")
    plt.ylabel("precision")
    if not os.path.exists(config.metrics_dir):
        os.makedirs(config.metrics_dir)
    figpath = os.path.join(config.metrics_dir, "roc_curve_epoch"+str(epoch)+".png")
    plt.savefig(figpath)
    plt.clf()


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


def training_loop(epoch, model, model_1, model_2, trainloader, \
                    trainset, optimizer, optimizer_1, optimizer_2, \
                    scheduler, scheduler_1, scheduler_2, \
                    config, loss_arr, loss_1_arr, loss_2_arr ,\
                    loss_d_arr, to_img, netD, \
                    optimizerD, criterionD):
    #  start training...
    model.train()
    running_loss = 0.0
    running_d_loss = 0.0
    running_1_loss = 0.0
    running_2_loss = 0.0

    real_label = 1.
    fake_label = 0.

    threshold = 0.5

    for trainIndex, trainData in enumerate(trainloader):
        if trainIndex % 10 == 0:
            print('Training {}/{}-th group...'.format(trainIndex, len(trainloader)))
        sys.stdout.flush()

        if config.dataset == "suzanne_exr":
            rgb_sample, masks, flow, folder, index = trainData
        elif config.dataset == "blender_cubes" or config.dataset == "SU" or config.dataset == "suzanne" \
                or re.search("pt", config.dataset) or config.dataset == "all" or config.dataset == "moving_gif":
            sample, rgb_sample, folder, index, masks, flow = trainData

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

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        optimizerD.zero_grad()
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
        
        #2 streams
        input_cat_1 = torch.cat([srcmask, F15], 1).to(memory_format=torch.channels_last)
        input_cat_2 = torch.cat([trgmask, F51], 1).to(memory_format=torch.channels_last)

        F13_output, decoder_output_1 = model_1(input_cat_1)
        if model_2 != None:
            F53_output, decoder_output_2 = model_2(input_cat_2)
        else:
            F53_output, decoder_output_2 = model_1(input_cat_2)

        decoder_output_1 = decoder_output_1.to(memory_format=torch.channels_last)
        decoder_output_2 = decoder_output_2.to(memory_format=torch.channels_last)

        loss_13 = F.mse_loss(F13_output, F13)
        loss_53 = F.mse_loss(F53_output, F53)
        loss_streams = loss_13 + loss_53

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

        if config.discrim == "patch":
            real_output1x1, real_output16x16, real_output70x70, real_output256x256 = netD(ibmask)
            real_outputs = [real_output1x1, real_output16x16, real_output70x70, real_output256x256]
            label_1x1 = torch.full(real_output1x1.shape, real_label, dtype=torch.float).cuda().to(memory_format=torch.channels_last)
            label_16x16 = torch.full(real_output16x16.shape, real_label, dtype=torch.float).cuda().to(memory_format=torch.channels_last)
            label_70x70 = torch.full(real_output70x70.shape, real_label, dtype=torch.float).cuda().to(memory_format=torch.channels_last)
            label_256x256 = torch.full(real_output256x256.shape, real_label, dtype=torch.float).cuda().to(memory_format=torch.channels_last)
            labels = [label_1x1, label_16x16, label_70x70, label_256x256]
            errD_real = 0.
            for i in range(len(labels)):
                if i == 0:
                    continue
                errD_real += criterionD(real_outputs[i].to(memory_format=torch.channels_last), labels[i])
        else:
            real_output = netD(ibmask)
            real_output = real_output.to(memory_format=torch.channels_last)
            label = torch.full(real_output.shape, real_label, dtype=torch.float).cuda().to(memory_format=torch.channels_last)
            errD_real = criterionD(real_output, label)
        
        errD_real.backward()

        if epoch % 10 == 0:
            # saves the input images in a folder
            srcmask_path = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[0][0], index[0][0]+".png")
            trgmask_path = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[-1][0], index[-1][0]+".png")
            save_mask_to_img(srcmask[0].cpu().detach().numpy(), srcmask_path)
            save_mask_to_img(trgmask[0].cpu().detach().numpy(), trgmask_path)
        
        input_cat = torch.cat([decoder_output_1, decoder_output_2], 1)
        input_cat = input_cat.to(memory_format=torch.channels_last)
        outputs, _ = model(input_cat)
        outputs = outputs.to(memory_format=torch.channels_last)
        output_answer = torch.where(outputs > threshold, 1., 0.)

        ## Train discriminator with all-fake batch
        # Classify all fake batch with D
        if config.discrim == "patch":
            # fake_output1x1, fake_output16x16, fake_output70x70, fake_output256x256 = netD(output_answer.detach())
            fake_output1x1, fake_output16x16, fake_output70x70, fake_output256x256 = netD(outputs.detach())
            fake_outputs = [fake_output1x1, fake_output16x16, fake_output70x70, fake_output256x256]
            errD_fake = 0.
            for i in range(len(fake_outputs)):
                if i == 0:
                    continue
                labels[i].fill_(fake_label)
                errD_fake += criterionD(fake_outputs[i].to(memory_format=torch.channels_last), labels[i])
        else:
            label.fill_(fake_label)
            # fake_output = netD(output_answer.detach())
            fake_output = netD(outputs.detach())
            fake_output = fake_output.to(memory_format=torch.channels_last)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterionD(fake_output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
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
            # fake_output_again1x1, fake_output_again16x16, \
            #     fake_output_again70x70, fake_output_again256x256 = netD(output_answer)
            fake_output_again1x1, fake_output_again16x16, \
                fake_output_again70x70, fake_output_again256x256 = netD(outputs)
            fake_outputs_again = [fake_output_again1x1, fake_output_again16x16, fake_output_again70x70, fake_output_again256x256]
            
            errG = 0.
            for i in range(len(fake_outputs_again)):
                if i == 0:
                    continue
                labels[i].fill_(real_label)
                errG += criterionD(fake_outputs_again[i].to(memory_format=torch.channels_last), labels[i])
        else:
            label.fill_(real_label) # fake labels are real for generator cost
            # fake_output_again = netD(output_answer)
            fake_output_again = netD(outputs)
            fake_output_again = fake_output_again.to(memory_format=torch.channels_last)
            # Calculate G's loss based on this output
            errG = criterionD(fake_output_again, label)

        # calculate loss(es)
        loss = 0.0
        if config.recon_loss:
            if config.mask_loss:
                loss += torch.mean(torch.mul(F.binary_cross_entropy_with_logits(outputs, ibmask), loss_weights))
            elif config.l1_loss:
                loss += torch.mean(torch.mul(F.l1_loss(outputs, ibmask, reduction='none'), loss_weights))
            elif config.l2_loss:
                loss += torch.mean(torch.mul(F.mse_loss(outputs, ibmask, reduction='none'), loss_weights))
        if config.gan_loss:
            loss += config.gan_weight*errG
        if config.warp_loss and epoch > 10:
            img1_from_inb = backwarp(0.5*F12i, outputs)
            img2_from_inb = backwarp(0.5*F21i, outputs)

            img1_from_inb_np = 255*(1 - img1_from_inb[0].cpu().detach().numpy())
            img2_from_inb_np = 255*(1 - img2_from_inb[0].cpu().detach().numpy())
            warp_output_img1 = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[0][0]+"_warp_from_inb.png")
            warp_output_img2 = os.path.join(config.train_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[-1][0]+"_warp_from_inb.png")

            if epoch % 10 == 0:
                print("writing warp output")
                cv2.imwrite(warp_output_img1, np.transpose(img1_from_inb_np, (1, 2, 0)))
                cv2.imwrite(warp_output_img2, np.transpose(img2_from_inb_np, (1, 2, 0)))

            #weight bce loss by inverse distance from black pixel
            num_bg_src = np.sum(np.where(srcmask.cpu().detach().numpy()==0, 1, 0))
            num_fg_src = np.sum(np.where(srcmask.cpu().detach().numpy()==1, 1, 0))
            p_bg_src = num_bg_src/(num_bg_src+num_fg_src)
            loss_weights_src = (1-srcmask)*(2*p_bg_src - 1) + (1 - p_bg_src)

            num_bg_trg = np.sum(np.where(trgmask.cpu().detach().numpy()==0, 1, 0))
            num_fg_trg = np.sum(np.where(trgmask.cpu().detach().numpy()==1, 1, 0))
            p_bg_trg = num_bg_trg/(num_bg_trg+num_fg_trg)
            loss_weights_trg = (1-trgmask)*(2*p_bg_trg - 1) + (1 - p_bg_trg)

            img1_from_inb_sig = torch.sigmoid(img1_from_inb)
            img2_from_inb_sig = torch.sigmoid(img2_from_inb)
            loss_img1 = F.binary_cross_entropy(img1_from_inb_sig, srcmask, weight=loss_weights_src)
            loss_img2 = F.binary_cross_entropy(img2_from_inb_sig, trgmask, weight=loss_weights_trg)
            loss += config.warp_weight*(loss_img1 + loss_img2)

        loss += loss_streams
        loss.backward()
        optimizer.step()
        optimizer_1.step()
        if optimizer_2 != None:
            optimizer_2.step()
        running_loss += loss.detach().item()

        running_1_loss += loss_13.item()
        running_2_loss += loss_53.item()

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
    scheduler_1.step()
    if scheduler_2 != None:
        scheduler_2.step()
    cur_loss = running_loss / (len(trainloader))
    cur_d_loss = running_d_loss / (len(trainloader))
    cur_1_loss = running_1_loss / (len(trainloader))
    cur_2_loss = running_2_loss / (len(trainloader))

    print("epoch", epoch, "loss", cur_loss, "discrim loss", cur_d_loss)
    loss_arr.append(cur_loss)
    loss_d_arr.append(cur_d_loss)
    loss_1_arr.append(cur_1_loss)
    loss_2_arr.append(cur_2_loss)

    return loss_arr, loss_d_arr, loss_1_arr, loss_2_arr, model, model_1, model_2


def testing_loop(epoch, model, model_1, model_2, testloader, testset, config, \
                    to_img, test_loss_arr):    
    default_threshold = 0.5
    thresholds_1 = np.arange(10)*0.1
    thresholds_2 = (np.arange(10)*0.01)+0.9
    thresholds = np.concatenate((thresholds_1, thresholds_2))
    
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    running_cham = 0.0
    running_lpips = 0.0
    running_auroc = 0.0
    running_accuracies = np.zeros(len(thresholds))
    running_precisions = np.zeros(len(thresholds))
    running_recalls = np.zeros(len(thresholds))
    running_f1s = np.zeros(len(thresholds))

    psnr = PSNR().cuda()
    ssim = SSIM(n_channels=1).cuda()
    auroc = BinaryAUROC(thresholds=torch.Tensor(thresholds))

    #  start testing...
    with torch.no_grad():
        model.eval()

        for validationIndex, validationData in enumerate(testloader):
            print('Testing {}/{}-th group...'.format(validationIndex, len(testloader)))
            sys.stdout.flush()

            if config.dataset == "suzanne_exr":
                rgb_sample, masks, flow, folder, index = validationData
            elif config.dataset == "blender_cubes" or config.dataset == "SU" or config.dataset == "suzanne" \
                    or re.search("pt", config.dataset) or config.dataset == "all":
                sample, rgb_sample, folder, index, masks, flow = validationData

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

            #2 streams
            input_cat_1 = torch.cat([srcmask, F15], 1)
            input_cat_2 = torch.cat([trgmask, F51], 1)
            input_cat_1 = input_cat_1.to(memory_format=torch.channels_last)
            input_cat_2 = input_cat_2.to(memory_format=torch.channels_last)

            F13_output, decoder_output_1 = model_1(input_cat_1)
            F53_output, decoder_output_2 = model_2(input_cat_2)
            decoder_output_1 = decoder_output_1.to(memory_format=torch.channels_last)
            decoder_output_2 = decoder_output_2.to(memory_format=torch.channels_last)

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
            
            input_cat = torch.cat([decoder_output_1, decoder_output_2], 1)
            outputs, _ = model(input_cat)
            outputs = outputs.to(memory_format=torch.channels_last)
            # outputs = torch.sigmoid(outputs)
            output_answer = torch.where(outputs > default_threshold, 1., 0.)

            loss = torch.mean(torch.mul(F.binary_cross_entropy_with_logits(outputs, ibmask), loss_weights))
            running_loss += loss.detach().item()
            running_psnr += psnr(output_answer.detach(), ibmask.detach())
            running_ssim += ssim(output_answer.detach(), ibmask.detach())
            running_cham += chamfer_distance(output_answer.detach()[0], ibmask.detach()[0])[0].detach().cpu().item()
            running_lpips += lpips(output_answer, ibmask, net_type='vgg', version='0.1').detach().cpu().squeeze().item()
            running_auroc += auroc(outputs.cpu().detach(), ibmask.cpu().detach().int()).detach().item()

            for i, threshold_in in enumerate(thresholds):
                acc_metric = BinaryAccuracy(threshold=threshold_in)
                prec_metric = BinaryPrecision(threshold=threshold_in)
                recall_metric = BinaryRecall(threshold=threshold_in)
                f1_metric = BinaryF1Score(threshold=threshold_in)

                running_accuracies[i] += acc_metric(outputs.cpu().detach(), ibmask.cpu().detach()).item()
                running_precisions[i] += prec_metric(outputs.cpu().detach(), ibmask.cpu().detach()).item()
                running_recalls[i] += recall_metric(outputs.cpu().detach(), ibmask.cpu().detach()).item()
                running_f1s[i] += f1_metric(outputs.cpu().detach(), ibmask.cpu().detach()).item()

            input_outfile = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_mask.png")
            output_outfile = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_est_mask.png")
            output_grayscale_outfile = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[1][0]+"_grayscale.png")
            output_np = output_answer.data.cpu().detach().numpy()
            target_np = ibmask.cpu().detach().numpy()
            output_not_binarized_np = outputs.cpu().detach().numpy()

            save_mask_to_img(output_np[0], output_outfile)
            save_mask_to_img(target_np[0], input_outfile)
            save_mask_to_img(output_not_binarized_np[0], output_grayscale_outfile)

            if config.warp_loss and epoch > 10:
                img1_from_inb = backwarp(0.5*F12i, output_answer)
                img2_from_inb = backwarp(0.5*F21i, output_answer)
                # img1_from_inb = torch.cat((img1_from_inb, img1_from_inb, img1_from_inb), dim=1)
                # img2_from_inb = torch.cat((img2_from_inb, img2_from_inb, img2_from_inb), dim=1)

                img1_from_inb_np = 255*(1 - img1_from_inb[0].cpu().detach().numpy())
                img2_from_inb_np = 255*(1 - img2_from_inb[0].cpu().detach().numpy())
                warp_output_img1 = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[0][0]+"_warp_from_inb.png")
                warp_output_img2 = os.path.join(config.test_store_path, "epoch_{:03d}".format(epoch), folder[1][0], index[-1][0]+"_warp_from_inb.png")

                print("writing test warp output")
                cv2.imwrite(warp_output_img1, np.transpose(img1_from_inb_np, (1, 2, 0)))
                cv2.imwrite(warp_output_img2, np.transpose(img2_from_inb_np, (1, 2, 0)))
                # save_progress_gif(config.test_store_path, epoch, folder[1][0])

        cur_loss = running_loss / len(testloader)
        cur_psnr = running_psnr / len(testloader)
        cur_ssim = running_ssim / len(testloader)
        cur_cham = running_cham / len(testloader)
        cur_lpips = running_lpips / len(testloader)
        cur_auroc = running_auroc / len(testloader)

        cur_accs = running_accuracies / len(testloader)
        cur_precs = running_precisions / len(testloader)
        cur_recalls = running_recalls / len(testloader)
        cur_f1s = running_f1s / len(testloader)

        print("epoch", epoch, "loss", cur_loss)
        test_loss_arr.append(cur_loss)

    return test_loss_arr, cur_psnr, cur_ssim, cur_cham, cur_lpips, cur_auroc, \
            cur_accs, cur_precs, cur_recalls, cur_f1s, model, model_1, model_2



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
    elif config.dataset == "blender_cubes" or config.dataset == "all" or config.dataset == "SU" \
                or config.dataset == "suzanne" or config.dataset == "moving_gif":
        trainset = datas.BlenderAniTripletPatch2(args, config.trainset_root, \
                                                config.num_ib_frames, \
                                                config.dataset, \
                                                img_size=config.test_size, \
                                                flow_type=config.flow_type, \
                                                flow_root=config.trainflow_root, \
                                                csv_root=config.csv_root, \
                                                random_reverse=config.random_reverse, \
                                                overfit=config.overfit, \
                                                small_dataset=config.small_dataset, \
                                                dt=config.dt, csv=config.csv)
        testset = datas.BlenderAniTripletPatchTest2(args, config.testset_root, \
                                                   config.csv_root, \
                                                   config.num_ib_frames, \
                                                   config.dataset, \
                                                   img_size=config.test_size, \
                                                   resize=config.test_resize, \
                                                   flow_type=config.flow_type, \
                                                   flow_root=config.testflow_root, \
                                                   small_dataset=config.small_dataset, \
                                                   dt=config.dt)
    elif config.dataset == "suzanne_exr":
        trainset = datas.CSVEXRTriplet(config.csv_root, config.trainset_root, config.flow_root, config.num_ib_frames, \
                                       train=True, img_size=config.test_size, patch_size=config.patch_size, \
                                       discrim_crop_size=None, random_flip=False, \
                                       random_reverse=config.random_reverse, dt=config.dt, \
                                       patch_location=None, overfit=config.overfit)
        testset = datas.CSVEXRTriplet(config.csv_root, config.testset_root, config.flow_root, config.num_ib_frames, \
                                       train=False, img_size=config.test_size, patch_size=config.patch_size, \
                                       discrim_crop_size=None, random_flip=False, \
                                       random_reverse=config.random_reverse, dt=config.dt, \
                                       patch_location=None, overfit=config.overfit)
    elif re.search("pt", config.dataset):
        trainset = datas.BlenderAniTripletPatch2(args, config.trainset_root, \
                                                config.num_ib_frames, \
                                                config.dataset, \
                                                csv_filename=config.csv_filename, \
                                                img_size=config.test_size, \
                                                flow_type=config.flow_type, \
                                                flow_root=config.trainflow_root, \
                                                csv_root=config.csv_root, \
                                                random_reverse=config.random_reverse, \
                                                overfit=config.overfit, \
                                                small_dataset=config.small_dataset, \
                                                dt=config.dt, csv=config.csv)
        testset = datas.BlenderAniTripletPatchTest2(args, config.testset_root, \
                                                   config.csv_root, \
                                                   config.num_ib_frames, \
                                                   config.dataset, \
                                                   csv_filename=config.csv_filename,\
                                                   img_size=config.test_size, \
                                                   resize=config.test_resize, \
                                                   flow_type=config.flow_type, \
                                                   flow_root=config.testflow_root, \
                                                   small_dataset=config.small_dataset, \
                                                   dt=config.dt)
    else:
        print("something went wrong, check your dataset name")
        return
    # sampler = torch.utils.data.RandomSampler(trainset)
    sampler = torch.utils.data.SequentialSampler(trainset)
    test_sampler = torch.utils.data.SequentialSampler(testset)
    trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    testloader = torch.utils.data.DataLoader(testset, sampler=test_sampler, batch_size=config.test_batch_size, shuffle=False, num_workers=config.num_workers)
    to_img = transforms.ToPILImage()
 
    sys.stdout.flush()

    # create the generator
    model_1 = Unet(
        encoder_name=config.stream_encoder_name,
        encoder_weights=config.stream_encoder_weights,
        encoder_depth=config.stream_encoder_depth,
        decoder_channels=config.stream_decoder_channels,
        in_channels=3,
        classes=2
    )
    if config.stream_share:
        model_2 = Unet(
            encoder_name=config.stream_encoder_name,
            encoder_weights=config.stream_encoder_weights,
            encoder_depth=config.stream_encoder_depth,
            decoder_channels=config.stream_decoder_channels,
            in_channels=3,
            classes=2
        )
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

        if not config.stream_share: 
            dict3 = torch.load(config.checkpoint_in_2)
            newdict3 = OrderedDict()
            for key, value in dict3.items():
                name = key[7:]
                newdict3[name] = value
            model_2.load_state_dict(newdict3, strict=True)
        else:
            model_2.load_state_dict(newdict2, strict=True)
    
    model = nn.DataParallel(model)
    model = model.cuda().to(memory_format=torch.channels_last)
    model_1 = nn.DataParallel(model_1)
    model_1 = model_1.cuda().to(memory_format=torch.channels_last)
    # if stream_share:
    model_2 = nn.DataParallel(model_2)
    model_2 = model_2.cuda().to(memory_format=torch.channels_last)
    # else:
    #     model_2 = None
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    scheduler = optim.lr_scheduler.SequentialLR(
                                            optimizer,
                                            [
                                                optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=200),
                                                optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200),
                                            ],
                                            milestones=[200],
                                        )
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs-1)
    optimizer_1 = optim.Adam(model_1.parameters(), lr=config.lr_stream, betas=(config.beta1, 0.999))
    scheduler_1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=config.num_epochs-1)
    # if stream_share:
    optimizer_2 = optim.Adam(model_2.parameters(), lr=config.lr_stream, betas=(config.beta1, 0.999))
    scheduler_2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=config.num_epochs-1)
    # else:
    #     optimizer_2 = None
    #     scheduler_2 = None

    # Create the Discriminator
    if config.discrim == "patch":
        netD = Discriminator_patch().cuda()
    else:
        netD = Discriminator_square().cuda()
    netD = nn.DataParallel(netD)
    netD = netD.cuda().to(memory_format=torch.channels_last)
    netD.apply(dcgan_weights_init)
    optimizerD = optim.Adam(netD.parameters(), lr = config.lr_d, betas=(config.beta1, 0.999))
    criterionD = nn.BCEWithLogitsLoss()

    print('Everything prepared. Ready to train...')
    sys.stdout.flush()

    epoch_arr = []
    loss_arr = []
    loss_d_arr = []
    loss_1_arr = []
    loss_2_arr = []

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
            loss_1_arr, \
            loss_2_arr, \
            model, \
            model_1, \
            model_2 = training_loop(epoch, model, model_1, \
                                    model_2, trainloader, trainset,\
                                    optimizer, optimizer_1, optimizer_2,\
                                    scheduler, scheduler_1, scheduler_2, \
                                    config, loss_arr, loss_d_arr,\
                                    loss_1_arr, loss_2_arr, \
                                    to_img, netD, optimizerD, criterionD)
        if epoch % 10 == 0:
            test_epoch_arr.append(epoch)
            test_loss_arr, \
                cur_psnr, \
                cur_ssim, \
                cur_cham, \
                cur_lpips, \
                cur_auroc, \
                cur_accs, \
                cur_precs, \
                cur_recalls, \
                cur_f1s, \
                model, \
                model_1, \
                model_2 = testing_loop(epoch, model, model_1, \
                                        model_2, testloader, testset, config, \
                                        to_img, test_loss_arr)
            save_psnr_ssim_to_txt(config, cur_psnr, cur_ssim, cur_cham, cur_lpips, epoch)
            save_classification_metrics_to_txt(config, cur_auroc, cur_accs, cur_precs, \
                                                cur_recalls, cur_f1s, epoch)
            plot_roc_curve(config, cur_precs, cur_recalls, epoch)

            if not os.path.exists(config.checkpoint_latest_dir):
                os.makedirs(config.checkpoint_latest_dir)
            checkpoint_path = os.path.join(config.checkpoint_latest_dir, config.checkpoint_latest_file+str(epoch)+".pth")
            checkpoint_path_1 = os.path.join(config.checkpoint_latest_dir, config.checkpoint_latest_file+str(epoch)+"_1.pth")
            checkpoint_path_2 = os.path.join(config.checkpoint_latest_dir, config.checkpoint_latest_file+str(epoch)+"_2.pth")
            torch.save(model.state_dict(), checkpoint_path)
            torch.save(model_1.state_dict(), checkpoint_path_1)
            torch.save(model_2.state_dict(), checkpoint_path_2)
        
        # plot_loss(config, epoch_arr, loss_arr, loss_d_arr, loss_1_arr, loss_2_arr, test_epoch_arr, test_loss_arr)
        # save_metrics_to_arr(config, epoch_arr, loss_arr, loss_d_arr, loss_1_arr, loss_2_arr, \
        #                     test_epoch_arr, test_loss_arr)
        
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

