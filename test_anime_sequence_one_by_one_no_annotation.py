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
from sklearn.metrics import precision_score, accuracy_score, recall_score
from PIL import Image
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryPrecisionRecallCurve, BinaryAccuracy, \
                                        BinaryF1Score, BinaryRecall, BinaryPrecision, \
                                        BinaryAUROC
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


def save_flow_to_img(flow, des):
    f = flow[0].data.cpu().numpy().transpose([1, 2, 0])
    fcopy = f.copy()
    fcopy[:, :, 0] = f[:, :, 1]
    fcopy[:, :, 1] = f[:, :, 0]
    cf = flow_to_color(-fcopy)
    cv2.imwrite(des + '.jpg', cf)


def save_mask_to_img(mask, name):
    #note mask should be np array
    if mask.shape[0] == 1:
        mask = np.transpose(mask, (1, 2, 0))
    cv2.imwrite(name, (1 - mask)*255)


def save_psnr_ssim_to_txt(config, cur_psnr, cur_ssim, cur_cham, cur_lpips):
    if not os.path.exists(config.metrics_dir):
        os.makedirs(config.metrics_dir)
    textpath = os.path.join(config.metrics_dir, "psnr_ssim.txt")
    with open(textpath, 'w') as f:
        f.write("psnr "+str(cur_psnr)+" ssim "+str(cur_ssim)+" cham "+str(cur_cham)+" lpips "+str(cur_lpips))


def save_classification_metrics_to_txt(config, cur_auroc, cur_accs, cur_precs, cur_recalls, cur_f1s):
    if not os.path.exists(config.metrics_dir):
        os.makedirs(config.metrics_dir)
    textpath = os.path.join(config.metrics_dir, "classification_metrics.txt")
    with open(textpath, 'w') as f:
        f.write("auroc "+str(cur_auroc)+"\n")
        f.write("accuracy "+str(cur_accs)+"\n")
        f.write("precision "+str(cur_precs)+"\n")
        f.write("recall "+str(cur_recalls)+"\n")
        f.write("f1 "+str(cur_f1s)+"\n")
    np.save(os.path.join(config.metrics_dir, "acc_arr.npy"), cur_accs)
    np.save(os.path.join(config.metrics_dir, "prec_arr.npy"), cur_precs)
    np.save(os.path.join(config.metrics_dir, "recall_arr.npy"), cur_recalls)
    np.save(os.path.join(config.metrics_dir, "f1_arr.npy"), cur_f1s)


def plot_roc_curve(config, cur_precs, cur_recalls):
    plt.plot(cur_recalls, cur_precs)
    plt.xlabel("recall")
    plt.ylabel("precision")
    if not os.path.exists(config.metrics_dir):
        os.makedirs(config.metrics_dir)
    figpath = os.path.join(config.metrics_dir, "roc_curve.png")
    plt.savefig(figpath)
    plt.clf()


def validate(config):
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
    
    test_sampler = torch.utils.data.SequentialSampler(testset)
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
    model_1 = nn.DataParallel(model_1)
    model_1 = model_1.cuda().to(memory_format=torch.channels_last)
    model_2 = nn.DataParallel(model_2)
    model_2 = model_2.cuda().to(memory_format=torch.channels_last)

    print('Everything prepared. Ready to test...')
    sys.stdout.flush()   

    default_threshold = 0.5
    thresholds_1 = np.arange(10)*0.1
    thresholds_2 = (np.arange(10)*0.01)+0.9
    thresholds = np.concatenate((thresholds_1, thresholds_2)) 

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

    folders = []

    print('Everything prepared. Ready to test...')  
    sys.stdout.flush()

    #  start testing...
    with torch.no_grad():
        model.eval()
        # pdb.set_trace()
        for testIndex, testData in enumerate(testloader, 0):
            print('Testing {}/{}-th group...'.format(testIndex, len(testset)))
            sys.stdout.flush()

            if config.dataset == "suzanne_exr":
                rgb_sample, masks, flow, folder, index = testData
            elif config.dataset == "blender_cubes" or config.dataset == "SU" or config.dataset == "suzanne":
                sample, rgb_sample, folder, index, masks, flow = testData

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

            outpath = os.path.join(config.test_store_path, folder[0][0])
            if not os.path.exists(outpath):
                os.makedirs(outpath)

            srcmask_path = os.path.join(config.test_store_path, folder[0][0], index[0][0]+".png")
            trgmask_path = os.path.join(config.test_store_path, folder[-1][0], index[-1][0]+".png")
            save_mask_to_img(srcmask[0].cpu().detach().numpy(), srcmask_path)
            save_mask_to_img(trgmask[0].cpu().detach().numpy(), trgmask_path)
            if not config.flow_type == None:
                save_flow_to_img(F15.cpu(), os.path.join(config.test_store_path, folder[1][0], index[1][0]+"_F15"))
                save_flow_to_img(F51.cpu(), os.path.join(config.test_store_path, folder[1][0], index[1][0]+"_F51"))
                save_flow_to_img(F13.cpu(), os.path.join(config.test_store_path, folder[1][0], index[1][0]+"_F13"))
                save_flow_to_img(F53.cpu(), os.path.join(config.test_store_path, folder[1][0], index[1][0]+"_F53"))
                save_flow_to_img(F13_output.cpu(), os.path.join(config.test_store_path, folder[1][0], index[1][0]+"_F13_est"))
                save_flow_to_img(F53_output.cpu(), os.path.join(config.test_store_path, folder[1][0], index[1][0]+"_F53_est"))
            
            input_cat = torch.cat([decoder_output_1, decoder_output_2], 1)
            outputs, _ = model(input_cat)
            outputs = outputs.to(memory_format=torch.channels_last)
            # outputs = torch.sigmoid(outputs)
            output_answer = torch.where(outputs > 0.5, 1., 0.)

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

            input_outfile = os.path.join(config.test_store_path, folder[1][0], index[1][0]+"_mask.png")
            output_outfile = os.path.join(config.test_store_path, folder[1][0], index[1][0]+"_est_mask.png")
            output_grayscale_outfile = os.path.join(config.test_store_path, folder[1][0], index[1][0]+"_grayscale.png")
            output_np = output_answer.data.cpu().detach().numpy()
            target_np = ibmask.cpu().detach().numpy()
            output_not_binarized_np = outputs.cpu().detach().numpy()

            save_mask_to_img(output_np[0], output_outfile)
            save_mask_to_img(target_np[0], input_outfile)
            save_mask_to_img(output_not_binarized_np[0], output_grayscale_outfile)

    cur_psnr = running_psnr / len(testloader)
    cur_ssim = running_ssim / len(testloader)
    cur_cham = running_cham / len(testloader)
    cur_lpips = running_lpips / len(testloader)
    cur_auroc = running_auroc / len(testloader)

    cur_accs = running_accuracies / len(testloader)
    cur_precs = running_precisions / len(testloader)
    cur_recalls = running_recalls / len(testloader)
    cur_f1s = running_f1s / len(testloader)


    return cur_psnr, cur_ssim, cur_cham, cur_lpips, cur_auroc, \
            cur_accs, cur_precs, cur_recalls, cur_f1s


if __name__ == "__main__":

    # loading configures
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    config = Config.from_file(args.config)

    psnr, ssim, cham, lpips, auroc, \
        accs, precs, recalls, f1s = validate(config)

    save_psnr_ssim_to_txt(config, psnr, ssim, cham, lpips)
    save_classification_metrics_to_txt(config, auroc, \
                                    accs, precs, \
                                    recalls, f1s)

