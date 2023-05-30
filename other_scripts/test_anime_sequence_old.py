import models
import datas
import argparse
import torch
import torchvision
import torchvision.transforms as TF
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import re
from math import log10
import numpy as np
import datetime
from utils.config import Config
import sys
import cv2
from utils.vis_flow import flow_to_color
import json
# import skimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import precision_score, accuracy_score, recall_score
from PIL import Image
import matplotlib.pyplot as plt
from models.segmentation_models_pytorch.unet import Unet
from models.discriminator_model.discriminator import Discriminator, dcgan_weights_init
# import segmentation_models_pytorch as smp

import pdb


def save_flow_to_img(flow, des):
    # pdb.set_trace()
    f = flow[0].data.cpu().numpy().transpose([1, 2, 0])
    fcopy = f.copy()
    fcopy[:, :, 0] = f[:, :, 1]
    fcopy[:, :, 1] = f[:, :, 0]
    cf = flow_to_color(-fcopy)
    cv2.imwrite(des + '.jpg', cf)


def save_mask_to_img(mask, name):
    image_np = 255*np.ones((mask.shape[0], mask.shape[1], 3))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 0:
                image_np[i][j] = [0,0,0]
    image = Image.fromarray(image_np.astype("uint8"))
    image.save(name)


def testing_loop(model, testloader, testset, config, \
                    to_img, revNormalize, revtrans,\
                    test_loss_arr, test_psnr_arr, \
                    test_ssim_arr, test_ie_arr,
                    test_acc_arr, test_precision_arr, test_recall_arr):
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    running_ie = 0.0
    running_acc = 0.0
    running_precision = 0.0
    running_recall = 0.0

    #  start testing...
    with torch.no_grad():
        model.eval()
        ii = 0
        # pdb.set_trace()
        for validationIndex, validationData in enumerate(testloader, 0):
            print('Testing {}/{}-th group...'.format(validationIndex, len(testset)))
            sys.stdout.flush()
            sample, flow, index, folder, masks = validationData

            frame1 = sample[0]
            frame2 = sample[-1]

            frameib = sample[1]

            ibmask = masks[1].cuda().float()
            num_bg = np.sum(np.where(ibmask.cpu().numpy()==0, 1, 0))
            num_fg = np.sum(np.where(ibmask.cpu().numpy()==1, 1, 0))
            multiplier = 1
            pos_weight_in = torch.FloatTensor([multiplier * num_bg/num_fg]).cuda()

            # folders.append(folder[0][0])
            
            # initial SGM flow
            F12i, F21i  = flow

            F12i = F12i.float().cuda() 
            F21i = F21i.float().cuda()

            # ITs = [sample[tt] for tt in range(1, 2)]
            I1 = frame1.cuda()
            I2 = frame2.cuda()
            Iib = frameib.cuda()
            
            if not os.path.exists(config.test_store_path + '/' + folder[0][0]+"_TEST"):
                os.makedirs(config.test_store_path + '/' + folder[0][0]+"_TEST")

            # pdb.set_trace()

            revtrans(I1.cpu()[0]).save(config.test_store_path + '/' + folder[0][0] + '_TEST/'  + index[0][0] + '.png')
            revtrans(I2.cpu()[0]).save(config.test_store_path + '/' + folder[-1][0] + '_TEST/' +  index[-1][0] + '.png')
            revtrans(Iib.cpu()[0]).save(config.test_store_path + '/' + folder[1][0] + '_TEST/' + index[1][0] + '_gt.png')
            
            for tt in range(config.inter_frames):
                x = config.inter_frames
                t = 1.0/(x+1) * (tt + 1)
                
                if config.model == "UNet":
                    if config.single_img_input:
                        outputs = model(Iib)
                    else:
                        input_ = torch.cat([I1, I2, F12i, F21i], 1)
                        outputs = model(input_)
                    It_warp = outputs
                    output_answer = outputs.sigmoid()
                    output_answer = torch.where(output_answer > 0.5, 1., 0.)
                    input_outfile = os.path.join(config.test_store_path, folder[1][0]+"_TEST", index[1][0]+"_mask.png")
                    output_outfile = os.path.join(config.test_store_path, folder[1][0]+"_TEST", index[1][0]+"_est_mask.png")
                    save_flow_to_img(F12i.cpu(), config.test_store_path + '/' + folder[1][0] + '_TEST/' + index[1][0] + '_F12')
                    save_flow_to_img(F21i.cpu(), config.test_store_path + '/' + folder[1][0] + '_TEST/' + index[1][0] + '_F21')
                    output_np = output_answer.data.cpu().numpy()
                    target_np = ibmask.data.cpu().numpy()

                    save_mask_to_img(output_np[0][0], output_outfile)
                    save_mask_to_img(target_np[0], input_outfile)

                    target_np_flat = target_np.flatten().astype(int)
                    output_np_flat = output_np.flatten().astype(int)

                    cur_acc = accuracy_score(target_np_flat, output_np_flat)
                    cur_precision = precision_score(target_np_flat, output_np_flat)
                    cur_recall = recall_score(target_np_flat, output_np_flat)
                else:
                    outputs = model(I1, I2, F12i, F21i, t)
                    It_warp = outputs[0].sigmoid()
                    save_flow_to_img(outputs[1].cpu(), config.test_store_path + '/' + folder[1][0] + '_TEST/' + index[1][0] + '_F12')
                    save_flow_to_img(outputs[2].cpu(), config.test_store_path + '/' + folder[1][0] + '_TEST/' + index[1][0] + '_F21')
                    save_mask_to_img(ibmask.data.cpu().numpy()[0], os.path.join(config.test_store_path, folder[1][0]+"_TEST", index[1][0]+"_mask.png"))
                    save_mask_to_img(It_warp.cpu()[0][0], \
                                config.test_store_path + '/' + folder[1][0] + '_TEST/' + index[1][0] + '_est_mask.png')

                # to_img(revNormalize(It_warp.cpu()[0]).clamp(0.0, 1.0)).save(config.test_store_path + '/' + folder[1][0] + '/' + index[1][0] + '_est.png')
                
                # estimated = revNormalize(It_warp[0].cpu()).clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
                # estimated = It_warp[0].cpu().numpy().transpose(1, 2, 0)
                # gt = revNormalize(Iib[0].cpu()).clamp(0.0, 1.0).numpy().transpose(1, 2, 0) 
                
                if config.model == "UNet":
                    # loss = criterion(It_warp.squeeze(dim=1), ibmask)
                    loss = F.binary_cross_entropy_with_logits(It_warp.squeeze(dim=1), ibmask, \
                        pos_weight=pos_weight_in)
                else:
                    ibmask_flatten = torch.flatten(ibmask, start_dim=1)
                    ibmask_negative = 1 - ibmask_flatten
                    ibmask_stack = torch.stack([ibmask_flatten, ibmask_negative], dim=-1)
                    It_warp_class_flatten = torch.flatten(It_warp, start_dim=1)
                    It_warp_class_negative = 1 - It_warp_class_flatten
                    It_warp_class_stack = torch.stack([It_warp_class_flatten, It_warp_class_negative], dim=-1)

                    loss = criterion(It_warp_class_stack, ibmask_stack)

                running_loss += loss.item()
                running_acc += cur_acc
                running_precision += cur_precision
                running_recall += cur_recall

                # labelFilePath = os.path.join(config.test_annotation_root,
                #                             folder[1][0], '%s.json'%folder[1][0])
                
                # crop region of interest
                # with open(labelFilePath, 'r') as f:
                #     jsonObj = json.load(f)
                #     motion_RoI = jsonObj["motion_RoI"]
                #     level = jsonObj["level"]

                # tempSize = jsonObj["image_size"]
                # scaleH = float(tempSize[1])/config.test_size[1]
                # scaleW = float(tempSize[0])/config.test_size[0]

                # RoI_x = int(jsonObj["motion_RoI"]['x'] // scaleW)
                # RoI_y = int(jsonObj["motion_RoI"]['y'] // scaleH)
                # RoI_W = int(jsonObj["motion_RoI"]['width'] // scaleW)
                # RoI_H = int(jsonObj["motion_RoI"]['height'] // scaleH)

                # print('RoI: %f, %f, %f, %f'%(RoI_x,RoI_y,RoI_W,RoI_H))

                # estimated_roi = estimated[RoI_y:RoI_y+RoI_H, RoI_x:RoI_x+RoI_W, :]
                # gt_roi = gt[RoI_y:RoI_y+RoI_H, RoI_x:RoI_x+RoI_W, :]

                # estimated_roi = estimated
                # gt_roi = gt

                # whole image value
                # this_psnr = skimage.metrics.peak_signal_noise_ratio(estimated, gt)
                # this_ssim = skimage.metrics.structural_similarity(estimated, gt, multichannel=True, gaussian=True)
                # this_psnr = peak_signal_noise_ratio(estimated, gt)
                # this_ssim = structural_similarity(estimated, gt, multichannel=True, gaussian=True)
                # this_ie = np.mean(np.sqrt(np.sum((estimated*255 - gt*255)**2, axis=2)))

                # running_psnr += this_psnr
                # running_ssim += this_ssim
                # running_ie += this_ie

                outputs = None

                # value for difficulty levels
                # psnrs_level[diff[level]] += this_psnr
                # ssims_level[diff[level]] += this_ssim
                # num_level[diff[level]] += 1

                # roi image value
                # this_roi_psnr = peak_signal_noise_ratio(estimated_roi, gt_roi)
                # this_roi_ssim = structural_similarity(estimated_roi, gt_roi, multichannel=True, gaussian=True)
                
                # psnr_roi += this_roi_psnr
                # ssim_roi += this_roi_ssim

        cur_loss = running_loss / (len(testloader) * config.inter_frames)
        # cur_psnr = running_psnr / (len(testloader) * config.inter_frames)
        # cur_ssim = running_ssim / (len(testloader) * config.inter_frames)
        # cur_ie = running_ie / (len(testloader) * config.inter_frames)
        cur_final_acc = running_acc / (len(testloader) * config.inter_frames)
        cur_final_precision = running_precision / (len(testloader) * config.inter_frames)
        cur_final_recall = running_recall / (len(testloader) * config.inter_frames)

        print("loss", cur_loss)
        test_loss_arr.append(cur_loss)
        # test_psnr_arr.append(cur_psnr)
        # test_ssim_arr.append(cur_ssim)
        # test_ie_arr.append(cur_ie)
        test_acc_arr.append(cur_final_acc)
        test_precision_arr.append(cur_final_precision)
        test_recall_arr.append(cur_final_recall)
    
    return test_loss_arr, test_psnr_arr, test_ssim_arr, test_ie_arr, model, test_acc_arr, test_precision_arr, test_recall_arr




def main(config):
    if not config.testing:
        print("NOPE")
        return
    # preparing datasets and normalization
    normalize1 = TF.Normalize(config.mean, [1.0, 1.0, 1.0])
    normalize2 = TF.Normalize([0, 0, 0], config.std)
    trans = TF.Compose([TF.ToTensor(), normalize1, normalize2, ])

    revmean = [-x for x in config.mean]
    revstd = [1.0 / x for x in config.std]
    revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
    revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
    revNormalize = TF.Compose([revnormalize1, revnormalize2])

    revtrans = TF.Compose([revnormalize1, revnormalize2, TF.ToPILImage()])

    testset = datas.CSVAniTripletWithSGMFlowTest(config.testset_root, config.csv_root, \
                                              config.num_ib_frames, config.data_source, \
                                              config.test_flow_root, config.hist_mask_root, \
                                              trans, config.test_size, \
                                              config.test_crop_size, train=False)
    # testset = datas.AniTripletWithSGMFlowTest(config.testset_root, \
    #                                           config.test_flow_root, config.hist_mask_root, \
    #                                           trans, config.test_size, \
    #                                           config.discrim_crop_size, train=False)
    test_sampler = torch.utils.data.SequentialSampler(testset)
    testloader = torch.utils.data.DataLoader(testset, sampler=test_sampler, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    to_img = TF.ToPILImage()
 
    sys.stdout.flush()

    # prepare model
    if config.model == "UNet":
        if config.single_img_input:
            model = Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1
            )
        else:
            model = Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=10,
            classes=1
            )
        model = model.cuda()

        ####### TAKE THIS OUT LATER????
        # dict1 = torch.load(config.checkpoint)
        # model.load_state_dict(dict1, strict=False)
    else:
        model = getattr(models, config.model)(config.pwc_path).cuda()
    
    #load saved model
    dict1 = torch.load(config.checkpoint_load)
    dict2 = {}
    if 'model_state_dict' in dict1.keys():
        model.load_state_dict(dict1['model_state_dict'], strict=False)
    else:
        #bc of parallelization, all parts have a "module.blah" format;
        #this is to get rid of that "module." in front
        regex = "^module\.(.*)$"
        for key in dict1.keys():
            name = re.search(regex, key).group(1)
            dict2[name] = dict1[key]
        model.load_state_dict(dict2, strict=True)
    model = nn.DataParallel(model)

    retImg = []

    ## values for whole image
    psnr_whole = 0
    psnrs = np.zeros([len(testset), config.inter_frames])
    ssim_whole = 0
    ssims = np.zeros([len(testset), config.inter_frames])
    ie_whole = 0
    ies = np.zeros([len(testset), config.inter_frames])

    ## values for ROI
    psnr_roi = 0
    ssim_roi = 0
    
    ## values for different levels
    psnrs_level = {'easy':0, 'mid': 0, 'hard':0}
    ssims_level = {'easy':0, 'mid': 0, 'hard':0}
    num_level = {'easy':0, 'mid': 0, 'hard':0}

    ## difficulty level dict
    diff = {0:'easy', 1:'mid', 2:'hard'}

    print('Everything prepared. Ready to test...')
    sys.stdout.flush()

    test_epoch_arr = []
    test_loss_arr = []
    test_psnr_arr = []
    test_ssim_arr = []
    test_ie_arr = []
    test_acc_arr = []
    test_precision_arr = []
    test_recall_arr = []

    test_loss_arr, \
    test_psnr_arr, \
    ssim_arr, \
    ie_arr, \
    model, \
    test_acc_arr, \
    test_precision_arr, \
    test_recall_arr = testing_loop(model, testloader, testset, config, \
                    to_img, revNormalize, revtrans, \
                    test_loss_arr, test_psnr_arr, \
                    test_ssim_arr, test_ie_arr, \
                    test_acc_arr, test_precision_arr, test_recall_arr)
    
    return test_loss_arr, test_acc_arr, test_precision_arr, test_recall_arr


if __name__ == "__main__":

    # loading configures
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    config = Config.from_file(args.config)


    # if not os.path.exists(config.store_path):
    #     os.mkdir(config.store_path)



    # psnrs, ssims, ies, psnr, ssim, psnr_roi, ssim_roi, psnrs_level, ssims_level, folder = validate(config)
    loss_arr, acc_arr, precision_arr, recall_arr = main(config)

    print("")
    print("######### RESULTS ###########")
    print("LOSS:", loss_arr)
    print("ACCURACY:", acc_arr)
    print("PRECISION:", precision_arr)
    print("RECALL:", recall_arr)
    print("#############################")
    print("\a"*2)

    # for ii in range(config.inter_frames):
    #     print('PSNR of validation frame' + str(ii+1) + ' is {}'.format(np.mean(psnrs[:, ii])))

    # for ii in range(config.inter_frames):
    #     print('PSNR of validation frame' + str(ii+1) + ' is {}'.format(np.mean(ssims[:, ii])))
            
    # for ii in range(config.inter_frames):
    #     print('PSNR of validation frame' + str(ii+1) + ' is {}'.format(np.mean(ies[:, ii])))
            
    # print('Whole PSNR is {}'.format(psnr) )
    # print('Whole SSIM is {}'.format(ssim) )

    # print('ROI PSNR is {}'.format(psnr_roi) )
    # print('ROI SSIM is {}'.format(ssim_roi) )

    # print('PSNRs for difficulties are {}'.format(psnrs_level) )
    # print('SSIMs for difficulties are {}'.format(ssims_level) )

    # with open(config.store_path + '/psnr.txt', 'w') as f:
    #     for index in sorted(range(len(psnrs[:, 0])), key=lambda k: psnrs[k, 0]):
    #         f.write("{}\t{}\n".format(folder[index], psnrs[index, 0]))

    # with open(config.store_path + '/ssim.txt', 'w') as f:
    #     for index in sorted(range(len(ssims[:, 0])), key=lambda k: ssims[k, 0]):
    #         f.write("{}\t{}\n".format(folder[index], ssims[index, 0]))

    # with open(config.store_path + '/ie.txt', 'w') as f:
    #     for index in sorted(range(len(ies[:, 0])), key=lambda k: ies[k, 0]):
    #         f.write("{}\t{}\n".format(folder[index], ies[index, 0]))