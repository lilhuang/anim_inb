import numpy as lumpy
import matplotlib.pyplot as plt
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import datas
from PIL import Image
from utils.config import Config
from piqa import PSNR, SSIM
from models.segmentation_models_pytorch.unet import Unet
from math import log10
import cv2
from utils.vis_flow import flow_to_color
import json
import matplotlib.pyplot as plt

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
    image_np = 255*lumpy.ones((mask.shape[0], mask.shape[1], 3))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 0:
                image_np[i][j] = [0,0,0]
    image = Image.fromarray(image_np.astype("uint8"))
    image.save(name)


def save_metrics_to_arr(config, test_psnr_arr, test_ssim_arr, test_ie_arr):
    if not os.path.exists(config.metrics_dir):
        os.makedirs(config.metrics_dir)
    test_psnr_arr_path = os.path.join(config.metrics_dir, "test_psnr_arr.npy")
    lumpy.save(test_psnr_arr_path, test_psnr_arr)
    test_ssim_arr_path = os.path.join(config.metrics_dir, "test_ssim_arr.npy")
    lumpy.save(test_ssim_arr_path, test_ssim_arr)
    test_ie_arr_path = os.path.join(config.metrics_dir, "test_ie_arr.npy")
    lumpy.save(test_ie_arr_path, test_ie_arr)


def testing_loop(config, revtrans, testloader, testset, \
                    netG_seg, netG_arm, test_psnr_arr, \
                    test_ssim_arr, test_ie_arr):

    psnr = PSNR().cuda()
    ssim = SSIM(n_channels=1).cuda()

    with torch.no_grad():
        netG_seg.eval()
        netG_arm.eval()
        for validationIndex, validationData in enumerate(testloader, 0):
            print(validationIndex)
            print('Testing {}/{}-th group...'.format(validationIndex, len(testset)))
            sys.stdout.flush()
            sample, flow, index, folder, masks = validationData

            frame1 = sample[0]
            frame2 = sample[-2]
            frameib = sample[2]
            frame1_rgb = sample[1]
            frame2_rgb = sample[-1]
            frameib_rgb = sample[3]
            I1 = frame1.cuda()
            I2 = frame2.cuda()
            Iib = frameib.cuda()
            I1_rgb = frame1_rgb.cuda()
            I2_rgb = frame2_rgb.cuda()
            Iib_rgb = frameib_rgb.cuda()

            ibmask = masks[1].cuda().float()
            
            # initial SGM flow
            F12i, F21i  = flow

            F12i = F12i.float().cuda() 
            F21i = F21i.float().cuda()

            store_path = os.path.join(config.test_store_path, folder[0][0])
            if not os.path.exists(store_path):
                os.makedirs(store_path)

            revtrans(I1.cpu().detach()[0]).save(os.path.join(config.test_store_path, folder[0][0], index[0][0] + '.png'))
            revtrans(I2.cpu().detach()[0]).save(os.path.join(config.test_store_path, folder[-1][0], index[-1][0] + '.png'))
            revtrans(Iib.cpu().detach()[0]).save(os.path.join(config.test_store_path, folder[1][0], index[1][0] + '_gt.png'))
            
            #### Segmentation part #####
            # input_cat = torch.cat([I1, I2, F12i, F21i], 1)
            input_cat = torch.cat([I1_rgb, I2_rgb, F12i, F21i], 1)
            output_mask = netG_seg(input_cat)
            output_mask = output_mask.sigmoid()
            output_mask = torch.where(output_mask > 0.5, 1., 0.)
            input_outfile = os.path.join(config.test_store_path, folder[1][0], index[1][0]+"_mask.png")
            output_outfile = os.path.join(config.test_store_path, folder[1][0], index[1][0]+"_est_mask.png")
            output_np = output_mask.data.cpu().detach().numpy()
            target_np = ibmask.cpu().detach().numpy()

            save_mask_to_img(output_np[0][0], output_outfile)
            save_mask_to_img(target_np[0], input_outfile)

            save_flow_to_img(F12i.detach().cpu(), os.path.join(config.test_store_path, folder[1][0], index[1][0]+"_F12"))
            save_flow_to_img(F21i.detach().cpu(), os.path.join(config.test_store_path, folder[1][0], index[1][0]+"_F21"))

            #### ARM part #####
            just_netG_arm_output = netG_arm(torch.unsqueeze(ibmask, 1))
            output = netG_arm(output_mask)

            residual_img_path = os.path.join(config.test_store_path, folder[1][0], index[1][0]+"_residual.png")
            residual = torch.clamp(output, 0, 1)
            residual_img_pil = transforms.ToPILImage()(residual.cpu().detach()[0])
            residual_img_pil.save(residual_img_path)
            
            if config.weighted_foreground:
                output = output + output_mask
                just_netG_arm_output = just_netG_arm_output + torch.unsqueeze(ibmask, 1)
            else:
                output = output.sigmoid()
            # output = output.tanh()

            #save generated image
            gen_img_path = os.path.join(config.test_store_path, folder[1][0], index[1][0]+"_gen.png")
            gen_img_from_gt_mask_path = os.path.join(config.test_store_path, folder[1][0], index[1][0]+"_gen_from_gt_mask.png")

            output_clamp = torch.clamp(output, 0, 1)
            output_clamp_np = output_clamp.cpu().detach().numpy()
            gen_img_pil = transforms.ToPILImage()(output_clamp.cpu().detach()[0])
            gen_img_pil.save(gen_img_path)

            output_clamp_gt = torch.clamp(just_netG_arm_output, 0, 1)
            output_clamp_gt_np = output_clamp_gt.cpu().detach().numpy()
            gen_img_gt_pil = transforms.ToPILImage()(output_clamp_gt.cpu().detach()[0])
            gen_img_gt_pil.save(gen_img_from_gt_mask_path)

            #we only evaluate on how close the image sample is to its input
            Iib_np = Iib.cpu().detach().numpy()

            this_psnr = psnr(output_clamp.detach(), Iib.detach()).cpu().detach().item()
            this_ssim = ssim(output_clamp.detach(), Iib.detach()).cpu().detach().item()
            this_ie = lumpy.mean(lumpy.sqrt(lumpy.sum((output_clamp_np*255 - Iib_np*255)**2, axis=2)))

            test_psnr_arr.append(this_psnr)
            test_ssim_arr.append(this_ssim)
            test_ie_arr.append(this_ie)

    return test_psnr_arr, test_ssim_arr, test_ie_arr


def main():
    #parse args from config file
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    config = Config.from_file(args.config)

    #create dataloaders for training and testing
    normalize1_rgb = transforms.Normalize(config.mean_rgb, [1.0, 1.0, 1.0])
    normalize2_rgb = transforms.Normalize([0, 0, 0], config.std_rgb)
    normalize1 = transforms.Normalize(config.mean, [1.0])
    normalize2 = transforms.Normalize([0], config.std)
    trans = transforms.Compose([transforms.ToTensor(), normalize1, normalize2, ])
    trans_rgb = transforms.Compose([transforms.ToTensor(), normalize1_rgb, normalize2_rgb, ])
    revmean = [-x for x in config.mean]
    revstd = [1.0 / x for x in config.std]
    revnormalize1 = transforms.Normalize([0.0], revstd)
    revnormalize2 = transforms.Normalize(revmean, [1.0])
    revNormalize = transforms.Compose([revnormalize1, revnormalize2])
    revtrans = transforms.Compose([revnormalize1, revnormalize2, transforms.ToPILImage()])
    if config.csv:
        print("SUPER TUNA!!!")
        testset = datas.CSVAniTripletWithSGMFlowTest(config.testset_root, config.csv_root, \
                                                config.num_ib_frames, \
                                                config.test_flow_root, config.hist_mask_test_root, \
                                                trans, trans_rgb, config.test_size, \
                                                config.test_crop_size)
    else:
        print("tae is goin for it on ig")
        testset = datas.AniTripletWithSGMFlowTest(config.testset_root, \
                                                config.test_flow_root, config.hist_mask_test_root, \
                                                trans, trans_rgb, config.test_size, \
                                                config.discrim_crop_size, \
                                                config.patch_location)
        print("yoongi's blurry pic of holly omg ToT")
    test_sampler = torch.utils.data.SequentialSampler(testset)
    testloader = torch.utils.data.DataLoader(testset, sampler=test_sampler, batch_size=1, shuffle=False, num_workers=config.num_workers)

    sys.stdout.flush()

    #create generators
    # netG_seg = Unet(
    #             encoder_name="resnet34",
    #             encoder_weights="imagenet",
    #             in_channels=6,
    #             classes=1
    #             )
    netG_seg = Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=10,
                classes=1
                )
    netG_arm = Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=1,
                classes=1
    )
    netG_seg = netG_seg.cuda()
    netG_seg = nn.DataParallel(netG_seg)
    netG_arm = netG_arm.cuda()

    dict_netG_seg = torch.load(config.checkpoint_g_seg)
    netG_seg.load_state_dict(dict_netG_seg, strict=True)
    dict_netG_arm = torch.load(config.checkpoint_g_arm)
    netG_arm.load_state_dict(dict_netG_arm, strict=True)
    
    netG_arm = nn.DataParallel(netG_arm)

    test_psnr_arr = []
    test_ssim_arr = []
    test_ie_arr = []

    test_psnr_arr, \
        test_ssim_arr, \
        test_ie_arr = testing_loop(config, revtrans, testloader, testset, \
                                    netG_seg, netG_arm, test_psnr_arr, \
                                    test_ssim_arr, test_ie_arr)
 
    save_metrics_to_arr(config, test_psnr_arr, test_ssim_arr, test_ie_arr)

    print("")
    print("######### RESULTS ###########")
    print("PSNR:", test_psnr_arr)
    print("SSIM:", test_ssim_arr)
    print("IE:", test_ie_arr)
    print("#############################")
    print("\a"*2)
        

if __name__ == "__main__":
    main()




