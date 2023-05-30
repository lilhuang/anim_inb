import datas
import cv2
import sys
import os
import numpy as lumpy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from utils.vis_flow import flow_to_color
from PIL import Image

import pdb



def save_flow_to_img(flow, name):
    # pdb.set_trace()
    f = flow[0].data.cpu().numpy().transpose([1, 2, 0])
    fcopy = f.copy()
    fcopy[:, :, 0] = f[:, :, 1]
    fcopy[:, :, 1] = f[:, :, 0]
    cf = flow_to_color(-fcopy)
    cv2.imwrite(name, cf)


def save_mask_to_img(mask, name):
    if mask.shape[0] == 1:
        mask = lumpy.transpose(mask.cpu().numpy(), (1, 2, 0))
    else:
        mask = mask.cpu().numpy()
    cv2.imwrite(name, (1 - mask)*255)


class backWarp(nn.Module):
    """
    A class for creating a backwarping object.
    This is used for backwarping to an image:
    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1, 
    it generates I0 <-- backwarp(F_0_1, I1).
    NOTE: Taken from Super-SloMo repo!
    ...
    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
    """


    def __init__(self, W, H, device):
        """
        Parameters
        ----------
            W : int
                width of the image.
            H : int
                height of the image.
            device : device
                computation device (cpu/cuda). 
        """


        super(backWarp, self).__init__()
        # create a grid
        gridX, gridY = lumpy.meshgrid(lumpy.arange(W), lumpy.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)
        
    def forward(self, img, flow):
        """
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
        I0  = backwarp(I1, F_0_1)
        Parameters
        ----------
            img : tensor
                frame I1.
            flow : tensor
                optical flow from I0 and I1: F_0_1.
        Returns
        -------
            tensor
                frame I0.
        """


        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x,y), dim=3)
        # Sample pixels using bilinear interpolation.
        img_float = img.float()
        imgOut = torch.nn.functional.grid_sample(img_float, grid)
        return imgOut



def interpolate_batch(loader, warp_output_root, back_warp, train=True):
    to_img = transforms.ToPILImage()
    if train:
        true_warp_output_root = warp_output_root+"_TRAIN"
    else:
        true_warp_output_root = warp_output_root+"_TEST"
    t = 0.5
    alpha = 0.5
    for index, data in enumerate(loader, 0):
        print("warping", index, "/", len(loader))
        sys.stdout.flush()
        sample, masks, flow, folder, indices = data

        framesrc = sample[0].cuda()
        frameinb = sample[1]
        frametrg = sample[2].cuda()

        flow_13, flow_31 = flow
        flow_13 = flow_13.cuda().float()
        flow_31 = flow_31.cuda().float()
        
        masksrc = masks[0].cuda().float()
        maskinb = masks[1]
        masktrg = masks[2].cuda().float()

        if not os.path.exists(os.path.join(true_warp_output_root, folder[0][0])):
            os.makedirs(os.path.join(true_warp_output_root, folder[0][0]))

        srcframe_path = os.path.join(true_warp_output_root, folder[0][0], "frame0.jpg")
        inbframe_path_gt = os.path.join(true_warp_output_root, folder[1][0], "frame1_gt.jpg")
        trgframe_path = os.path.join(true_warp_output_root, folder[2][0], "frame2.jpg")
        flow_13_path = os.path.join(true_warp_output_root, folder[0][0], "flow_13.jpg")
        flow_31_path = os.path.join(true_warp_output_root, folder[0][0], "flow_31.jpg")
        srcmask_path = os.path.join(true_warp_output_root, folder[0][0], "mask0.jpg")
        inbmask_path_gt = os.path.join(true_warp_output_root, folder[1][0], "mask1_gt.jpg")
        trgmask_path = os.path.join(true_warp_output_root, folder[2][0], "mask2.jpg")

        masksrc_img = save_mask_to_img(masksrc[0][0], srcmask_path)
        maskinb_img = save_mask_to_img(maskinb[0], inbmask_path_gt)
        masktrg_img = save_mask_to_img(masktrg[0][0], trgmask_path)

        framesrc_pil = to_img(framesrc.cpu().detach()[0])
        frameinb_pil = to_img(frameinb.cpu().detach()[0])
        frametrg_pil = to_img(frametrg.cpu().detach()[0])
        framesrc_pil.save(srcframe_path)
        frameinb_pil.save(inbframe_path_gt)
        frametrg_pil.save(trgframe_path)
        save_flow_to_img(flow_13, flow_13_path)
        save_flow_to_img(flow_31, flow_31_path)

        F_t0 = -(1-t)*t*flow_13 + (t**2)*flow_31
        F_0t = ((1-t)**2)*flow_13 - t*(1-t)*flow_31

        backwarp_framesrc = back_warp(framesrc, flow_13)
        backwarp_frametrg = back_warp(frametrg, flow_31)

        backwarp_masksrc = back_warp(masksrc, flow_13)
        backwarp_masktrg = back_warp(masktrg, flow_31)

        frameinb_est = alpha*backwarp_framesrc + (1-alpha)*backwarp_frametrg
        maskinb_est = alpha*backwarp_masksrc + (1-alpha)*backwarp_masktrg

        # frameinb_est_pil = to_img(frameinb_est[0])
        # maskinb_est_pil = to_img(maskinb_est[0])
        frameinb_est_np = lumpy.transpose(frameinb_est[0].cpu().detach().numpy(), (1, 2, 0))
        maskinb_est_np = 255*lumpy.transpose((1 - maskinb_est[0].cpu().detach().numpy()), (1, 2, 0))
        inbframe_path_est = os.path.join(true_warp_output_root, folder[1][0], "frame1_warp.jpg")
        # frameinb_est_pil.save(inbframe_path_est)
        cv2.imwrite(inbframe_path_est, frameinb_est_np)
        inbmask_path_est = os.path.join(true_warp_output_root, folder[1][0], "mask1_warp.jpg")
        # maskinb_est_pil.save(inbmask_path_est)
        cv2.imwrite(inbmask_path_est, maskinb_est_np)



def main():
    trainset_root = '/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_png'
    testset_root = '/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_png'
    flow_root = '/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_npz_flow'
    csv_root = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_csv"

    warp_output_root = '/fs/cfar-projects/anim_inb/outputs/warp_output'

    dataset = "suzanne_exr" #either "suzanne_exr", "blender_cubes", "suzanne", or "SU"
    test_size = (2048, 1024)
    patch_size = 512
    test_resize = None
    random_reverse = False
    dt = False
    num_ib_frames = 1
    flow_type = "gt" #tvl1, raft/dl name, or gt
    small_dataset = False

    lr = 1e-4
    lr_d = 1e-5
    beta1 = 0.5
    # warp_weight = 0.1
    warp_weight = 1

    # model = 'UNet_RRDB' #either UNet or UNet_RRDB
    model = 'UNet'
    discrim = None #either 'patch', 'multiple patch', or None
    mask_loss = True

    recon_loss = False
    gan_loss = False
    warp_loss = True

    num_workers = 0
    batch_size = 96

    trainset = datas.CSVEXRTriplet(csv_root, trainset_root, flow_root, num_ib_frames, \
                                    train=True, img_size=test_size, patch_size=patch_size, \
                                    discrim_crop_size=None, random_flip=False, \
                                    random_reverse=random_reverse, dt=dt, \
                                    patch_location=None)
    testset = datas.CSVEXRTriplet(csv_root, testset_root, flow_root, num_ib_frames, \
                                    train=False, img_size=test_size, patch_size=patch_size, \
                                    discrim_crop_size=None, random_flip=False, \
                                    random_reverse=random_reverse, dt=dt, \
                                    patch_location=None)

    sampler = torch.utils.data.RandomSampler(trainset)
    test_sampler = torch.utils.data.SequentialSampler(testset)
    trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, sampler=test_sampler, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    back_warp = backWarp(test_size[0], test_size[1], "cuda").cuda()

    # interpolate_batch(trainloader, warp_output_root, back_warp, train=True)
    interpolate_batch(testloader, warp_output_root, back_warp, train=False)



if __name__ == "__main__":
    main()


