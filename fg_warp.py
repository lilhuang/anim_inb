import numpy as lumpy
import os
import re
import torch
import cv2
import datas
import argparse
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from utils.vis_flow import flow_to_color

import pdb



def save_flow_to_img(flow, name):
    f = flow[0].data.cpu().numpy().transpose([1, 2, 0])
    fcopy = f.copy()
    fcopy[:, :, 0] = f[:, :, 1]
    fcopy[:, :, 1] = f[:, :, 0]
    cf = flow_to_color(-fcopy)
    cv2.imwrite(name, cf)


def warp_samples(img_root, flow_root, output_root):
    all_train_samples = os.listdir(img_root)
    correct_regex = "Disney_(.*)_2048x1024_t_3_k_3"
    to_tensor = transforms.ToTensor()
    to_img = transforms.ToPILImage()
    for sample in all_train_samples:
        if not re.search(correct_regex, sample):
            continue
        print(sample)
        path_img1 = os.path.join(img_root, sample, "frame1_smooth.jpg")
        path_img2 = os.path.join(img_root, sample, "frame2_smooth.jpg")
        path_img3 = os.path.join(img_root, sample, "frame3_smooth.jpg")
        img1_pil = Image.open(path_img1)
        img2_pil = Image.open(path_img2)
        img3_pil = Image.open(path_img3)

        img1 = to_tensor(img1_pil)
        img2 = to_tensor(img2_pil)
        img3 = to_tensor(img3_pil)

        #note first flow channel is x, second is y
        path_flo13 = os.path.join(flow_root, sample, "flo_smooth_dt_13.npy")
        path_flo31 = os.path.join(flow_root, sample, "flo_smooth_dt_31.npy")
        flo13 = lumpy.load(path_flo13)
        flo31 = lumpy.load(path_flo31)

        flo13 = torch.Tensor(flo13)
        flo31 = torch.Tensor(flo31)

        gridX, gridY = lumpy.meshgrid(lumpy.arange(img1.shape[2]), lumpy.arange(img1.shape[1]))
        gridX = torch.Tensor(gridX)
        gridY = torch.Tensor(gridY)

        x_13 = gridX + flo13[0]
        y_13 = gridY + flo13[1]

        x_31 = gridX + flo31[0]
        y_31 = gridY + flo31[1]

        x_12 = gridX + 0.5*flo13[0]
        y_12 = gridY + 0.5*flo13[1]

        x_32 = gridX + 0.5*flo31[0]
        y_32 = gridY + 0.5*flo31[1]


        x_13 = 2*(x_13/img1.shape[2] - 0.5)
        y_13 = 2*(y_13/img1.shape[1] - 0.5)

        x_31 = 2*(x_31/img1.shape[2] - 0.5)
        y_31 = 2*(y_31/img1.shape[1] - 0.5)

        x_12 = 2*(x_12/img1.shape[2] - 0.5)
        y_12 = 2*(y_12/img1.shape[1] - 0.5)

        x_32 = 2*(x_32/img1.shape[2] - 0.5)
        y_32 = 2*(y_32/img1.shape[1] - 0.5)

        grid_13 = torch.stack((x_13, y_13), dim=2)
        grid_31 = torch.stack((x_31, y_31), dim=2)
        grid_12 = torch.stack((x_12, y_12), dim=2)
        grid_32 = torch.stack((x_32, y_32), dim=2)

        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        img3 = img3.unsqueeze(0)
        grid_13 = grid_13.unsqueeze(0)
        grid_31 = grid_31.unsqueeze(0)
        grid_12 = grid_12.unsqueeze(0)
        grid_32 = grid_32.unsqueeze(0)

        img1_from_3 = torch.nn.functional.grid_sample(img3, grid_13)
        img3_from_1 = torch.nn.functional.grid_sample(img1, grid_31)
        img1_from_2 = torch.nn.functional.grid_sample(img2, grid_12)
        img3_from_2 = torch.nn.functional.grid_sample(img2, grid_32)

        img1_from_3 = to_img(img1_from_3[0])
        img3_from_1 = to_img(img3_from_1[0])
        img1_from_2 = to_img(img1_from_2[0])
        img3_from_2 = to_img(img3_from_2[0])

        save_dir = os.path.join(output_root, sample)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        img1_pil.save(os.path.join(save_dir, "frame1.png"))
        img2_pil.save(os.path.join(save_dir, "frame2.png"))
        img3_pil.save(os.path.join(save_dir, "frame3.png"))

        img1_from_3.save(os.path.join(save_dir, "frame1_from_3.png"))
        img3_from_1.save(os.path.join(save_dir, "frame3_from_1.png"))
        img1_from_2.save(os.path.join(save_dir, "frame1_from_2.png"))
        img3_from_2.save(os.path.join(save_dir, "frame3_from_2.png"))

    

def read_flo(filename):
    with open(filename, "rb") as f:
        magic = lumpy.fromfile(f, lumpy.float32, count=1)

        assert magic == 202021.25, f"{filename} is not a valid .flo file"

        w = lumpy.fromfile(f, lumpy.int32, count=1)
        h = lumpy.fromfile(f, lumpy.int32, count=1)
        data = lumpy.fromfile(f, lumpy.float32, count=2 * int(w) * int(h))
        flow = lumpy.resize(data, (int(h), int(w), 2))
        valid = lumpy.logical_not(lumpy.logical_or(lumpy.abs(flow[..., 0]) > 1e9, lumpy.abs(flow[..., 1]) > 1e9))

        return flow, valid.astype("int32")
        



def warp_chairs():
    img1_path = "/fs/vulcan-projects/anim_inb_lilhuang/FlyingChairs_release/data/00001_img1.ppm"
    img2_path = "/fs/vulcan-projects/anim_inb_lilhuang/FlyingChairs_release/data/00001_img2.ppm"
    flow_path = "/fs/vulcan-projects/anim_inb_lilhuang/FlyingChairs_release/data/00001_flow.flo"

    img1 = Image.open(img1_path)
    img1.save("tangerineboi_1.png")
    img2 = Image.open(img2_path)
    img2.save("tangerineboi_2.png")
    flow, valid = read_flo(flow_path)
    
    flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
    cv2.imwrite("tangerineboi_flo.png", flow_color)

    to_tensor = transforms.ToTensor()

    img1_torch = to_tensor(img1)
    img2_torch = to_tensor(img2)

    # img1_lumpy = lumpy.asarray(img1)
    # img2_lumpy = lumpy.asarray(img2)
    gridX, gridY = lumpy.meshgrid(lumpy.arange(img1_torch.shape[2]), lumpy.arange(img1_torch.shape[1]))

    # img1_torch = torch.permute(torch.Tensor(img1_lumpy), (2, 0, 1))
    # img2_torch = torch.permute(torch.Tensor(img2_lumpy), (2, 0, 1))

    # x_13_mask = lumpy.clip(gridX + flow[:,:,0], 0., img1_lumpy.shape[1]-1).astype("int32")
    # y_13_mask = lumpy.clip(gridY + flow[:,:,1], 0., img1_lumpy.shape[0]-1).astype("int32")

    x_13_mask = gridX + flow[:,:,0]
    y_13_mask = gridY + flow[:,:,1]

    x_13 = 2*(torch.Tensor(x_13_mask)/img1_torch.shape[2] - 0.5)
    y_13 = 2*(torch.Tensor(y_13_mask)/img1_torch.shape[1] - 0.5)

    grid = torch.stack((x_13, y_13), dim=2)

    img2_torch = img2_torch.unsqueeze(0)
    grid = grid.unsqueeze(0)

    empty = torch.nn.functional.grid_sample(img2_torch, grid)

    # empty = lumpy.ones(img1_lumpy.shape)
    # # empty[y_13_mask.reshape(-1), x_13_mask.reshape(-1)] = img1[gridY.reshape(-1), gridX.reshape(-1)]
    # for i in range(x_13_mask.shape[0]):
    #     for j in range(x_13_mask.shape[1]):
    #         empty[i][j] = img2_lumpy[y_13_mask[i][j]][x_13_mask[i][j]]
    #         # empty[y_13_mask[i][j]][x_13_mask[i][j]] = img1_lumpy[i][j]
    

    # cv2.imwrite("tangerineboi_gen.png", empty)

    to_img = transforms.ToPILImage()
    empty_pil = to_img(empty[0])
    empty_pil.save("tangerineboi_gen.png")

    print("done!!!!")




def test_flo():
    flow_path = Path("/fs/vulcan-projects/anim_inb_lilhuang/FlyingChairs_release/data/00001_flow.flo")

    flow, valid = read_flo(flow_path)

    flow_color =  flow_vis.flow_to_color(flow, convert_to_bgr=False)
    Image.fromarray(flow_color).save('tangerineboi_AHHH.png')





def main():
    trainset_root = '/fs/vulcan-projects/anim_inb_lilhuang/datasets/train_10k_preprocess_dog'
    testset_root = '/fs/vulcan-projects/anim_inb_lilhuang/datasets/test_2k_original_preprocess_dog'
    train_flow_root = '/fs/vulcan-projects/anim_inb_lilhuang/datasets/train_10k_preprocess_raft_flows'
    test_flow_root = '/fs/vulcan-projects/anim_inb_lilhuang/datasets/test_2k_preprocess_raft_flows'
    hist_mask_root = "/fs/vulacn-projects/anim_inb_lilhuang/datasets/train_10k_preprocess_dog"
    hist_mask_test_root = '/fs/vulcan-projects/anim_inb_lilhuang/datasets/test_2k_original_preprocess_dog'

    warp_output_root = '/fs/vulcan-projects/anim_inb_lilhuang/outputs/warp_fg_output'

    # "flo_smooth_13.npy" "flo_smooth_dt_13.npy" "frame1_smooth.npy" "frame2_smooth.npy" "frame3_smooth.npy"


    warp_samples(trainset_root, train_flow_root, warp_output_root)
    # warp_chairs()
    # test_flo()


def backwarp(flow, image):
    W = image.shape[3]
    H = image.shape[2]
    gridX, gridY = lumpy.meshgrid(lumpy.arange(W), lumpy.arange(H))
    gridX = torch.Tensor(gridX).unsqueeze(0).expand_as(flow[:,0,:,:]).float().cuda()
    gridY = torch.Tensor(gridY).unsqueeze(0).expand_as(flow[:,1,:,:]).float().cuda()

    x = gridX + flow[:,0,:,:]
    y = gridY + flow[:,1,:,:]

    x = 2*(x/W - 0.5)
    y = 2*(y/H - 0.5)

    grid = torch.stack((x, y), dim=3)

    warped = F.grid_sample(image, grid, mode="nearest")

    both_0 = torch.bitwise_and(x==0, y==0)
    imgout = torch.where(both_0, warped, image)

    # empty = lumpy.ones((W, H))
    # # empty[y_13_mask.reshape(-1), x_13_mask.reshape(-1)] = img1[gridY.reshape(-1), gridX.reshape(-1)]
    # for i in range(x.shape[0]):
    #     for j in range(x.shape[1]):
    #         # empty[i][j] = image[y[i][j]][x[i][j]]
    #         empty[y[i][j]][x[i][j]] = image[i][j]

    
    return imgout, warped


def main_warp_SU():
    # trainset_root = '/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_dog_patches_large'
    # testset_root = '/fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog'
    # csv_root = '/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_csv'
    # trainflow_root = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_tvl1_flows_patches_large"
    # testflow_root = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_tvl1_flows"

    trainset_root = '/fs/cfar-projects/anim_inb/datasets/Blender_cubes_dog_patches_large'
    testset_root = '/fs/cfar-projects/anim_inb/datasets/Blender_cubes_dog'
    csv_root = '/fs/cfar-projects/anim_inb/datasets/Blender_cubes_csv'
    trainflow_root = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_tvl1_flows_patches_large"
    testflow_root = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_tvl1_flows"

    num_ib_frames = 1
    test_size = (2048, 1024)
    flow_type = "tvl1"
    dataset = "blender_cubes"

    # dataset_root_filepath_train = "/fs/cfar-projects/anim_inb/datasets/pickles/train_SU_"+str(num_ib_frames)+"ib_"+str(test_size[0])+"x"+str(test_size[1])+"_patches"
    # dataset_root_filepath_test = "/fs/cfar-projects/anim_inb/datasets/pickles/test_SU_"+str(num_ib_frames)+"ib_"+str(test_size[0])+"x"+str(test_size[1])

    dataset_root_filepath_train = "/fs/cfar-projects/anim_inb/datasets/train_Blender_cubes_"+str(num_ib_frames)+"ib_"+str(test_size[0])+"x"+str(test_size[1])+"_patches"
    dataset_root_filepath_test = "/fs/cfar-projects/anim_inb/datasets/test_Blender_cubes_"+str(num_ib_frames)+"ib_"+str(test_size[0])+"x"+str(test_size[1])


    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    testset = datas.BlenderAniTripletPatchTest(args, testset_root, \
                                                csv_root, \
                                                num_ib_frames, \
                                                dataset_root_filepath_test, \
                                                dataset, \
                                                img_size=test_size, \
                                                resize=None, \
                                                flow_type=flow_type, \
                                                flow_root=testflow_root, \
                                                small_dataset=True, \
                                                dt=False)

    test_sampler = torch.utils.data.SequentialSampler(testset)
    testloader = torch.utils.data.DataLoader(testset, sampler=test_sampler, batch_size=1, shuffle=False, num_workers=0)
    to_img = transforms.ToPILImage()

    for validationIndex, validationData in enumerate(testloader):
        sample, rgb_sample, folder, index, masks, flow = validationData

        # img0 = masks[0].float().cuda()
        # img1 = masks[1].float().cuda()
        # img2 = masks[2].float().cuda()

        img0 = sample[0].float().cuda()
        img1 = sample[1].float().cuda()
        img2 = sample[2].float().cuda()

        F12i = flow[0].float().cuda()
        F21i = flow[1].float().cuda()
        # F12i = (1./testset.maxflow) * F12i
        # F21i = (1./testset.maxflow) * F21i

        img0_backwarp_from_img2, blah1 = backwarp(F12i, img2)
        img2_backwarp_from_img0, blah2 = backwarp(F21i, img0)

        # img0_backwarp_from_img2_np = 255*(1-img0_backwarp_from_img2[0].cpu().detach().numpy())
        # img2_backwarp_from_img0_np = 255*(1-img2_backwarp_from_img0[0].cpu().detach().numpy())
        img0_backwarp_from_img2_np = img0_backwarp_from_img2[0].cpu().detach().numpy()
        img2_backwarp_from_img0_np = img2_backwarp_from_img0[0].cpu().detach().numpy()

        # blah1_np = 255*(1-blah1[0].cpu().detach().numpy())
        # blah2_np = 255*(1-blah2[0].cpu().detach().numpy())
        blah1_np = blah1[0].cpu().detach().numpy()
        blah2_np = blah2[0].cpu().detach().numpy()

        # img0_save = 255*(1-img0[0].cpu().detach().numpy())
        # img2_save = 255*(1-img2[0].cpu().detach().numpy())
        img0_save = img0[0].cpu().detach().numpy()
        img2_save = img2[0].cpu().detach().numpy()

        cv2.imwrite("joonkoo.png", lumpy.transpose(img0_backwarp_from_img2_np, (1, 2, 0)))
        cv2.imwrite("joonkoo2.png", lumpy.transpose(img2_backwarp_from_img0_np, (1, 2, 0)))
        cv2.imwrite("joonkoo3.png", lumpy.transpose(blah1_np, (1, 2, 0)))
        cv2.imwrite("joonkoo4.png", lumpy.transpose(blah2_np, (1, 2, 0)))
        cv2.imwrite("joonkoo5.png", lumpy.transpose(img0_save, (1, 2, 0)))
        cv2.imwrite("joonkoo6.png", lumpy.transpose(img2_save, (1, 2, 0)))
        save_flow_to_img(F12i, "joonkoo7.png")
        save_flow_to_img(F21i, "joonkoo8.png")


        pdb.set_trace()






if __name__ == "__main__":
    # main()
    main_warp_SU()



