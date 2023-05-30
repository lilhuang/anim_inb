import wandb
import os
from PIL import Image
import numpy as lumpy

import pdb


def main():
    print("airplane pt 2")

    # results_wandb = wandb.init(project="table-test")
    # results_wandb = wandb.init(project="3ib_dt")
    # results_wandb = wandb.init(project="3ib")
    # results_wandb = wandb.init(project="1ib")
    # results_wandb = wandb.init(project="7ib_tvl1")
    # results_wandb = wandb.init(project="1ib_tvl1")
    results_wandb = wandb.init(project="1ib_tvl1_weighted_loss")
    # results_wandb = wandb.init(project="3ib_no_warp")
    columns_to_add = ["id", "flows", "gt"]

    # root = "/fs/cfar-projects/anim_inb/outputs/avi_blender_cubes_1ib_gan_lrg_1e-3_lrd_2e-4_unet_rrdb_dt_patchgan_large_patch_warp_TEST_results"
    # root = "/fs/cfar-projects/anim_inb/outputs/avi_blender_cubes_3ib_gan_lrg_1e-3_lrd_2e-4_unet_rrdb_dt_patchgan_large_patch_warp_TEST_results"
    # root = "/fs/cfar-projects/anim_inb/outputs/avi_blender_cubes_3ib_gan_lrg_1e-3_lrd_2e-4_unet_rrdb_patchgan_large_patch_warp_TEST_results"
    # root = "/fs/cfar-projects/anim_inb/outputs/avi_blender_cubes_3ib_gan_lrg_1e-3_lrd_2e-4_unet_rrdb_patchgan_large_patch_TEST_results"
    # root = "/fs/cfar-projects/anim_inb/outputs/avi_blender_cubes_1ib_gan_lrg_1e-3_lrd_2e-4_unet_rrdb_patchgan_large_patch_warp_TEST_results"
    # root = "/fs/cfar-projects/anim_inb/outputs/avi_blender_cubes_7ib_gan_lrg_1e-3_lrd_2e-4_unet_rrdb_patchgan_large_patch_warp_tvl1_TEST_results"
    # root = "/fs/cfar-projects/anim_inb/outputs/avi_blender_cubes_1ib_gan_lrg_1e-3_lrd_2e-4_unet_rrdb_patchgan_large_patch_warp_tvl1_3_TEST_results"
    root = "/fs/cfar-projects/anim_inb/outputs/avi_blender_cubes_1ib_gan_lrg_1e-3_lrd_2e-4_unet_rrdb_patchgan_large_patch_warp_tvl1_weighted_loss_TEST_results"
    all_ex = os.listdir(root)
    list_epochs = os.listdir(os.path.join(root, all_ex[0]))
    list_epochs.sort()
    for epoch in list_epochs:
        columns_to_add.append("est "+epoch)
    results_wandb_table = wandb.Table(columns=columns_to_add) 

    for ex in all_ex:
        all_epochs = os.listdir(os.path.join(root, ex))
        all_epochs.sort()
        
        data_to_add = [ex]
        for j, epoch in enumerate(all_epochs):
            curdir = os.path.join(root, ex, epoch)
            srcframe = Image.open(os.path.join(curdir, "0.png"))
            trgframe = Image.open(os.path.join(curdir, "2.png"))
            inbframe_gt = Image.open(os.path.join(curdir, "1_mask.png"))
            inbframe_est = Image.open(os.path.join(curdir, "1_est_mask.png"))
            F12 = Image.open(os.path.join(curdir, "1_F12_"+epoch+".jpg"))
            F21 = Image.open(os.path.join(curdir, "1_F21_"+epoch+".jpg"))

            if j == 0:
                print("henlo")
                data_to_add.append([wandb.Image(F12), wandb.Image(F21)])
                data_to_add.append([wandb.Image(srcframe), wandb.Image(inbframe_gt), wandb.Image(trgframe)])
            data_to_add.append([wandb.Image(srcframe), wandb.Image(inbframe_est), wandb.Image(trgframe)])
        results_wandb_table.add_data(*data_to_add)
    
    results_wandb.log({"Table Name": results_wandb_table})




if __name__ == "__main__":
    main()
