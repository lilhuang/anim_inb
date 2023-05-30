import subprocess
import os
import re
import shutil
import time
from PIL import Image

import pdb



def main():
    root = "/fs/cfar-projects/anim_inb"
    outputroot = os.path.join(root, "outputs")
    gifroot = os.path.join(root, "output_gifs")
    # root = "/fs/cfar-projects/anim_inb/DAIN"
    # outputroot = os.path.join(root, "outputs")
    # gifroot = os.path.join(root, "output_gifs_dain")
    # root = "/fs/cfar-projects/anim_inb/arXiv2020-RIFE"
    # outputroot = os.path.join(root, "outputs")
    # gifroot = os.path.join(root, "output_gifs_rife")
    # root = "/fs/cfar-projects/anim_inb/AnimeInterp"
    # outputroot = os.path.join(root, "outputs")
    # gifroot = os.path.join(root, "output_gifs_AnimeInterp")

    # outputs_SU = os.path.join(outputroot, "avi_SU_3ib_gan_lrg_5e-4_lrd_5e-5_unet_large_patch_warp_tvl1_rgb_2_TEST_results")
    # outputs_cubes = os.path.join(outputroot, "avi_blender_cubes_1ib_gan_lrg_1e-4_lrd_1e-5_unet_large_patch_warp_tvl1_rgb_small_dataset_TEST_results")
    # outputs_suzannes = os.path.join(outputroot, "avi_suzannes_1ib_gan_lrg_1e-4_lrd_1e-5_unet_large_patch_warp_tvl1_rgb_small_dataset_TEST_results")
    # outputs_SU = os.path.join(outputroot, "SU_final_output")
    # outputs_cubes = os.path.join(outputroot, "blender_cubes_final_output")
    # outputs_suzannes = os.path.join(outputroot, "suzanne_final_output")
    # outputs_SU = os.path.join(outputroot, "final_SU_test")
    # outputs_cubes = os.path.join(outputroot, "final_Blender_cubes_test")
    # outputs_suzannes = os.path.join(outputroot, "final_Suzanne_test")
    # outputs_SU = os.path.join(outputroot, "final_SU_results")
    # outputs_cubes = os.path.join(outputroot, "final_Blender_cubes_results")
    # outputs_suzannes = os.path.join(outputroot, "final_Suzanne_results")
    outputs_suzannes_exr = os.path.join(outputroot, "avi_suzannes_exr_1ib_recon_lrg_1e-3_lrd_1e-5_deep_unet_TEST_results")

    # gif_SU = os.path.join(gifroot, "gif_SU")
    # if not os.path.exists(gif_SU):
    #     os.makedirs(gif_SU)
    
    # gif_cubes = os.path.join(gifroot, "gif_cubes")
    # if not os.path.exists(gif_cubes):
    #     os.makedirs(gif_cubes)

    # gif_suzannes = os.path.join(gifroot, "gif_suzannes")
    # if not os.path.exists(gif_suzannes):
    #     os.makedirs(gif_suzannes)

    gif_suzannes_exr = os.path.join(gifroot, "gif_suzannes_exr")
    if not os.path.exists(gif_suzannes_exr):
        os.makedirs(gif_suzannes_exr)

    # experiments_SU = os.listdir(outputs_SU)
    # experiments_SU.sort()

    # experiments_cubes = os.listdir(outputs_cubes)
    # experiments_cubes.sort()

    # experiments_suzannes = os.listdir(outputs_suzannes)
    # experiments_suzannes.sort()

    experiments_suzannes_exr = os.listdir(outputs_suzannes_exr)
    experiments_suzannes_exr.sort()

    # domains = [experiments_SU, experiments_cubes, experiments_suzannes]
    domains = [experiments_suzannes_exr]
    # gif_outputs = [gif_SU, gif_cubes, gif_suzannes]
    gif_outputs = [gif_suzannes_exr]
    # outputs = [outputs_SU, outputs_cubes, outputs_suzannes]
    outputs = [outputs_suzannes_exr]

    working_dir_gen = os.path.join(gifroot, "gif_gen_working_dir")
    if not os.path.exists(working_dir_gen):
        os.mkdir(working_dir_gen)

    working_dir_gt = os.path.join(gifroot, "gif_gt_working_dir")
    if not os.path.exists(working_dir_gt):
        os.mkdir(working_dir_gt)

    for i, cur_domain in enumerate(domains):
        final_epoch = cur_domain[-1]
        all_ex = os.listdir(os.path.join(outputs[i], final_epoch))
        for ex in all_ex:
            regex = "sample_"
            if ex == "metrics":
                continue
            elif re.search(regex, ex):
                continue
            full_path = os.path.join(outputs[i], final_epoch, ex)

            frame0_path = os.path.join(full_path, "0.png")
            frame2_path = os.path.join(full_path, "2.png")
            frame1_gen_path = os.path.join(full_path, "1_est_mask.png")
            frame1_gt_path = os.path.join(full_path, "1_mask.png")

            # f12_path = os.path.join(full_path, "1_F12_"+epochs[-1]+".jpg")
            # f21_path = os.path.join(full_path, "1_F21_"+epochs[-1]+".jpg")
            # frame0_warp_path = os.path.join(full_path, "0_warp_from_inb.png")
            # frame2_warp_path = os.path.join(full_path, "2_warp_from_inb.png")

            if not os.path.exists(os.path.join(gif_outputs[i], ex)):
                os.makedirs(os.path.join(gif_outputs[i], ex))

            shutil.copy(frame0_path, os.path.join(gif_outputs[i]+"/"+ex+"/0.png"))
            shutil.copy(frame2_path, os.path.join(gif_outputs[i]+"/"+ex+"/2.png"))
            shutil.copy(frame1_gen_path, os.path.join(gif_outputs[i]+"/"+ex+"/1_est.png"))
            shutil.copy(frame1_gt_path, os.path.join(gif_outputs[i]+"/"+ex+"/1.png"))
            # shutil.copy(f12_path, os.path.join(gif_outputs[i]+"/"+ex+"/flo13.jpg"))
            # shutil.copy(f21_path, os.path.join(gif_outputs[i]+"/"+ex+"/flo31.jpg"))

            # if os.path.exists(frame0_warp_path):
            #     shutil.copy(frame0_warp_path, os.path.join(gif_outputs[i]+"/"+ex+"/0_warp_from_inb.jpg"))
            #     shutil.copy(frame2_warp_path, os.path.join(gif_outputs[i]+"/"+ex+"/2_warp_from_inb.jpg"))

            trg_frame0_path = os.path.join(working_dir_gen, "frame0.png")
            trg_frame2_path = os.path.join(working_dir_gen, "frame2.png")
            trg_frame1_path = os.path.join(working_dir_gen, "frame1.png")

            trg_frame0_gt_path = os.path.join(working_dir_gt, "frame0.png")
            trg_frame2_gt_path = os.path.join(working_dir_gt, "frame2.png")
            trg_frame1_gt_path = os.path.join(working_dir_gt, "frame1.png")

            shutil.copy(frame0_path, trg_frame0_path)
            shutil.copy(frame2_path, trg_frame2_path)
            shutil.copy(frame1_gen_path, trg_frame1_path)

            shutil.copy(frame0_path, trg_frame0_gt_path)
            shutil.copy(frame2_path, trg_frame2_gt_path)
            shutil.copy(frame1_gt_path, trg_frame1_gt_path)

            bashCommand_gen_gif = "ffmpeg -f image2 -framerate 1 -i "+working_dir_gen+"/frame%d.png "+gif_outputs[i]+"/"+ex+"/gen.gif"
            bashCommand_gen_mp4 = "ffmpeg -f image2 -framerate 1 -i "+working_dir_gen+"/frame%d.png "+gif_outputs[i]+"/"+ex+"/gen.mp4"

            bashCommand_gt_gif = "ffmpeg -f image2 -framerate 1 -i "+working_dir_gt+"/frame%d.png "+gif_outputs[i]+"/"+ex+"/gt.gif"
            bashCommand_gt_mp4 = "ffmpeg -f image2 -framerate 1 -i "+working_dir_gt+"/frame%d.png "+gif_outputs[i]+"/"+ex+"/gt.mp4"

            subprocess.run(bashCommand_gen_gif, shell=True)
            subprocess.run(bashCommand_gen_mp4, shell=True)
            subprocess.run(bashCommand_gt_gif, shell=True)
            subprocess.run(bashCommand_gt_mp4, shell=True)

            os.remove(trg_frame0_path)
            os.remove(trg_frame1_path)
            os.remove(trg_frame2_path)

            os.remove(trg_frame0_gt_path)
            os.remove(trg_frame1_gt_path)
            os.remove(trg_frame2_gt_path)


        





if __name__ == "__main__":
    main()

