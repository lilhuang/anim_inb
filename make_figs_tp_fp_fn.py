import cv2
import os
import re
import numpy as lumpy

import pdb


def main_SU():
    # results_root = "/fs/cfar-projects/anim_inb/arXiv2020-RIFE/outputs/final_SU_smol_test"
    # pictures_root = "/fs/cfar-projects/anim_inb/arXiv2020-RIFE/outputs/final_SU_smol_fig"
    # results_root = "/fs/cfar-projects/anim_inb/AnimeInterp/outputs/final_SU_smol_3ib_TEST_results"
    # pictures_root = "/fs/cfar-projects/anim_inb/AnimeInterp/outputs/final_SU_smol_3ib_FIG_results"
    # results_root = "/fs/cfar-projects/anim_inb/outputs/avi_SU_FINAL_TEST_results_2/epoch_300"
    # pictures_root = "/fs/cfar-projects/anim_inb/outputs/avi_SU_FINAL_FIG_results_2/epoch_300"
    # results_root = "/fs/cfar-projects/anim_inb/outputs/final_pt_JamesBaxterChel_3ib_TEST_results/epoch_200"
    # pictures_root = "/fs/cfar-projects/anim_inb/outputs/final_pt_JamesBaxterChel_3ib_FIG_results/epoch_200"
    # results_root = "/fs/cfar-projects/anim_inb/outputs/final_pt_JamesBaxterMoses_3ib_TEST_results/epoch_200"
    # pictures_root = "/fs/cfar-projects/anim_inb/outputs/final_pt_JamesBaxterMoses_3ib_FIG_results/epoch_200"
    # results_root = "/fs/cfar-projects/anim_inb/outputs/final_pt_JamesBaxterThumper_3ib_TEST_results/epoch_200"
    # pictures_root = "/fs/cfar-projects/anim_inb/outputs/final_pt_JamesBaxterThumper_3ib_FIG_results/epoch_200"
    results_root = "/fs/cfar-projects/anim_inb/outputs/final_pt_MiltKahlRobinHoodWalk_3ib_TEST_results/epoch_200"
    pictures_root = "/fs/cfar-projects/anim_inb/outputs/final_pt_MiltKahlRobinHoodWalk_3ib_FIG_results/epoch_200"

    all_examples = os.listdir(results_root)
    all_examples.sort()
    # all_examples = [
    #     "StevenHug_2048x1024_019_to_023",
    #     "StevenHug_2048x1024_021_to_025",
    #     "StevenHug_2048x1024_026_to_030",
    #     "StevenHug_2048x1024_039_to_043",
    #     "StevenHug_2048x1024_060_to_064",
    #     "StevenHug_2048x1024_065_to_069",
    #     "StevenHug_2048x1024_066_to_070",
    # ]
    for ex in all_examples:
        if ex == "metrics":
            continue
        full_path = os.path.join(results_root, ex)
        outpath = os.path.join(pictures_root, ex)
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        # srcframe_path = os.path.join(full_path, "0.png")
        # inbframe_gt_path = os.path.join(full_path, "1_mask.png")
        # trgframe_path = os.path.join(full_path, "2.png")
        # srcframe_path = os.path.join(full_path, "im0.jpg")
        # inbframe_gt_path = os.path.join(full_path, "im1_gt.jpg")
        # trgframe_path = os.path.join(full_path, "im2.jpg")
        srcframe_path = os.path.join(full_path, "0.png")
        inbframe_gt_path = os.path.join(full_path, "1_mask.png")
        trgframe_path = os.path.join(full_path, "2.png")

        # inb_grayscale_path = os.path.join(full_path, "1_grayscale.png")
        inbframe_est_path = os.path.join(full_path, "1_est_mask.png")
        # inbframe_est_path = os.path.join(full_path, "im1_est.jpg")
        # inbframe_est_path = os.path.join(full_path, "1_est.png")

        #note cv2 is BGR not RGB
        #src red, trg blue
        #tp black, fp purple (in est but not gt), fn green (in gt but not est)

        #src trg overlay
        srcframe = cv2.imread(srcframe_path, 0)
        trgframe = cv2.imread(trgframe_path, 0)

        background = 255*lumpy.ones(srcframe.shape)
        srcframe_red = lumpy.stack([srcframe, srcframe, background], axis=2)
        trgframe_blue = lumpy.stack([background, trgframe, trgframe], axis=2)
        srctrg = cv2.addWeighted(srcframe_red, 0.5, trgframe_blue, 0.5, 0)
        srctrg_path = os.path.join(outpath, "0_2_overlay.png")
        cv2.imwrite(srctrg_path, srctrg)

        #tp fp fn
        inbframe_gt = cv2.imread(inbframe_gt_path, 0)
        # inb_grayscale = cv2.imread(inb_grayscale_path, 0)
        inbframe_est = cv2.imread(inbframe_est_path, 0)

        inbmask_gt = 1 - inbframe_gt/255
        # inbgrayscale_mask = 1 - inb_grayscale/255
        inbmask_est = 1 - inbframe_est/255

        # where_tp = lumpy.bitwise_and(inbmask_gt.astype("uint8"), inbmask_est.astype("uint8"))
        # difference = inbmask_gt - inbmask_est
        # where_fp = lumpy.where(difference == -1, 1, 0)
        # where_fn = lumpy.where(difference == 1, 1, 0)

        tp_fp_fn = 255*lumpy.ones((srcframe.shape[0], srcframe.shape[1], 3))
        for i in range(tp_fp_fn.shape[0]):
            for j in range(tp_fp_fn.shape[1]):
                if inbmask_gt[i][j] >= 0.1 and inbmask_est[i][j] > 0.1:
                    tp_fp_fn[i][j][0] = 0
                    tp_fp_fn[i][j][1] = 0
                    tp_fp_fn[i][j][2] = 0
                elif inbmask_gt[i][j] < 0.1 and inbmask_est[i][j] > 0.1:
                    tp_fp_fn[i][j][0] = 128
                    tp_fp_fn[i][j][1] = 0
                    tp_fp_fn[i][j][2] = 128
                elif inbmask_gt[i][j] >= 0.1 and inbmask_est[i][j] < 0.1:
                    tp_fp_fn[i][j][0] = 0
                    tp_fp_fn[i][j][1] = 255
                    tp_fp_fn[i][j][2] = 0
        tp_fp_fn_path = os.path.join(outpath, "tp_fp_fn.png")
        cv2.imwrite(tp_fp_fn_path, tp_fp_fn)

        print(ex)
        # count += 1



def main_suzannes_exr():
    # results_root = "/fs/cfar-projects/anim_inb/outputs/avi_suzannes_exr_1ib_recon_gan_lrg_1e-3_deep_unet_TEST_results/epoch_060"
    # pictures_root = "/fs/cfar-projects/anim_inb/outputs/avi_suzannes_exr_1ib_recon_gan_lrg_1e-3_deep_unet_FIG_results/epoch_060"
    # results_root = "/fs/cfar-projects/anim_inb/outputs/avi_SU_smol_1ib_lrg_1e-3_lrd_1e-4_deep_unet_recon_gan_TEST_results/epoch_100"
    # pictures_root = "/fs/cfar-projects/anim_inb/outputs/avi_SU_smol_1ib_lrg_1e-3_lrd_1e-4_deep_unet_recon_gan_FIG_results/epoch_100"
    # results_root = "/fs/cfar-projects/anim_inb/outputs/avi_SU_smol_3ib_lrg_1e-3_lrd_1e-4_deep7_unet_recon_gan_TEST_results/epoch_100"
    # pictures_root = "/fs/cfar-projects/anim_inb/outputs/avi_SU_smol_3ib_lrg_1e-3_lrd_1e-4_deep7_unet_recon_gan_FIG_results/epoch_100"
    # results_root = "/fs/cfar-projects/anim_inb/arXiv2020-RIFE/outputs/final_Suzanne_exr_test"
    # pictures_root = "/fs/cfar-projects/anim_inb/arXiv2020-RIFE/outputs/final_Suzanne_exr_fig"
    # results_root = "/fs/cfar-projects/anim_inb/outputs/ours_final_suzannes_exr_1ib_TEST_results"
    # pictures_root = "/fs/cfar-projects/anim_inb/outputs/ours_final_suzannes_exr_1ib_FIG_results"
    results_root = "/fs/cfar-projects/anim_inb/outputs/final_suzannes_exr_1ib_TEST_results"
    pictures_root = "/fs/cfar-projects/anim_inb/outputs/final_Suzannes_exr_1ib_FIG_results"
    

    with open("examples.txt", "r") as ex_file:
        all_examples = ex_file.readlines()

        # all_examples = os.listdir(results_root)
        # all_examples.sort()
        for ex in all_examples:
            ex = ex.strip()
            if ex == "metrics":
                continue
            full_path = os.path.join(results_root, ex)
            outpath = os.path.join(pictures_root, ex)
            if not os.path.exists(outpath):
                os.makedirs(outpath)

            # srcframe_path = os.path.join(full_path, "0.png")
            # inbframe_gt_path = os.path.join(full_path, "1_mask.png")
            # trgframe_path = os.path.join(full_path, "2.png")
            # srcframe_path = os.path.join(full_path, "im0.jpg")
            # inbframe_gt_path = os.path.join(full_path, "im1_gt.jpg")
            # trgframe_path = os.path.join(full_path, "im2.jpg")
            srcframe_path = os.path.join(full_path, "0.png")
            inbframe_gt_path = os.path.join(full_path, "1_gt.png")
            trgframe_path = os.path.join(full_path, "2.png")

            # inb_grayscale_path = os.path.join(full_path, "1_grayscale.png")
            # inbframe_est_path = os.path.join(full_path, "1_est_mask.png")
            inbframe_est_path = os.path.join(full_path, "1_est.png")

            #note cv2 is BGR not RGB
            #src red, trg blue
            #tp black, fp purple (in est but not gt), fn green (in gt but not est)

            #src trg overlay
            srcframe = cv2.imread(srcframe_path, 0)
            trgframe = cv2.imread(trgframe_path, 0)

            background = 255*lumpy.ones(srcframe.shape)
            srcframe_red = lumpy.stack([srcframe, srcframe, background], axis=2)
            trgframe_blue = lumpy.stack([background, trgframe, trgframe], axis=2)
            srctrg = cv2.addWeighted(srcframe_red, 0.5, trgframe_blue, 0.5, 0)
            srctrg_path = os.path.join(outpath, "0_2_overlay.png")
            cv2.imwrite(srctrg_path, srctrg)

            #tp fp fn
            inbframe_gt = cv2.imread(inbframe_gt_path, 0)
            # inb_grayscale = cv2.imread(inb_grayscale_path, 0)
            inbframe_est = cv2.imread(inbframe_est_path, 0)

            inbmask_gt = 1 - inbframe_gt/255
            # inbgrayscale_mask = 1 - inb_grayscale/255
            inbmask_est = 1 - inbframe_est/255

            # where_tp = lumpy.bitwise_and(inbmask_gt.astype("uint8"), inbmask_est.astype("uint8"))
            # difference = inbmask_gt - inbmask_est
            # where_fp = lumpy.where(difference == -1, 1, 0)
            # where_fn = lumpy.where(difference == 1, 1, 0)

            tp_fp_fn = 255*lumpy.ones((srcframe.shape[0], srcframe.shape[1], 3))
            for i in range(tp_fp_fn.shape[0]):
                for j in range(tp_fp_fn.shape[1]):
                    if inbmask_gt[i][j] == 1 and inbmask_est[i][j] >= 0.05:
                        tp_fp_fn[i][j][0] = 0
                        tp_fp_fn[i][j][1] = 0
                        tp_fp_fn[i][j][2] = 0
                    elif inbmask_gt[i][j] == 0 and inbmask_est[i][j] >= 0.05:
                        tp_fp_fn[i][j][0] = 128
                        tp_fp_fn[i][j][1] = 0
                        tp_fp_fn[i][j][2] = 128
                    elif inbmask_gt[i][j] == 1 and inbmask_est[i][j] < 0.05:
                        tp_fp_fn[i][j][0] = 0
                        tp_fp_fn[i][j][1] = 255
                        tp_fp_fn[i][j][2] = 0
            tp_fp_fn_path = os.path.join(outpath, "tp_fp_fn.png")
            cv2.imwrite(tp_fp_fn_path, tp_fp_fn)

            print(ex)
            # count += 1

        



if __name__ == "__main__":
    # main_suzannes_exr()
    main_SU()





