import os
import re
import numpy as lumpy

import pdb



def write_individual_page(model, source, ex, root):
    print("confidence confidence")
    targetpath = os.path.join("./", model, source, ex, "index.html")
    file = open(targetpath, "w")
    file.write("<html>\n")
    file.write("<head>\n<title>"+ex+"</title>\n")
    file.write("<meta charset=\"utf-8\">\n")
    file.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
    file.write("<link rel=\"stylesheet\" href=\"../../../style.css\">\n")
    file.write("</head>\n")
    file.write("<h1>"+ex+"</h1>\n")
    file.write("<body>\n")

    if model == "inbetween_mask":
        file.write("<div class=\"row\">\n")
        file.write("<div class=\"column\">\n")
        file.write("<h4>Estimate mask</h4>\n")
        file.write("</div>\n")
        file.write("<div class=\"column\">\n")
        file.write("<h4>Ground truth mask</h4>\n")
        file.write("</div>\n")
        file.write("<div class=\"column\">\n")
        file.write("<h4>Ground truth image</h4>\n")
        file.write("</div>\n")
        file.write("</div>\n")

        local_path = os.path.join(root, "outputs", ex)
        all_examples = os.listdir(local_path)
        for example in all_examples:
            epoch_dirs = os.listdir(os.path.join(local_path, example))
            epoch_dirs.sort()
            if len(epoch_dirs) <= 0:
                continue

            file.write("<div class=\"row\">\n")

            file.write("<div class=\"column\">\n")
            newroot = os.path.join(local_path, example, epoch_dirs[-1])

            bucketpath_0 = os.path.join("../../../", "outputs", model, ex, example, epoch_dirs[-1], "0.png")
            bucketpath_ib = os.path.join("../../../", "outputs", model, ex, example, epoch_dirs[-1], "1_est_mask.png")
            bucketpath_1 = os.path.join("../../../", "outputs", model, ex, example, epoch_dirs[-1], "2.png")
            bucketpath_ib_gt = os.path.join("../../../", "outputs", model, ex, example, epoch_dirs[-1], "1_mask.png")
            bucketpath_ib_gt_im = os.path.join("../../../", "outputs", model, ex, example, epoch_dirs[-1], "1_gt.png")

            file.write("<img src=\""+bucketpath_0+"\">\n")
            file.write("<img src=\""+bucketpath_ib+"\">\n")
            file.write("<img src=\""+bucketpath_1+"\">\n")
            file.write("</div>\n")

            file.write("<div class=\"column\">\n")

            file.write("<img src=\""+bucketpath_0+"\">\n")
            file.write("<img src=\""+bucketpath_ib_gt+"\">\n")
            file.write("<img src=\""+bucketpath_1+"\">\n")
            file.write("</div>\n")

            file.write("<div class=\"column\">\n")

            file.write("<img src=\""+bucketpath_0+"\">\n")
            file.write("<img src=\""+bucketpath_ib_gt_im+"\">\n")
            file.write("<img src=\""+bucketpath_1+"\">\n")
            file.write("</div>\n")

            file.write("</div>\n")
            file.write("<hr>")
    
    elif model == "mask_to_img":
        file.write("<div class=\"row\">\n")
        file.write("<div class=\"column\">\n")
        file.write("<h4>Input mask</h4>\n")
        file.write("</div>\n")
        file.write("<div class=\"column\">\n")
        file.write("<h4>Estimated image</h4>\n")
        file.write("</div>\n")
        file.write("<div class=\"column\">\n")
        file.write("<h4>Ground truth image</h4>\n")
        file.write("</div>\n")
        file.write("</div>\n")

        local_path = os.path.join(root, "outputs", ex)
        all_examples = os.listdir(local_path)
        for example in all_examples:
            epoch_dirs = os.listdir(os.path.join(local_path, example))
            epoch_dirs.sort()
            if len(epoch_dirs) <= 0:
                continue

            file.write("<div class=\"row\">\n")

            file.write("<div class=\"column\">\n")
            if epoch_dirs == None:
                pdb.set_trace()
            newroot = os.path.join(local_path, example, epoch_dirs[-1])

            bucketpath_gen_im = os.path.join("../../../", "outputs", model, ex, example, epoch_dirs[-1], "gen_im.png")
            bucketpath_gt_img = os.path.join("../../../", "outputs", model, ex, example, epoch_dirs[-1], "gt_img.png")
            bucketpath_mask = os.path.join("../../../", "outputs", model, ex, example, epoch_dirs[-1], "mask.png")

            file.write("<img src=\""+bucketpath_mask+"\">\n")
            file.write("</div>\n")

            file.write("<div class=\"column\">\n")

            file.write("<img src=\""+bucketpath_gen_im+"\">\n")
            file.write("</div>\n")

            file.write("<div class=\"column\">\n")

            file.write("<img src=\""+bucketpath_gt_img+"\">\n")
            file.write("</div>\n")

            file.write("</div>\n")
            file.write("<hr>")


    file.write("</body>\n</html>\n")

    file.close()


def get_model_links(model, source, root):
    nextfilename = os.path.join("./", model, source, "index.html")
    nextfile = open(nextfilename, "w")
    nextfile.write("<html>\n")
    nextfile.write("<head>\n<title>Results</title>\n")
    nextfile.write("<meta charset=\"utf-8\">\n")
    nextfile.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
    nextfile.write("<link rel=\"stylesheet\" href=\"../../style.css\">\n")
    nextfile.write("</head>\n")
    nextfile.write("<body>\n")
    nextfile.write("<h1>Test results by experiment</h1>\n")

    experiments = []

    #incoming extreme hack
    if model == "inbetween_mask":
        experiments = ["avi_atd12k_adam_1e-3_gan_pos_weight_bg_over_fg_tvl1_dog_unet_TEST_results"]
    elif model == "mask_to_img":
        experiments = ["mask_to_im_weighted_fg_lrg_1e-5_beta_0-5_TEST_results", \
                        "mask_to_im_weighted_fg_lrg_1e-6_beta_0-5_TEST_results", \
                        "mask_to_im_weighted_fg_lrg_1e-5_lrd_1e-5_beta_0-5_percloss_dumb_a1_1e-1_a2_3e-1_a3_1_TEST_results", \
                        "mask_to_im_weighted_fg_lrg_1e-5_lrd_1e-5_beta_0-5_percloss_dumb_a1_1_a2_2e-1_a3_1_TEST_results", \
                        "mask_to_im_weighted_fg_lrg_1e-5_lrd_1e-5_beta_0-5_percloss_dumb_a1_1_a2_3e-1_a3_1_TEST_results", \
                        'mask_to_im_gan_weighted_fg_lrg_1e-4_beta_0-5_2_overfit_cosanneal_TRAIN_results', \
                        'mask_to_im_gan_weighted_fg_lrg_1e-4_beta_0-5_overfit_cosanneal_TRAIN_results']

    for ex in experiments:
        if not os.path.exists(os.path.join(model, source, ex)):
            os.mkdir(os.path.join(model, source, ex))
        nextfile.write("<p>\n")
        nextfile.write("<a href=\""+ex+"/index.html\" target=\"_blank\">"+ex+"</a>\n")
        nextfile.write("</p>\n")
        print(model, source, ex)
        write_individual_page(model, source, ex, root)
    
    nextfile.write("</body>\n")
    nextfile.write("</html>\n")
    nextfile.close()


def get_source_links(model, sources, root):
    #model is inbetween_mask or mask_to_img
    #sources is SU or atd12k
    #root is...i guess root project dir
    nextfilename = os.path.join("./", model, "index.html")
    nextfile = open(nextfilename, "w")
    nextfile.write("<html>\n")
    nextfile.write("<head>\n<title>Results</title>\n")
    nextfile.write("<meta charset=\"utf-8\">\n")
    nextfile.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
    nextfile.write("<link rel=\"stylesheet\" href=\"../style.css\">\n")
    nextfile.write("</head>\n")
    nextfile.write("<body>\n")
    nextfile.write("<h1>Test results by dataset</h1>\n")

    for source in sources:
        #for now, don't have these results
        if source == "SU":
            continue
        if not os.path.exists(os.path.join(model, source)):
            os.mkdir(os.path.join(model, source))
        nextfile.write("<p>\n")
        nextfile.write("<a href=\""+source+"/index.html\" target=\"_blank\">"+source+"</a>\n")
        nextfile.write("</p>\n")
        get_model_links(model, source, root)

    nextfile.write("</body>\n")
    nextfile.write("</html>\n")
    nextfile.close()


def write_compare_test_output_page(seg, arm, results_dir, metrics_dir, source, root):
    print("he's a hollywood star")
    targetpath = os.path.join(seg+"__"+arm, source, "compare", "index.html")
    file = open(targetpath, "w")
    file.write("<html>\n")
    file.write("<head>\n<title>Results</title>\n")
    file.write("<meta charset=\"utf-8\">\n")
    file.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
    file.write("<link rel=\"stylesheet\" href=\"../../../style.css\">\n")
    file.write("</head>\n")
    file.write("<h1>seg "+seg+" arm"+arm+"</h1>\n")
    file.write("<body>\n")

    local_path = os.path.join(root, "outputs", results_dir)
    local_metrics = os.path.join(root, "outputs", metrics_dir)

    if source == "SU":
        local_DAIN_path = "/vulcanscratch/lilhuang/DAIN/outputs/SU_128x256"
        local_AnimeInterp_path = "/vulcanscratch/lilhuang/AnimeInterp/outputs/final_SU_7ib_baseline"
        local_RIFE_path = "/vulcanscratch/lilhuang/arXiv2020-RIFE/outputs/final_SU_test"
        gif_root = os.path.join(root, "gif_atd12k")
    else:
        local_DAIN_path = "/vulcanscratch/lilhuang/DAIN/outputs/atd12k_test"
        local_AnimeInterp_path = "/vulcanscratch/lilhuang/AnimeInterp/outputs/final_atd12k_baseline"
        local_RIFE_path = "/vulcanscratch/lilhuang/arXiv2020-RIFE/outputs/final_atd12k_test"

    our_psnr = lumpy.load(os.path.join(local_metrics, "test_psnr_arr.npy")).mean()
    our_ssim = lumpy.load(os.path.join(local_metrics, "test_ssim_arr.npy")).mean()
    dain_psnr = lumpy.load(os.path.join(local_DAIN_path, "metrics", "psnr_arr_"+source+".npy")).mean()
    dain_ssim = lumpy.load(os.path.join(local_DAIN_path, "metrics", "ssim_arr_"+source+".npy")).mean()
    animeInterp_psnr = lumpy.load(os.path.join(local_AnimeInterp_path, "metrics", "psnr_arr.npy")).mean()
    animeInterp_ssim = lumpy.load(os.path.join(local_AnimeInterp_path, "metrics", "ssim_arr.npy")).mean()
    rife_psnr = lumpy.load(os.path.join(local_RIFE_path, "metrics", "psnr_arr.npy")).mean()
    rife_ssim = lumpy.load(os.path.join(local_RIFE_path, "metrics", "ssim_arr.npy")).mean()


    file.write("<div class=\"row\">\n")
    file.write("<div class=\"column\">\n")
    file.write("<h4>Ground truth</h4>\n")
    file.write("</div>\n")
    file.write("<div class=\"column\">\n")
    file.write("<h4>Ours (psnr "+str(our_psnr)+" ssim "+str(our_ssim)+")</h4>\n")
    file.write("</div>\n")
    file.write("<div class=\"column\">\n")
    file.write("<h4>AnimeInterp (psnr "+str(animeInterp_psnr)+" ssim "+str(animeInterp_ssim)+")</h4>\n")
    file.write("</div>\n")
    file.write("<div class=\"column\">\n")
    file.write("<h4>DAIN (psnr "+str(dain_psnr)+" ssim "+str(dain_ssim)+")</h4>\n")
    file.write("</div>\n")
    file.write("<div class=\"column\">\n")
    file.write("<h4>RIFE (psnr "+str(rife_psnr)+" ssim "+str(rife_ssim)+")</h4>\n")
    file.write("</div>\n")
    file.write("</div>\n")

    all_our_examples = os.listdir(local_path)
    all_our_examples.sort()

    for i in range(len(all_our_examples)):
        example = all_our_examples[i]

        if source == "SU":
            regex = 'frame([0-9]+)_to_frame([0-9]+)'
            supertuna = re.search(regex, example)
            frame0_num = int(supertuna.group(1))
            frame2_num = int(supertuna.group(2))
            sample_num = int((frame0_num + frame2_num)/2)
            sample_name = "sample_"+str(sample_num)

        file.write("<div class=\"row\">\n")

        bucketpath_0 = os.path.join("../../../", "outputs", results_dir, example, "0.png")
        bucketpath_ib = os.path.join("../../../", "outputs", results_dir, example, "1_gen.png")
        bucketpath_1 = os.path.join("../../../", "outputs", results_dir, example, "2.png")
        bucketpath_ib_gt = os.path.join("../../../", "outputs", results_dir, example, "1_gt.png")
        
        if source == "atd12k":
            bucketpath_ib_animeinterp = os.path.join("../../../", "baselines", "AnimeInterp", "final_atd12k_baseline", example, "frame2_est.png")
            bucketpath_ib_dain = os.path.join("../../../", "baselines", "DAIN", "atd12k_test", example, "im1_est.jpg")
            bucketpath_ib_rife = os.path.join("../../../", "baselines", "RIFE", "final_atd12k_test", example, "im1_est.jpg")
            bucketpath_gif = os.path.join("../../../", "outputs", "gif_atd12k", example)
        else:
            bucketpath_ib_animeinterp = os.path.join("../../../", "baselines", "AnimeInterp", "final_SU_7ib_baseline", example, "1_est.png")
            bucketpath_ib_dain = os.path.join("../../../", "baselines", "DAIN", "SU_128x256", sample_name, "im1_est.jpg")
            bucketpath_ib_rife = os.path.join("../../../", "baselines", "RIFE", "final_SU_test", sample_name, "im1_est.jpg")

        file.write("<div class=\"column\">\n")

        file.write("<img src=\""+bucketpath_0+"\">\n")
        file.write("<img src=\""+bucketpath_ib_gt+"\">\n")
        file.write("<img src=\""+bucketpath_1+"\">\n")
        if source == "atd12k":
            file.write("<img src=\""+bucketpath_gif+"/gt.gif\">\n")
        file.write("</div>\n")

        file.write("<div class=\"column\">\n")

        file.write("<img src=\""+bucketpath_0+"\">\n")
        file.write("<img src=\""+bucketpath_ib+"\">\n")
        file.write("<img src=\""+bucketpath_1+"\">\n")
        if source == "atd12k":
            file.write("<img src=\""+bucketpath_gif+"/gen.gif\">\n")
        file.write("</div>\n")

        file.write("<div class=\"column\">\n")

        file.write("<img src=\""+bucketpath_0+"\">\n")
        file.write("<img src=\""+bucketpath_ib_animeinterp+"\">\n")
        file.write("<img src=\""+bucketpath_1+"\">\n")
        if source == "atd12k":
            file.write("<img src=\""+bucketpath_gif+"/animeinterp.gif\">\n")
        file.write("</div>\n")

        file.write("<div class=\"column\">\n")

        file.write("<img src=\""+bucketpath_0+"\">\n")
        file.write("<img src=\""+bucketpath_ib_dain+"\">\n")
        file.write("<img src=\""+bucketpath_1+"\">\n")
        if source == "atd12k":
            file.write("<img src=\""+bucketpath_gif+"/dain.gif\">\n")
        file.write("</div>\n")

        file.write("<div class=\"column\">\n")

        file.write("<img src=\""+bucketpath_0+"\">\n")
        file.write("<img src=\""+bucketpath_ib_rife+"\">\n")
        file.write("<img src=\""+bucketpath_1+"\">\n")
        if source == "atd12k":
            file.write("<img src=\""+bucketpath_gif+"/rife.gif\">\n")
        file.write("</div>\n")

        file.write("</div>\n")
        file.write("<hr>")

    file.write("</body>\n</html>\n")

    file.close()


def write_our_test_output_page(seg, arm, results_dir, source, root):
    print("three dollaaaaa")
    targetpath = os.path.join(seg+"__"+arm, source, "ours", "index.html")
    file = open(targetpath, "w")
    file.write("<html>\n")
    file.write("<head>\n<title>Results</title>\n")
    file.write("<meta charset=\"utf-8\">\n")
    file.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
    file.write("<link rel=\"stylesheet\" href=\"../../../style.css\">\n")
    file.write("</head>\n")
    file.write("<h1>seg "+seg+" arm"+arm+"</h1>\n")
    file.write("<body>\n")

    file.write("<div class=\"row\">\n")
    file.write("<div class=\"column\">\n")
    file.write("<h4>Final estimate image</h4>\n")
    file.write("</div>\n")
    file.write("<div class=\"column\">\n")
    file.write("<h4>Ground truth image</h4>\n")
    file.write("</div>\n")
    file.write("<div class=\"column\">\n")
    file.write("<h4>Estimate image (gt mask)</h4>\n")
    file.write("</div>\n")
    file.write("<div class=\"column\">\n")
    file.write("<h4>Estimated mask</h4>\n")
    file.write("</div>\n")
    file.write("<div class=\"column\">\n")
    file.write("<h4>Ground truth mask</h4>\n")
    file.write("</div>\n")
    file.write("<div class=\"column\">\n")
    file.write("<h4>F12, F21, resid</h4>\n")
    file.write("</div>\n")
    file.write("</div>\n")

    local_path = os.path.join(root, "outputs", results_dir)
    all_examples = os.listdir(local_path)
    all_examples.sort()
    for example in all_examples:
        file.write("<div class=\"row\">\n")

        file.write("<div class=\"column\">\n")

        bucketpath_0 = os.path.join("../../../", "outputs", results_dir, example, "0.png")
        bucketpath_ib = os.path.join("../../../", "outputs", results_dir, example, "1_gen.png")
        bucketpath_1 = os.path.join("../../../", "outputs", results_dir, example, "2.png")
        bucketpath_ib_gt = os.path.join("../../../", "outputs", results_dir, example, "1_gt.png")
        bucketpath_ib_from_gt_mask = os.path.join("../../../", "outputs", results_dir, example, "1_gen_from_gt_mask.png")
        bucketpath_ib_mask = os.path.join("../../../", "outputs", results_dir, example, "1_est_mask.png")
        bucketpath_ib_mask_gt = os.path.join("../../../", "outputs", results_dir, example, "1_mask.png")
        bucketpath_f12 = os.path.join("../../../", "outputs", results_dir, example, "1_F12.jpg")
        bucketpath_f21 = os.path.join("../../../", "outputs", results_dir, example, "1_F21.jpg")
        bucketpath_ib_resid = os.path.join("../../../", "outputs", results_dir, example, "1_residual.png")

        file.write("<img src=\""+bucketpath_0+"\">\n")
        file.write("<img src=\""+bucketpath_ib+"\">\n")
        file.write("<img src=\""+bucketpath_1+"\">\n")
        file.write("</div>\n")

        file.write("<div class=\"column\">\n")

        file.write("<img src=\""+bucketpath_0+"\">\n")
        file.write("<img src=\""+bucketpath_ib_gt+"\">\n")
        file.write("<img src=\""+bucketpath_1+"\">\n")
        file.write("</div>\n")

        file.write("<div class=\"column\">\n")

        file.write("<img src=\""+bucketpath_0+"\">\n")
        file.write("<img src=\""+bucketpath_ib_from_gt_mask+"\">\n")
        file.write("<img src=\""+bucketpath_1+"\">\n")
        file.write("</div>\n")

        file.write("<div class=\"column\">\n")

        file.write("<img src=\""+bucketpath_0+"\">\n")
        file.write("<img src=\""+bucketpath_ib_mask+"\">\n")
        file.write("<img src=\""+bucketpath_1+"\">\n")
        file.write("</div>\n")

        file.write("<div class=\"column\">\n")

        file.write("<img src=\""+bucketpath_0+"\">\n")
        file.write("<img src=\""+bucketpath_ib_mask_gt+"\">\n")
        file.write("<img src=\""+bucketpath_1+"\">\n")
        file.write("</div>\n")

        file.write("<div class=\"column\">\n")

        file.write("<img src=\""+bucketpath_f12+"\">\n")
        file.write("<img src=\""+bucketpath_f21+"\">\n")
        file.write("<img src=\""+bucketpath_ib_resid+"\">\n")
        file.write("</div>\n")

        file.write("</div>\n")
        file.write("<hr>")

    file.write("</body>\n</html>\n")

    file.close()


def get_choice_of_test_results(seg, arm, results_dir, metrics_dir, source, root):
    nextfilename = os.path.join(seg+"__"+arm, source, "index.html")
    nextfile = open(nextfilename, "w")
    nextfile.write("<html>\n")
    nextfile.write("<head>\n<title>Results</title>\n")
    nextfile.write("<meta charset=\"utf-8\">\n")
    nextfile.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
    nextfile.write("<link rel=\"stylesheet\" href=\"../../style.css\">\n")
    nextfile.write("</head>\n")
    nextfile.write("<body>\n")
    nextfile.write("<h1>Please choose what you want to see</h1>\n")

    oursdir = os.path.join(seg+"__"+arm, source, "ours")
    if not os.path.exists(oursdir):
        os.makedirs(oursdir)
    nextfile.write("<p>\n")
    nextfile.write("<a href=\"ours/index.html\" target=\"_blank\">all our output lol</a>\n")
    nextfile.write("</p>\n")

    write_our_test_output_page(seg, arm, results_dir, source, root)

    comparedir = os.path.join(seg+"__"+arm, source, "compare")
    if not os.path.exists(comparedir):
        os.makedirs(comparedir)
    nextfile.write("<p>\n")
    nextfile.write("<a href=\"compare/index.html\" target=\"_blank\">comparison against baselines</a>\n")
    nextfile.write("</p>\n")

    write_compare_test_output_page(seg, arm, results_dir, metrics_dir, source, root)

    nextfile.write("</body>\n")
    nextfile.write("</html>\n")
    nextfile.close()


def get_test_source_links(seg, arm, atd12k_results_dir, atd12k_metrics_dir, \
                        SU_results_dir, SU_metrics_dir, sources, root):
    #sources is SU or atd12k
    #root is...i guess root project dir
    nextfilename = os.path.join("./", seg+"__"+arm, "index.html")
    nextfile = open(nextfilename, "w")
    nextfile.write("<html>\n")
    nextfile.write("<head>\n<title>Results</title>\n")
    nextfile.write("<meta charset=\"utf-8\">\n")
    nextfile.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
    nextfile.write("<link rel=\"stylesheet\" href=\"../style.css\">\n")
    nextfile.write("</head>\n")
    nextfile.write("<body>\n")
    nextfile.write("<h1>Some test results by test dataset</h1>\n")

    for source in sources:
        if not os.path.exists(os.path.join(seg+"__"+arm, source)):
            os.mkdir(os.path.join(seg+"__"+arm, source))
        nextfile.write("<p>\n")
        nextfile.write("<a href=\""+source+"/index.html\" target=\"_blank\">"+source+"</a>\n")
        nextfile.write("</p>\n")
        if source == "SU":
            get_choice_of_test_results(seg, arm, SU_results_dir, SU_metrics_dir, source, root)
        else:
            get_choice_of_test_results(seg, arm, atd12k_results_dir, atd12k_metrics_dir, source, root)

    nextfile.write("</body>\n")
    nextfile.write("</html>\n")
    nextfile.close()


def main_get_model_combinations():
    root = "/fs/vulcan-projects/anim_inb_lilhuang"
    sources = ["SU", "atd12k"]

    nextfile = open("index.html", "w")
    nextfile.write("<html>\n")
    nextfile.write("<head>\n<title>Results by model combination</title>\n")
    nextfile.write("<meta charset=\"utf-8\">\n")
    nextfile.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
    nextfile.write("<link rel=\"stylesheet\" href=\"style.css\">\n")
    nextfile.write("</head>\n")
    nextfile.write("<body>\n")
    nextfile.write("<h1>Test results by model combination</h1>\n")

    seg_models = ["latest_model_atd12k_adam_1e-3_gan_pos_weight_bg_over_fg_dog_unet_tvl1_1e-3_EPOCH_99.pth"]
    arm_models = ["latest_model_mask_to_im_gan_weighted_fg_lrg_1e-4_lrd_1e-4_a1_1_a2_1_a3_0_EPOCH_73.pth"]
    test_result_map = {("latest_model_atd12k_adam_1e-3_gan_pos_weight_bg_over_fg_dog_unet_tvl1_1e-3_EPOCH_99.pth", \
                        "latest_model_mask_to_im_gan_weighted_fg_lrg_1e-4_lrd_1e-4_a1_1_a2_1_a3_0_EPOCH_73.pth"): ("test_inb_TEST_results", \
                                                                                                                "test_inb_METRICS_results", \
                                                                                                                "test_inb_SU_TEST_results", \
                                                                                                                "test_inb_SU_METRICS_results")}

    for seg, arm in zip(seg_models, arm_models):
        atd12k_results_dir, atd12k_metrics_dir, SU_results_dir, SU_metrics_dir = test_result_map[(seg, arm)]
        combinationdir = os.path.join(seg+"__"+arm)
        if not os.path.exists(combinationdir):
            os.makedirs(combinationdir)
        nextfile.write("<p>\n")
        nextfile.write("<a href=\""+combinationdir+"/index.html\" target=\"_blank\">seg "+seg+" arm"+arm+"</a>\n")
        nextfile.write("</p>\n")
        get_test_source_links(seg, arm, atd12k_results_dir, atd12k_metrics_dir, \
                            SU_results_dir, SU_metrics_dir, sources, root)
    
    nextfile.write("</body>\n")
    nextfile.write("</html>\n")
    nextfile.close()


def main():
    # the old baseline data, fine-tuned on one Blender video
    root = "/fs/vulcan-projects/anim_inb_lilhuang"
    models = ["inbetween_mask", "mask_to_img"]
    sources = ["SU", "atd12k"]


    indexfile = open("index_test.html", "w")
    indexfile.write("<html>\n")
    indexfile.write("<head>\n<title>Results</title>\n")
    indexfile.write("<meta charset=\"utf-8\">\n")
    indexfile.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
    indexfile.write("<link rel=\"stylesheet\" href=\"style.css\">\n")
    indexfile.write("</head>\n")
    indexfile.write("<body>\n")
    indexfile.write("<h1>Results</h1>\n")
    for model in models:
        if not os.path.exists(model):
            os.mkdir(model)
        targetpath = os.path.join("./", model, "index.html")
        indexfile.write("<p>\n")
        indexfile.write("<a href=\""+targetpath+"\" target=\"_blank\">"+model+"</a>\n")
        indexfile.write("</p>\n")
        get_source_links(model, sources, root)

    indexfile.write("</body>\n</html>\n")

    indexfile.close()
    



if __name__ == '__main__':
    # main()
    main_get_model_combinations()



