import os
import re
import numpy as lumpy

import pdb



def write_individual_page(t, root):
    repo_root = "/fs/cfar-projects/anim_inb/"
    print("confidence confidence")
    targetpath = os.path.join(root, t, "index.html")
    file = open(targetpath, "w")
    file.write("<html>\n")
    file.write("<head>\n<title>"+t+"</title>\n")
    file.write("<meta charset=\"utf-8\">\n")
    file.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
    file.write("<link rel=\"stylesheet\" href=\"../style.css\">\n")
    file.write("</head>\n")
    file.write("<h1>"+t+"</h1>\n")
    file.write("<body>\n")

    # root_folder_name = "avi_atd12k_gan_lrg_1e-3_lrd_2e-4_unet_rrdb_dt_raft_smooth_centerish_patchgan_patch_"+t+"_results"
    # root_folder_name = "avi_atd12k_gan_lrg_1e-3_lrd_2e-4_unet_rrdb_dt_raft_smooth_patchgan_highmag_large_patch_warp_"+t+"_results"
    root_folder_name = "avi_blender_cubes_1ib_gan_lrg_1e-3_lrd_2e-4_unet_rrdb_dt_patchgan_large_patch_warp_"+t+"_results"
    bucketroot = os.path.join("../", "outputs", root_folder_name)
    localroot = os.path.join(repo_root, "outputs", root_folder_name)
    all_samples = os.listdir(localroot)

    file.write("<div class=\"row\">\n")
    file.write("<div class=\"column\">\n")
    file.write("<h4>Estimate mask</h4>\n")
    file.write("</div>\n")
    file.write("<div class=\"column\">\n")
    file.write("<h4>Ground truth mask</h4>\n")
    file.write("</div>\n")
    file.write("<div class=\"column\">\n")
    file.write("<h4>Flows</h4>\n")
    file.write("</div>\n")
    if t == "TRAIN":
        file.write("<div class=\"column\">\n")
        file.write("<h4>Warps</h4>\n")
        file.write("</div>\n")
    file.write("</div>\n")
    


    for sample in all_samples:
        if t == "TEST":
            bucketpath = os.path.join(bucketroot, sample, "epoch_30") #soon have to fix to epoch_60
            bucketpath_F12 = os.path.join(bucketpath, "1_F12_epoch_30.jpg")
            bucketpath_F21 = os.path.join(bucketpath, "1_F21_epoch_30.jpg")
        else:
            epochs = os.listdir(os.path.join(localroot, sample))
            epoch_nums = []
            regex = "epoch_([0-9]+)"
            for epoch in epochs:
                epoch_num = re.search(regex, epoch).group(1)
                epoch_nums.append(int(epoch_num))
            epoch_nums.sort()
            if epoch_nums[-1] < 20:
                continue
            bucketpath = os.path.join(bucketroot, sample, "epoch_"+str(epoch_nums[-1]))
            bucketpath_F12 = os.path.join(bucketpath, "1_F12_epoch_"+str(epoch_nums[-1])+".jpg")
            bucketpath_F21 = os.path.join(bucketpath, "1_F21_epoch_"+str(epoch_nums[-1])+".jpg")
            
            bucketpath_warp_0_from_1 = os.path.join(bucketpath, "0_warp_from_inb.png")
            bucketpath_warp_2_from_1 = os.path.join(bucketpath, "2_warp_from_inb.png")
        bucketpath_0 = os.path.join(bucketpath, "0.png")
        bucketpath_1_est = os.path.join(bucketpath, "1_est_mask.png")
        bucketpath_2 = os.path.join(bucketpath, "2.png")
        bucketpath_1_gt = os.path.join(bucketpath, "1_mask.png")
            

        file.write("<div class=\"row\">\n")

        file.write("<div class=\"column\">\n")

        file.write("<img src=\""+bucketpath_0+"\">\n")
        file.write("<img src=\""+bucketpath_1_est+"\">\n")
        file.write("<img src=\""+bucketpath_2+"\">\n")
        file.write("</div>\n")

        file.write("<div class=\"column\">\n")

        file.write("<img src=\""+bucketpath_0+"\">\n")
        file.write("<img src=\""+bucketpath_1_gt+"\">\n")
        file.write("<img src=\""+bucketpath_2+"\">\n")
        file.write("</div>\n")

        file.write("<div class=\"column\">\n")

        file.write("<img src=\""+bucketpath_F12+"\">\n")
        file.write("<img src=\""+bucketpath_F21+"\">\n")
        file.write("</div>\n")

        if t == "TRAIN":
            file.write("<div class=\"column\">\n")

            file.write("<img src=\""+bucketpath_warp_0_from_1+"\">\n")
            file.write("<img src=\""+bucketpath_warp_2_from_1+"\">\n")
            file.write("</div>\n")

        file.write("</div>\n")
        file.write("<hr>")


    file.write("</body>\n</html>\n")

    file.close()



def main():
    # the old baseline data, fine-tuned on one Blender video
    print("hello")
    root = "/fs/cfar-projects/anim_inb/results_web/rrdb_results"
    arr = ['TRAIN', 'TEST']

    indexfile = open("index.html", "w")
    indexfile.write("<html>\n")
    indexfile.write("<head>\n<title>Results</title>\n")
    indexfile.write("<meta charset=\"utf-8\">\n")
    indexfile.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
    indexfile.write("<link rel=\"stylesheet\" href=\"style.css\">\n")
    indexfile.write("</head>\n")
    indexfile.write("<body>\n")
    indexfile.write("<h1>Results</h1>\n")

    print("my name is inigo montoya")

    for t in arr:

        print("you killed my father")

        if not os.path.exists(os.path.join(root, t)):
            os.makedirs(os.path.join(root, t))
        targetpath = os.path.join("./", t, "index.html")
        indexfile.write("<p>\n")
        indexfile.write("<a href=\""+targetpath+"\" target=\"_blank\">"+t+"</a>\n")
        indexfile.write("</p>\n")
        write_individual_page(t, root)
    
    print("prepare to die")

    indexfile.write("</body>\n</html>\n")

    indexfile.close()
    



if __name__ == '__main__':
    main()
    # main_get_model_combinations()



