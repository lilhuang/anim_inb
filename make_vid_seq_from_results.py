import subprocess
import os
import re
import shutil
import time
from PIL import Image

import pdb



def main():
    # root = "/fs/cfar-projects/anim_inb"
    # outputroot = os.path.join(root, "outputs")
    # gifroot = os.path.join(root, "output_gifs")

    # outputs_test = os.path.join(outputroot, "avi_SU_FINAL_FULL_seq_halfhalf_2_results_")

    root = "/fs/cfar-projects/anim_inb/arXiv2020-RIFE"
    outputroot = os.path.join(root, "outputs")
    gifroot = os.path.join(root, "output_gifs_rife")

    # outputs_test = os.path.join(outputroot, "final_SU_halfhalf_2_pretrained_test")
    outputs_test = os.path.join(outputroot, "final_SU_halfhalf_2_test")

    gif_su_seq = os.path.join(gifroot, "gif_su_seq_ourdata_2")
    if not os.path.exists(gif_su_seq):
        os.makedirs(gif_su_seq)

    working_dir_gen = os.path.join(gifroot, "gif_gen_working_dir")
    if not os.path.exists(working_dir_gen):
        os.mkdir(working_dir_gen)

    working_dir_gt = os.path.join(gifroot, "gif_gt_working_dir")
    if not os.path.exists(working_dir_gt):
        os.mkdir(working_dir_gt)

    start = 4
    end = 70
    jump = 4
    for i in range(start, end, jump):
        print(i)
        if i + jump > end:
            break
        dirname = "StevenHug_2048x1024_{:03d}_to_{:03d}".format(i, i+jump)
        # frame0_path = os.path.join(outputs_test, dirname, "0.png")
        # frame1_gen_path = os.path.join(outputs_test, dirname, "1_est_mask.png")
        # frame1_gt_path = os.path.join(outputs_test, dirname, "1_mask.png")

        frame0_path = os.path.join(outputs_test, dirname, "im0.jpg")
        frame1_gen_path = os.path.join(outputs_test, dirname, "im1_est.jpg")
        frame1_gt_path = os.path.join(outputs_test, dirname, "im1_gt.jpg")

        shutil.copy(frame0_path, os.path.join(gif_su_seq+"/{:03d}.jpg".format(i//2)))
        shutil.copy(frame1_gen_path, os.path.join(gif_su_seq+"/{:03d}_est.jpg".format((i+(jump//2))//2)))
        shutil.copy(frame1_gt_path, os.path.join(gif_su_seq+"/{:03d}_gt.jpg".format((i+(jump//2))//2)))

        trg_frame0_path = os.path.join(working_dir_gen, "frame{:03d}.jpg".format(i//2))
        trg_frame1_path = os.path.join(working_dir_gen, "frame{:03d}.jpg".format((i+(jump//2))//2))

        trg_frame0_gt_path = os.path.join(working_dir_gt, "frame{:03d}.jpg".format(i//2))
        trg_frame1_gt_path = os.path.join(working_dir_gt, "frame{:03d}.jpg".format((i+(jump//2))//2))

        shutil.copy(frame0_path, trg_frame0_path)
        shutil.copy(frame1_gen_path, trg_frame1_path)

        shutil.copy(frame0_path, trg_frame0_gt_path)
        shutil.copy(frame1_gt_path, trg_frame1_gt_path)

        if i == 64:
            # frame2_path = os.path.join(outputs_test, dirname, "2.png")
            frame2_path = os.path.join(outputs_test, dirname, "im2.jpg")
            shutil.copy(frame2_path, os.path.join(gif_su_seq+"/{:03d}.jpg".format(end//2)))
            trg_frame2_path = os.path.join(working_dir_gen, "frame{:03d}.jpg".format(end//2))
            trg_frame2_gt_path = os.path.join(working_dir_gt, "frame{:03d}.jpg".format(end//2))
            shutil.copy(frame2_path, trg_frame2_path)
            shutil.copy(frame2_path, trg_frame2_gt_path)        

    bashCommand_gen_gif = "ffmpeg -f image2 -start_number 1 -framerate 12 -i "+working_dir_gen+"/frame%03d.jpg "+gif_su_seq+"/gen.gif"
    bashCommand_gen_mp4 = "ffmpeg -f image2 -start_number 1 -framerate 12 -i "+working_dir_gen+"/frame%03d.jpg "+gif_su_seq+"/gen.mp4"

    bashCommand_gen_gif_slow = "ffmpeg -f image2 -start_number 1 -framerate 3 -i "+working_dir_gen+"/frame%03d.jpg "+gif_su_seq+"/gen_slow.gif"
    bashCommand_gen_mp4_slow = "ffmpeg -f image2 -start_number 1 -framerate 3 -i "+working_dir_gen+"/frame%03d.jpg "+gif_su_seq+"/gen_slow.mp4"

    bashCommand_gt_gif = "ffmpeg -f image2 -start_number 1 -framerate 12 -i "+working_dir_gt+"/frame%03d.jpg "+gif_su_seq+"/gt.gif"
    bashCommand_gt_mp4 = "ffmpeg -f image2 -start_number 1 -framerate 12 -i "+working_dir_gt+"/frame%03d.jpg "+gif_su_seq+"/gt.mp4"

    bashCommand_gt_gif_slow = "ffmpeg -f image2 -start_number 1 -framerate 3 -i "+working_dir_gt+"/frame%03d.jpg "+gif_su_seq+"/gt_slow.gif"
    bashCommand_gt_mp4_slow = "ffmpeg -f image2 -start_number 1 -framerate 3 -i "+working_dir_gt+"/frame%03d.jpg "+gif_su_seq+"/gt_slow.mp4"

    # subprocess.run(bashCommand_gen_gif, shell=True)
    # subprocess.run(bashCommand_gen_mp4, shell=True)
    # subprocess.run(bashCommand_gt_gif, shell=True)
    # subprocess.run(bashCommand_gt_mp4, shell=True)
    subprocess.run(bashCommand_gen_gif_slow, shell=True)
    subprocess.run(bashCommand_gen_mp4_slow, shell=True)
    subprocess.run(bashCommand_gt_gif_slow, shell=True)
    subprocess.run(bashCommand_gt_mp4_slow, shell=True)

    # os.remove(trg_frame0_path)
    # os.remove(trg_frame1_path)
    # os.remove(trg_frame2_path)

    # os.remove(trg_frame0_gt_path)
    # os.remove(trg_frame1_gt_path)
    # os.remove(trg_frame2_gt_path)


        





if __name__ == "__main__":
    main()

