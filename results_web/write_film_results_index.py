import os
import re
import numpy as lumpy

import pdb



def main():
    targetpath = "./film_results/index.html"
    file = open(targetpath, "w")
    file.write("<html>\n")
    file.write("<head>\n<title>FILM baseline</title>\n")
    file.write("<meta charset=\"utf-8\">\n")
    file.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
    file.write("<link rel=\"stylesheet\" href=\"style.css\">\n")
    file.write("</head>\n")
    file.write("<h1>FILM baseline</h1>\n")
    file.write("<body>\n")

    file.write("<div class=\"row\">\n")

    file.write("<div class=\"column\">\n")
    file.write("<h4>Estimated inbs</h4>\n")
    file.write("</div>\n")

    file.write("<div class=\"column\">\n")
    file.write("<h4>Ground truth inbs</h4>\n")
    file.write("</div>\n")

    file.write("</div>\n")

    localroot = "/fs/vulcan-projects/anim_inb_lilhuang/frame_interpolation/photos/atd12k_output/test"
    all_samples = os.listdir(localroot)
    for sample in all_samples:
        print(sample)
        bucketpath = os.path.join("./", "atd12k_output", "test", sample)
        bucketpath_0 = os.path.join(bucketpath, "frame1.jpg")
        bucketpath_1_est = os.path.join(bucketpath, "frame2_est.jpg")
        bucketpath_1_gt = os.path.join(bucketpath, "frame2_gt.jpg")
        bucketpath_2 = os.path.join(bucketpath, "frame3.jpg")

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

        file.write("</div>\n")


    file.write("/body>\n</html>\n")


    file.close()


if __name__ == "__main__":
    main()








