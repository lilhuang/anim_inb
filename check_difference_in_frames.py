import numpy as lumpy
import os
import re
import csv
import cv2
import pickle
import h5py
import random
import pyexr
from collections import OrderedDict
import matplotlib.pyplot as plt
from PIL import Image

import pdb


def main():
    dataroot = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog/StevenHug_2048x1024"

    frame1 = cv2.imread(os.path.join(dataroot, "frame041.png")) / 255
    frame3 = cv2.imread(os.path.join(dataroot, "frame043.png")) / 255

    difference = lumpy.square(frame1 - frame3)

    pdb.set_trace()




if __name__ == "__main__":
    main()



