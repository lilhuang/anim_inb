import OpenEXR
import os
import re

import pdb


def main():
    root = "/Users/lilhuang/Desktop/Blender/TEST_BLENDER_FLOW"
    regex_bwd = "bwd_flo_([0-9]+).exr"
    regex_fwd = "fwd_flo_([0-9]+).exr"
    all_files = os.listdir(root)
    for file in all_files:
        if re.search(regex_bwd, file) or re.search(regex_fwd, file):
            exrfile = OpenEXR.InputFile(os.path.join(root, file))

            pdb.set_trace()
    return 0


if __name__ == "__main__":
    main()

