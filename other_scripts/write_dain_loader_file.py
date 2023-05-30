import os
import re

def main(im_size):
    root = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/train_10k_preprocess_dog"
    test_root = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/test_2k_original_preprocess_dog"

    regex = str(im_size[0])+"x"+str(im_size[1])+"_t_3_k_3$"

    output_train_filename = str(im_size[0])+"x"+str(im_size[1])+"_tri_trainlist.txt"
    output_test_filename = str(im_size[0])+"x"+str(im_size[1])+"_tri_testlist.txt"

    train_file = open(output_train_filename, "w")
    test_file = open(output_test_filename, "w")

    train_folders = os.listdir(root)
    for folder in train_folders:
        if not re.search(regex, folder):
            continue
        else:
            train_file.write(folder+"\n")
    
    test_folders = os.listdir(test_root)
    for folder in test_folders:
        if not re.search(regex, folder):
            continue
        else:
            test_file.write(folder+"\n")

    train_file.close()
    test_file.close()

if __name__ == "__main__":
    im_size = (2048, 1024)
    main(im_size)

