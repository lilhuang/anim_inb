import os
import shutil
import re

import pdb

def create_new_dataset():
    dataset_root = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_png"
    testset_root = os.path.join(dataset_root, "test", "1ib")
    trainset_root = os.path.join(dataset_root, "train", "1ib")

    testset_root_trg = os.path.join(dataset_root+"_example", "test")
    trainset_root_trg = os.path.join(dataset_root+"_example", "train")

    roots = [testset_root, trainset_root]
    roots_trg = [testset_root_trg, trainset_root_trg]
    for i, root in enumerate(roots):
        all_vids = os.listdir(root)
        for vid in all_vids:
            all_examples = os.listdir(os.path.join(root, vid))
            for example in all_examples:
                oldname = os.path.join(root, vid, example)
                newname = os.path.join(roots_trg[i], vid+"_"+example)
                shutil.copytree(oldname, newname)
                print(newname)


def create_new_flowset():
    flow_root = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_npz_2stream_flow"
    testset_root = os.path.join(flow_root, "test", "1ib")
    trainset_root = os.path.join(flow_root, "train", "1ib")

    testset_root_trg = os.path.join(flow_root+"_example", "test")
    trainset_root_trg = os.path.join(flow_root+"_example", "train")

    regexes_ex_num = ["flows_([0-9]+)_to_[0-9]+.npz", "flows_([0-9]+)_to_[0-9]+_[0-9]+_[0-9]+.npz"]
    regex_patch = "flows_[0-9]+_to_[0-9]+_([0-9]+_[0-9]+).npz"

    roots = [testset_root, trainset_root]
    roots_trg = [testset_root_trg, trainset_root_trg]
    for i, root in enumerate(roots):
        if not os.path.exists(roots_trg[i]):
            os.makedirs(roots_trg[i])
        all_vids = os.listdir(root)
        for vid in all_vids:
            all_examples = os.listdir(os.path.join(root, vid))
            for example in all_examples:
                oldname = os.path.join(root, vid, example)
                exnum = re.search(regexes_ex_num[i], example).group(1)
                if i == 1:
                    patchnum = re.search(regex_patch, example).group(1)
                    newname = os.path.join(roots_trg[i], vid+"_example_"+exnum+"_"+patchnum+".npz")
                else:
                    newname = os.path.join(roots_trg[i], vid+"_example_"+exnum+".npz")
                shutil.copyfile(oldname, newname)
                print(newname)





if __name__ == "__main__":
    # create_new_dataset()
    create_new_flowset()





