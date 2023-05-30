import pyexr
import os
import re
import csv
import h5py
import argparse
import numpy as lumpy


import pdb


def check_for_corruption():
    csvroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_csv"
    flowroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_flow"
    ibs = [1, 3, 7]
    splits = ["train", "test"]
    regex = "(t_.*_s_.*_r_.*_[0-9]_exr).csv"
    corrupted_filenames = []
    for split in splits:
        for ib in ibs:
            csvroot_2 = os.path.join(csvroot, split, str(ib)+"ib")
            csv_files = os.listdir(csvroot_2)
            for csvfile in csv_files:
                if not re.search(regex, csvfile):
                    continue
                folder = re.search(regex, csvfile).group(1)
                flowfile = os.path.join(flowroot, split, str(ib)+"ib", folder, "flo.h5")
                try:
                    h5 = h5py.File(flowfile, "r")
                    h5.close()
                    print("yay")
                except:
                    # pdb.set_trace()
                    print("nah")
                    corrupted_filenames.append(flowfile+"\n")
    with open("corrupted_filenames.txt", "w") as f:
        f.writelines(corrupted_filenames)
                

def main_repair_corrupted():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_corrupted', type=int, required=True) #1-8
    args = parser.parse_args()      

    flowroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_flow"
    csvroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_csv"
    dataroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr"
    corrupt_files = "/fs/cfar-projects/anim_inb/corrupted_filenames.txt"
    regex = os.path.join(flowroot, "(.*)", "([0-9])ib", "(t_.*_s_.*_r_.*_[0-9]_exr)", "flo.h5")
    with open(corrupt_files, "r") as f:
        all_files = f.readlines()
        corrupted_filename = all_files[args.num_corrupted].strip()
        match = re.search(regex, corrupted_filename)
        split = match.group(1)
        ib = match.group(2)
        folder = match.group(3)
        os.remove(corrupted_filename)
        flowroot_2 = os.path.join(flowroot, split, ib+"ib", folder)
        csvroot_2 = os.path.join(csvroot, split, ib+"ib")
        file = os.path.join(csvroot_2, folder+".csv")
        write_h5_flow_from_exr(flowroot_2, csvroot_2, dataroot, file, split, int(ib), folder)



def write_h5_flow_from_exr(flowroot_2, csvroot_2, dataroot, file, split, ib, folder):
    with h5py.File(os.path.join(flowroot_2, "flo.h5"), "a") as h5f:
        filename = os.path.join(csvroot_2, file)
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                vectorx_arr = []
                vectory_arr = []
                vectorz_arr = []
                vectorw_arr = []
                regex2 = "multilayer_([0-9]+).exr"
                src_num = int(re.search(regex2, row[0]).group(1))
                trg_num = int(re.search(regex2, row[2]).group(1))
                groupname = "/{:04d}".format(src_num)
                if "{:04d}".format(src_num) in h5f.keys():
                    print("skip")
                    continue

                print(split, ib, folder, str(src_num))

                for i in range(src_num, trg_num+1):
                    exrpath = os.path.join(dataroot, folder, "multilayer_{:04d}.exr".format(i))
                    exrfile = pyexr.open(exrpath)
                    if i < trg_num:
                        vectorx_arr.append(exrfile.get("View Layer.Vector.X"))
                        vectory_arr.append(exrfile.get("View Layer.Vector.Y"))
                    if i > src_num:
                        vectorz_arr.append(exrfile.get("View Layer.Vector.Z"))
                        vectorw_arr.append(exrfile.get("View Layer.Vector.W"))
                
                vectorx_arr = lumpy.asarray(vectorx_arr)
                vectory_arr = lumpy.asarray(vectory_arr)
                vectorz_arr = lumpy.asarray(vectorz_arr)
                vectorw_arr = lumpy.asarray(vectorw_arr)
                
                flo13_x = lumpy.sum(vectorz_arr[:-1], axis=0)
                flo13_y = lumpy.sum(vectorw_arr[:-1], axis=0)
                flo31_x = -1*lumpy.sum(vectorx_arr[1:], axis=0)
                flo31_y = -1*lumpy.sum(vectory_arr[1:], axis=0)

                flo13_np = lumpy.concatenate((flo13_x, flo13_y), axis=2)
                flo31_np = lumpy.concatenate((flo31_x, flo31_y), axis=2)

                h5f.require_group(groupname)
                h5f.create_dataset(os.path.join(groupname, "flo13"), data=flo13_np)
                h5f.create_dataset(os.path.join(groupname, "flo31"), data=flo31_np)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', type=int, required=True)
    parser.add_argument('--ib', type=int, required=True)
    parser.add_argument('--split', type=str, required=True)
    args = parser.parse_args()


    csvroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_csv"
    dataroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr"
    flowroot = "/fs/cfar-projects/anim_inb/datasets/Suzanne_exr_flow"
    if not os.path.exists(flowroot):
        os.makedirs(flowroot)
    # ibs = [1, 3, 7]
    # splits = ["train", "test"]

    regex = "(t_.*_s_.*_r_.*_[0-9]_exr).csv"
    # for split in splits:
    #     for ib in ibs:
    csvroot_2 = os.path.join(csvroot, args.split, str(args.ib)+"ib")
    csv_files = os.listdir(csvroot_2)
    csv_files.sort()
    if args.example >= len(csv_files):
        print("YOU'RE DONE")
        return
    file = csv_files[args.example]
    # for file in csv_files:
    if not re.search(regex, file):
        print("YOU'RE DONE AGAIN")
        return
    folder = re.search(regex, file).group(1)
    flowroot_2 = os.path.join(flowroot, args.split, str(args.ib)+"ib", folder)
    if not os.path.exists(flowroot_2):
        os.makedirs(flowroot_2)
    write_h5_flow_from_exr(flowroot_2, csvroot_2, dataroot, \
                            file, args.split, args.ib, folder)







if __name__ == "__main__":
    # main()
    main_repair_corrupted()
    # check_for_corruption()


