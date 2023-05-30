import csv
import os

import pdb

def main():
    pencil_tests = ["JamesBaxterChel_2s_3ib",
                    "JamesBaxterMoses_2s_3ib",
                    "JamesBaxterThumper_2s_1s_1ib",
                    "MiltKahlMadamMim_1s_2s_1ib",
                    "MiltKahlRobinHood_2s_3ib",
                    "MiltKahlRobinHoodWalk_2s_1s_1ib"]
    dataroot = "/fs/cfar-projects/anim_inb/datasets"
    csvroot = os.path.join(dataroot, "all_pencil_tests_csvs_seq")
    trainfile = os.path.join(csvroot, "train_triplets.csv")
    testfile = os.path.join(csvroot, "test_triplets.csv")

    traincsv = open(trainfile, "w", newline="")
    testcsv = open(testfile, "w", newline="")

    trainwriter = csv.writer(traincsv, delimiter = ' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    testwriter = csv.writer(testcsv, delimiter = ' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    csvroot_SU = os.path.join(dataroot, "SU_24fps", "StevenHug_2048x1024_smol_sequential_halfhalf_2_csv")
    train_csvs = [os.path.join(csvroot_SU, "train_triplets_3ib.csv")]
    test_csvs = [os.path.join(csvroot_SU, "test_triplets_3ib.csv")]

    csvroot_pt = os.path.join(dataroot, "vid_pngs_csvs_seq")
    for pt in pencil_tests:
        train_csvs.append(os.path.join(csvroot_pt, "train_"+pt+".csv"))
        test_csvs.append(os.path.join(csvroot_pt, "test_"+pt+".csv"))

    for traincsv_old in train_csvs:
        with open(traincsv_old, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                trainwriter.writerow(row)

    for testcsv_old in test_csvs:
        with open(testcsv_old, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                testwriter.writerow(row)

    traincsv.close()
    testcsv.close() 



if __name__ == "__main__":
    main()




