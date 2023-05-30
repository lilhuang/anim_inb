# final SU seq reverse augmentation prepatation 
#  image path - /fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog/StevenHug_2048x1024/frame066.png
# flow1ib - /fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_2048x1024_1ib/frame066_to_frame070.npz
# flow3ib - /fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_2048x1024/frame066_to_frame070.npz
# csv path - /fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_smol_sequential_halfhalf_2_csv/train_triplets_3ib.c





import os
from os.path import join, basename
import numpy as np
import shutil


csv_dir = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_smol_sequential_halfhalf_2_csv"
train_csv_file = "train_triplets_3ib.csv"
test_csv_file = "test_triplets_3ib.csv"
flow_3ib_dir = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_2048x1024"
flow_rev_3ib_dir=flow_3ib_dir+"_rev"
flow_flip_3ib_dir=flow_3ib_dir+"_flip"
flow_1ib_dir = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_2048x1024_1ib"
flow_rev_1ib_dir=flow_1ib_dir+"_rev"
flow_flip_1ib_dir=flow_1ib_dir+"_flip"

dataset_dir = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog/StevenHug_2048x1024"
flip_dataset_dir = dataset_dir+"_flip"

csv_aug_dir = csv_dir+"_aug_2"
print("Aug csv directory: ", csv_aug_dir)
os.makedirs(csv_aug_dir, exist_ok=True)

#copy test csv file to aug dir
test_csv_file_path = os.path.join(csv_dir, test_csv_file)
test_csv_file_aug_path = os.path.join(csv_aug_dir, test_csv_file)
print("writing to ", test_csv_file_aug_path)
shutil.copyfile(test_csv_file_path, test_csv_file_aug_path)

#create train csv file
# first copy all the lines from the original train csv file
train_csv_file_path = os.path.join(csv_dir, train_csv_file)
train_csv_file_aug_path = os.path.join(csv_aug_dir, train_csv_file)
print("writing to ", train_csv_file_aug_path)
with open(train_csv_file_path, 'r') as f:
    lines = f.readlines()
with open(train_csv_file_aug_path, 'w') as f:
    f.writelines(lines)

# then add the reversed lines
## Why are these called csv and thery are not comman separated values????
# why Lillian? why?
with open(train_csv_file_aug_path, 'a') as f:
    for line in lines:
        row = line.split(' ')
        img1 = row[0]
        img3 = row[1]
        img5 = row[2]

        img1_name = os.path.basename(img1).split('.')[0]
        img3_name = os.path.basename(img3).split('.')[0]
        img5_name = os.path.basename(img5).split('.')[0]

        flow51 = join(flow_rev_3ib_dir, img5_name+"_to_"+img1_name+".npz")
        flow53 = join(flow_rev_1ib_dir, img5_name+"_to_"+img3_name+".npz")
        flow31 = join(flow_rev_1ib_dir, img3_name+"_to_"+img1_name+".npz")

        # frame066.png frame068.png frame070.png frame066_to_frame070.npz frame066_to_frame068.npz frame068_to_frame070.npz
        # 1 3 5 15 13 35
        # frame070.png frame068.png frame066.png frame070_to_frame066.npz frame070_to_frame068.npz frame068_to_frame066.npz
        # 5 3 1 51 53 31
        new_line = img5+' '+img3+' '+img1+' '+flow51+' '+flow53+' '+flow31+'\n'
        f.write(new_line)

# then add the flipped lines
with open(train_csv_file_aug_path, 'a') as f:
    for line in lines:
        row = line.split(' ')
        img1 = row[0]
        img3 = row[1]
        img5 = row[2]

        img1_name = os.path.basename(img1).split('.')[0]
        img3_name = os.path.basename(img3).split('.')[0]
        img5_name = os.path.basename(img5).split('.')[0]

        def write_line(suffix):
            # flip_img1 = join(flip_dataset_dir, img1_name+"_"+suffix+".png")
            # flip_img3 = join(flip_dataset_dir, img3_name+"_"+suffix+".png")
            # flip_img5 = join(flip_dataset_dir, img5_name+"_"+suffix+".png")

            flip_img1 = join(flip_dataset_dir.replace('flip', suffix), img1_name+".png")
            flip_img3 = join(flip_dataset_dir.replace('flip', suffix), img3_name+".png")
            flip_img5 = join(flip_dataset_dir.replace('flip', suffix), img5_name+".png")


            flip_flow15 = join(flow_flip_3ib_dir, img1_name+"_"+suffix+"_to_"+img5_name+"_"+suffix+".npz")
            flip_flow13 = join(flow_flip_1ib_dir, img1_name+"_"+suffix+"_to_"+img3_name+"_"+suffix+".npz")
            flip_flow35 = join(flow_flip_1ib_dir, img3_name+"_"+suffix+"_to_"+img5_name+"_"+suffix+".npz")
            # frame066.png frame068.png frame070.png frame066_to_frame070.npz frame066_to_frame068.npz frame068_to_frame070.npz
            # 1 3 5 15 13 35
            new_line = flip_img1+' '+flip_img3+' '+flip_img5+' '+flip_flow15+' '+flip_flow13+' '+flip_flow35+'\n'
            f.write(new_line)
        write_line("flipv")
        write_line("fliph")
        write_line("flipvh")









        

