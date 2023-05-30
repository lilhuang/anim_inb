# final SU seq reverse augmentation prepatation 
#  image path - /fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog/StevenHug_2048x1024/frame066.png


import os
import cv2
import numpy as np
import random

dataset_dir = "/fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog/StevenHug_2048x1024"
transalate_dataset_dir = dataset_dir+"_t"
os.makedirs(transalate_dataset_dir, exist_ok=True)

flow_3ib_dir = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_2048x1024"
flow_transalate_3ib_dir=flow_3ib_dir+"_t"
os.makedirs(flow_transalate_3ib_dir, exist_ok=True)

flow_1ib_dir = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_2048x1024_1ib"
flow_transalate_1ib_dir=flow_1ib_dir+"_flip"
os.makedirs(flow_transalate_1ib_dir, exist_ok=True)


for img_file in sorted(os.listdir(dataset_dir)):
    #load the image file
    img_file_path = os.path.join(dataset_dir, img_file)
    #read the image file
    img = cv2.imread(img_file_path)
    def translate(image, tx, ty):
        height, width = image.shape[:2]
        T = np.float32([[1, 0, tx], [0, 1, ty]])
        img_translation = cv2.warpAffine(image, T, (width, height))
        return img_translation

    # random translation
    tx = random.randint(-100, 100)
    ty = random.randint(-100, 100)

    #translate the image
    img_t = translate(img, tx, ty)
    #save the flipped image
    img_t_file_path = os.path.join(transalate_dataset_dir, img_file.split(".")[0]+"_t.png")
    cv2.imwrite(img_t_file_path, img_t)

    flow_file = 





    


def gen_flip_flow_dir(flow_dir, flip_flow_dir):
    os.makedirs(flip_flow_dir, exist_ok=True)
    for flow_file in tqdm(sorted(os.listdir(flow_dir)), total=len(os.listdir(flow_dir))):
        print(flow_file)
        flow_file_path = os.path.join(flow_dir, flow_file)
        #load the flow file
        flow_npz = np.load(flow_file_path)


        def process_flow(suffix, fn):
            flip_flow_file = flow_file.split("_to_")[0]+"_"+suffix+"_to_"+flow_file.split("_to_")[1].split(".")[0]+"_"+suffix+".npz"
            flip_flow_file_path = os.path.join(flip_flow_dir, flip_flow_file)
            flo13 = fn(flow_npz['flo13'])
            flo31 = fn(flow_npz['flo31'])
            np.savez_compressed(flip_flow_file_path, flo13=flo13, flo31=flo31)

        #fliperse the frame numbers in the flow file name
        suffix = "flipv"
        #save the flipersed flow file
        def flipv_flow(flow):
            flow = np.transpose(flow, (1,2,0)) #[2, H, W]->[H, W, 2]
            flow = cv2.flip(flow, 0) #[H, W, 2]
            flow = np.transpose(flow, (2,0,1)) #[H, W, 2]->[2, H, W]
            flow[1]=flow[1]*-1 #dy becomes -dy
        process_flow(suffix, flipv_flow)
        
        suffix = "fliph"
        #save the flipersed flow file
        def fliph_flow(flow):
            flow = np.transpose(flow, (1,2,0)) #[2, H, W]->[H, W, 2]
            flow = cv2.flip(flow, 1) #[H, W, 2]
            flow = np.transpose(flow, (2,0,1)) #[H, W, 2]->[2, H, W]
            flow[0]=flow[0]*-1 #dx becomes -dx
        process_flow(suffix, fliph_flow)

        suffix = "flipvh"
        #save the flipersed flow file
        def flipvh_flow(flow):
            flow = np.transpose(flow, (1,2,0)) #[2, H, W]->[H, W, 2]
            flow = cv2.flip(flow, -1) #[H, W, 2]
            flow = np.transpose(flow, (2,0,1)) #[H, W, 2]->[2, H, W]
            flow=flow*-1 #dx,dy becomes -dx,-dy
        process_flow(suffix, flipvh_flow)




flow_3ib_dir = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_2048x1024"
flow_flip_3ib_dir=flow_3ib_dir+"_flip"
gen_flip_flow_dir(flow_3ib_dir, flow_flip_3ib_dir)

flow_1ib_dir = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_2048x1024_1ib"
flow_flip_1ib_dir=flow_1ib_dir+"_flip"
gen_flip_flow_dir(flow_1ib_dir, flow_flip_1ib_dir)
