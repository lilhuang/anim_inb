# final SU seq reverse augmentation prepatation 
#  image path - /fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog/StevenHug_2048x1024/frame066.png
# flow1ib - /fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_2048x1024_1ib/frame066_to_frame070.npz
# flow3ib - /fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_2048x1024/frame066_to_frame070.npz
# csv path - /fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_smol_sequential_halfhalf_2_csv/train_triplets_3ib.c





import os
import numpy as np
from tqdm import tqdm
import cv2



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
            # print(type(flow_npz['flo13']), type(flo13))
            np.savez_compressed(flip_flow_file_path, flo13=flo13, flo31=flo31)

        #fliperse the frame numbers in the flow file name
        suffix = "flipv"
        #save the flipersed flow file
        def flipv_flow(flow):
            flow = np.transpose(flow, (1,2,0)) #[2, H, W]->[H, W, 2]
            flow = cv2.flip(flow, 0) #[H, W, 2]
            flow = np.transpose(flow, (2,0,1)) #[H, W, 2]->[2, H, W]
            flow[1]=flow[1]*-1 #dy becomes -dy
            return flow
        process_flow(suffix, flipv_flow)
        
        suffix = "fliph"
        #save the flipersed flow file
        def fliph_flow(flow):
            flow = np.transpose(flow, (1,2,0)) #[2, H, W]->[H, W, 2]
            flow = cv2.flip(flow, 1) #[H, W, 2]
            flow = np.transpose(flow, (2,0,1)) #[H, W, 2]->[2, H, W]
            flow[0]=flow[0]*-1 #dx becomes -dx
            return flow
        process_flow(suffix, fliph_flow)

        suffix = "flipvh"
        #save the flipersed flow file
        def flipvh_flow(flow):
            flow = np.transpose(flow, (1,2,0)) #[2, H, W]->[H, W, 2]
            flow = cv2.flip(flow, -1) #[H, W, 2]
            flow = np.transpose(flow, (2,0,1)) #[H, W, 2]->[2, H, W]
            flow=flow*-1 #dx,dy becomes -dx,-dy
            return flow
        process_flow(suffix, flipvh_flow)




# flow_3ib_dir = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_2048x1024"
# flow_flip_3ib_dir=flow_3ib_dir+"_flip"
# gen_flip_flow_dir(flow_3ib_dir, flow_flip_3ib_dir)

flow_1ib_dir = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_2048x1024_1ib"
flow_flip_1ib_dir=flow_1ib_dir+"_flip"
gen_flip_flow_dir(flow_1ib_dir, flow_flip_1ib_dir)


