# final SU seq reverse augmentation prepatation 
#  image path - /fs/cfar-projects/anim_inb/datasets/SU_24fps/preprocess_dog/StevenHug_2048x1024/frame066.png
# flow1ib - /fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_2048x1024_1ib/frame066_to_frame070.npz
# flow3ib - /fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_2048x1024/frame066_to_frame070.npz
# csv path - /fs/cfar-projects/anim_inb/datasets/SU_24fps/StevenHug_2048x1024_smol_sequential_halfhalf_2_csv/train_triplets_3ib.c





import os
import numpy as np



def gen_rev_flow_dir(flow_dir, rev_flow_dir):
    os.makedirs(rev_flow_dir, exist_ok=True)
    for flow_file in os.listdir(flow_dir):
        flow_file_path = os.path.join(flow_dir, flow_file)
        #load the flow file
        flow_npz = np.load(flow_file_path)

        #reverse the frame numbers in the flow file name
        rev_flow_file = flow_file.split("_to_")[1].split(".")[0]+"_to_"+flow_file.split("_to_")[0]+".npz"
        rev_flow_file_path = os.path.join(rev_flow_dir, rev_flow_file)

        #save the reversed flow file
        np.savez_compressed(rev_flow_file_path, flo13=flow_npz['flo31'], flo31=flow_npz['flo13'])



flow_3ib_dir = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_2048x1024"
flow_rev_3ib_dir=flow_3ib_dir+"_rev"
gen_rev_flow_dir(flow_3ib_dir, flow_rev_3ib_dir)

flow_1ib_dir = "/fs/cfar-projects/anim_inb/pips/generate_flows/flows_su_smol_2048x1024_1ib"
flow_rev_1ib_dir=flow_1ib_dir+"_rev"
gen_rev_flow_dir(flow_1ib_dir, flow_rev_1ib_dir)
