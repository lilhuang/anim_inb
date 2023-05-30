import argparse
import torch
from torchvision.transforms import transforms
from PIL import Image

from RAFT.core.raft import RAFT

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--model', help="restore checkpoint")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
args = parser.parse_args()

raftmodelpath  = "/fs/cfar-projects/anim_inb/RAFT/models/raft-things.pth"

raftmodel      = torch.nn.DataParallel(RAFT(args))
raftmodel.load_state_dict(torch.load(raftmodelpath))

raftmodel = raftmodel.module
raftmodel.cuda()
raftmodel.eval()


frame1_path = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_dog_patches_large/1ib/t_bezier_s_facing_bezier_r_none_0_png_2048x1024_64_640/frame1.png"
frame3_path = "/fs/cfar-projects/anim_inb/datasets/Blender_cubes_dog_patches_large/1ib/t_bezier_s_facing_bezier_r_none_0_png_2048x1024_64_640/frame3.png"

pil1 = Image.open(frame1_path)
pil3 = Image.open(frame3_path)

pil1_rgb = pil1.convert("RGB")
pil3_rgb = pil3.convert("RGB")

to_tensor = transforms.PILToTensor()

frame1 = torch.unsqueeze(to_tensor(pil1).float().cuda(), 0)
frame3 = torch.unsqueeze(to_tensor(pil3).float().cuda(), 0)

frame1_rgb = torch.unsqueeze(to_tensor(pil1_rgb).float().cuda(), 0)
frame3_rgb = torch.unsqueeze(to_tensor(pil3_rgb).float().cuda(), 0)


with torch.no_grad():
    flow_low_13, flow_up_13 = raftmodel(frame1_rgb, frame3_rgb, iters=20, test_mode=True)

    pdb.set_trace()

    print("blah??")


