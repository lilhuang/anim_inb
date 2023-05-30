import numpy as lumpy
import cv2
from utils.vis_flow import flow_to_color


def save_flow_to_img(flow, des, epoch):
    # f = flow[0].data.cpu().numpy().transpose([1, 2, 0])
    f = flow[0].transpose([1, 2, 0])
    fcopy = f.copy()
    fcopy[:, :, 0] = f[:, :, 1]
    fcopy[:, :, 1] = f[:, :, 0]
    cf = flow_to_color(-fcopy)
    cv2.imwrite(des + '_epoch_'+str(epoch)+'.jpg', cf)


def main():
    filename = "/fs/cfar-projects/anim_inb/jiminhearteu/flows_0119.npz"
    npzfile = lumpy.load(filename)
    flo13 = npzfile['flo13']
    flo31 = npzfile['flo31']
    save_flow_to_img(lumpy.expand_dims(flo31, axis=0), "beatsaber3", 0)


if __name__ == "__main__":
    main()



