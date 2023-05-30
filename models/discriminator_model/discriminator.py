import torch.nn as nn
from .convolutional_filter_manifold import ConvolutionalFilterManifold
from .coherent_frequency_block import CoherentFrequencyBlock
from .weight_init import weight_init
from torch.nn.utils import spectral_norm

import pdb


def dcgan_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# class Discriminator(torch.nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()

#         self.block_y = ConvolutionalFilterManifold(1, 64, 8, 8)
#         self.block_cb = ConvolutionalFilterManifold(1, 64, 8, 8)
#         self.block_cr = ConvolutionalFilterManifold(1, 64, 8, 8)

#         self.block_clean = torch.nn.Conv2d(3, 192, 8, 8)

#         self.net = torch.nn.Sequential(            
#             torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),

#             spectral_norm(torch.nn.Conv2d(384, 128, 1, 1, 0, bias=True)), 
#             torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),

#             spectral_norm(torch.nn.Conv2d(128, 256, 1, 1, 0, bias=True)), 
#             torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
 
#             spectral_norm(torch.nn.Conv2d(256, 512, 1, 1, 0, bias=True)), 
#             torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),

#             spectral_norm(torch.nn.Conv2d(512, 1, 1, 1, 0, bias=True))
#         )

#         self.apply(dcgan_weights_init)

#     def forward(self, clean, compressed, q_y, q_c):
#         y = self.block_y(q_y, compressed[:, 0:1, :, :])
#         cb = self.block_cb(q_c, compressed[:, 1:2, :, :])
#         cr = self.block_cr(q_c, compressed[:, 2:3, :, :])

#         clean_blocks = self.block_clean(clean)

#         return self.net(torch.cat([clean_blocks, y, cb, cr], dim=1)).view(-1, 1)

class Discriminator_non_square(nn.Module):
    def __init__(self):
        super(Discriminator_non_square, self).__init__()
        #number of channels in training images
        #in grayscale images OR in masks, only one channel
        nc = 1
        # number of feature maps in discriminator
        ndf = 64
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, (8, 16), (4, 8), (2, 4), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator_non_square_dumb(nn.Module):
    def __init__(self):
        super(Discriminator_non_square_dumb, self).__init__()
        #number of channels in training images
        #in grayscale images OR in masks, only one channel
        nc = 1
        # number of feature maps in discriminator
        ndf = 32
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, (8, 16), (4, 8), (2, 4), bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 8, 4, 2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 2, 0, bias=True),
        )
        # self.first = nn.Sequential(
        #     nn.Conv2d(nc, ndf, (8, 16), (4, 8), (2, 4), bias=True),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        # self.second = nn.Sequential(
        #     nn.Conv2d(ndf, ndf * 2, 8, 4, 2, bias=False),
        #     nn.BatchNorm2d(ndf * 2),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        # self.third = nn.Sequential(
        #     nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        # self.fourth = nn.Sequential(
        #     nn.Conv2d(ndf * 4, 1, 4, 2, 0, bias=True),
        # )

    def forward(self, input):
        return self.main(input)


class Discriminator_square(nn.Module):
    def __init__(self):
        super(Discriminator_square, self).__init__()
        #number of channels in training images
        #in grayscale images OR in masks, only one channel
        nc = 1
        # number of feature maps in discriminator
        ndf = 64
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 8, 4, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(nc, ndf, 8, 4, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)





class Discriminator_square_dumb(nn.Module):
    #essentially the same as the square but with fewer features
    #and fewer layers
    def __init__(self):
        super(Discriminator_square_dumb, self).__init__()
        #number of channels in training images
        #in grayscale images OR in masks, only one channel
        nc = 1
        # number of feature maps in discriminator
        ndf = 32

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 8, 4, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 8, 2, 0, bias=False),
            # state size. (ndf*4) x 8 x 8
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

