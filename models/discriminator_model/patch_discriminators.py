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


class Discriminator_patch(nn.Module):
    def __init__(self):
        super(Discriminator_patch, self).__init__()
        #number of channels in training images
        #in grayscale images OR in masks, only one channel
        nc = 1
        # number of feature maps in discriminator
        ndf = 64

        self.conv1 = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 256 x 256; receptive field 1x1
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 128 x 128; receptive field 16x16
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 64 x 64; receptive field 34x34
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 6, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 6),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*6) x 32 x 32; receptive field 70x70
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(ndf * 6, ndf * 8, 8, 4, 2, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 7 x 7
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 7, 1, 0, bias=False),
            # state size. 1 x 1 x 1; receptive field 256x256
            # nn.Sigmoid()
        )

        self.reduce_channel_recep1x1 = nn.Sequential(
            nn.Conv2d(ndf, 1, 1, 1, 0, bias=False),
            # nn.Sigmoid()
        )

        self.reduce_channel_recep16x16 = nn.Sequential(
            nn.Conv2d(ndf * 2, 1, 1, 1, 0, bias=False),
            # nn.Sigmoid()
        )

        self.reduce_channel_recep70x70 = nn.Sequential(
            nn.Conv2d(ndf * 6, 1, 1, 1, 0, bias = False),
            # nn.Sigmoid()
        )

    def forward(self, input):
        output_recep1x1 = self.conv1(input)
        output_recep16x16 = self.conv2(output_recep1x1)
        intermediate1 = self.conv3(output_recep16x16)
        output_recep70x70 = self.conv4(intermediate1)
        intermediate2 = self.conv5(output_recep70x70)
        output_recep256x256 = self.conv6(intermediate2)

        output_recep1x1 = self.reduce_channel_recep1x1(output_recep1x1)
        output_recep16x16 = self.reduce_channel_recep16x16(output_recep16x16)
        output_recep70x70 = self.reduce_channel_recep70x70(output_recep70x70)

        return output_recep1x1, output_recep16x16, \
            output_recep70x70, output_recep256x256



class Discriminator_1x1_output(nn.Module):
    def __init__(self):
        super(Discriminator_1x1_output, self).__init__()
        #number of channels in training images
        #in grayscale images OR in masks, only one channel
        nc = 1
        # number of feature maps in discriminator
        ndf = 64
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 8, 4, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 8, 4, 2, bias=False),
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
            nn.Conv2d(ndf, ndf*2, 8, 4, 2, bias=False),
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


class Discriminator_256x256_output(nn.Module):
    def __init__(self):
        super(Discriminator_256x256_output, self).__init__()
        #number of channels in training images
        #in grayscale images OR in masks, only one channel
        nc = 1
        # number of feature maps in discriminator
        ndf = 64
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 256 x 256
            nn.Conv2d(ndf, ndf * 2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 256 x 256
            nn.Conv2d(ndf * 2, ndf * 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 256 x 256
            nn.Conv2d(ndf * 4, ndf * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 256 x 256
            nn.Conv2d(ndf * 8, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )        

    def forward(self, input):
        return self.main(input)


class Discriminator_30x30_output(nn.Module):
    def __init__(self):
        super(Discriminator_30x30_output, self).__init__()
        #number of channels in training images
        #in grayscale images OR in masks, only one channel
        nc = 1
        # number of feature maps in discriminator
        ndf = 64
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 256 x 256
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 256 x 256
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 256 x 256
            nn.Conv2d(ndf * 8, 1, 4, 1, 1, bias=False),
            nn.Sigmoid()
        )        

    def forward(self, input):
        return self.main(input)

    
class Discriminator_252x252_output(nn.Module):
    def __init__(self):
        super(Discriminator_252x252_output, self).__init__()
        #number of channels in training images
        #in grayscale images OR in masks, only one channel
        nc = 1
        # number of feature maps in discriminator
        ndf = 64
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 256 x 256
            nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 256 x 256
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 256 x 256
            nn.Conv2d(ndf * 8, 1, 4, 1, 1, bias=False),
            nn.Sigmoid()
        )        

    def forward(self, input):
        return self.main(input)


