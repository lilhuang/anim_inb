from typing import Optional, Union, List
from .rrdb import RRDB
from .decoder import UnetDecoder
from ..encoders import get_encoder
from ..base import SegmentationModel
from ..base import SegmentationHead, ClassificationHead
from ..base import initialization as init
import torch.nn as nn

import pdb


class Unet_RRDB(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "rrdb",
        classes: int = 1,
        ###### ^only changed/used these two ##########
        kernel_size: int = 1,
        in_channels: int = 3,
        intermed_channels: int = 64,
        ###### ^added these two ###########
    ):
        super().__init__()

        self.widen = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=intermed_channels,
                      kernel_size=kernel_size
            ),
            nn.LeakyReLU()
        )

        self.encoder = RRDB(
            kernel_size=kernel_size,
            channels=intermed_channels
        )

        self.unwiden = nn.Sequential(
            nn.Conv2d(in_channels=intermed_channels,
                      out_channels=classes,
                      kernel_size=kernel_size),
            nn.LeakyReLU()
        )

        self.name = "UNet_RRDB"



    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        x_widen = self.widen(x)
        features = self.encoder(x_widen)
        masks = self.unwiden(features)
        # pdb.set_trace()

        return masks
