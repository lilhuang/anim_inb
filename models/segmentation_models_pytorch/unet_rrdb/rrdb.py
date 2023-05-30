import torch.nn
from .dense_block import DenseBlock

import pdb


class RRDB(torch.nn.Module):
    def __init__(self, kernel_size, channels, conv_op=torch.nn.Conv2d, padding=0, stride=1, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU):
        super(RRDB, self).__init__()
        self.scaler = 0.2

        self.block1 = DenseBlock(kernel_size=kernel_size, channels=channels, conv_op=conv_op, stride=stride, padding=padding, dilation=dilation, groups=groups, scaler=self.scaler, bias=bias, activation=activation)
        self.block2 = DenseBlock(kernel_size=kernel_size, channels=channels, conv_op=conv_op, stride=stride, padding=padding, dilation=dilation, groups=groups, scaler=self.scaler, bias=bias, activation=activation)
        self.block3 = DenseBlock(kernel_size=kernel_size, channels=channels, conv_op=conv_op, stride=stride, padding=padding, dilation=dilation, groups=groups, scaler=self.scaler, bias=bias, activation=activation)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        return out * self.scaler + x
