import torch
from .perfrequency_convolution import PerFrequencyConvolution


class CoherentFrequencyResampler(torch.nn.Module):
    def __init__(self, in_channels, out_channels, transposed=False, bias=True):
        super(CoherentFrequencyResampler, self).__init__()
        self.filter = PerFrequencyConvolution(in_channels, out_channels, kernel_size=2, stride=2, padding=1, bias=bias, transposed=transposed)

    def forward(self, x):
        return self.filter(x)
