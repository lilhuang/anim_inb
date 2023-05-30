import torch.nn
from .coherent_frequency_resampler import CoherentFrequencyResampler
from .perfrequency_convolution import PerFrequencyConvolution


class CoherentFrequencyBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, transposed=False, resample=False, activation=torch.nn.LeakyReLU, bias=True):
        super(CoherentFrequencyBlock, self).__init__()

        if resample:
            self.conv1 = CoherentFrequencyResampler(in_channels=in_channels, out_channels=out_channels, transposed=transposed, bias=bias)
        else:
            self.conv1 = PerFrequencyConvolution(in_channels=in_channels, out_channels=out_channels, bias=bias)

        self.conv2 = PerFrequencyConvolution(in_channels=out_channels, out_channels=out_channels, bias=bias)

        if resample:
            self.resampler = CoherentFrequencyResampler(in_channels=in_channels, out_channels=out_channels, transposed=transposed, bias=bias)
        else:
            self.resampler = None

        self.a1 = activation()
        self.a2 = activation()

    def forward(self, x):
        out = self.conv1(x)
        out = self.a1(out)
        out = self.conv2(out)

        if self.resampler is not None:
            input = self.resampler(x)
        else:
            input = x

        out += input
        out = self.a2(out)

        return out
