import torch.nn

class PerFrequencyConvolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, transposed=False, groups=1, bias=True):
        super(PerFrequencyConvolution, self).__init__()

        if transposed:
            self.filter = torch.nn.ConvTranspose2d(in_channels=in_channels * 64, 
                                      out_channels=out_channels * 64, 
                                      kernel_size=kernel_size,
                                      padding=padding,
                                      stride=stride,
                                      dilation=dilation,
                                      groups=64,
                                      bias=bias)
        else:
            self.filter = torch.nn.Conv2d(in_channels=in_channels * 64, 
                                        out_channels=out_channels * 64, 
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        stride=stride,
                                        dilation=dilation,
                                        groups=64,
                                        bias=bias)
        
    def forward(self, x):
        return self.filter(x)
