import torch.nn


class DenseBlock(torch.nn.Module):
    def __init__(self, kernel_size, channels, conv_op, padding=0, stride=1, dilation=1, groups=1, scaler=0.2, bias=True, activation=torch.nn.LeakyReLU):
        super(DenseBlock, self).__init__()

        self.scaler = scaler

        self.conv1 = conv_op(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.relu1 = activation()
        self.conv2 = conv_op(in_channels=channels * 2, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.relu2 = activation()
        self.conv3 = conv_op(in_channels=channels * 3, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups,  bias=bias)
        self.relu3 = activation()
        self.conv4 = conv_op(in_channels=channels * 4, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.relu4 = activation()
        self.conv5 = conv_op(in_channels=channels * 5, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.relu5 = activation()

    def forward(self, x):
        x1 = self.relu1(self.conv1(x))
        x2 = self.relu2(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.relu3(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.relu4(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.relu5(self.conv5(torch.cat((x, x1, x2, x3, x4), 1)))
        return x5 * self.scaler + x
