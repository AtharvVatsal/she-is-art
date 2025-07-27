# transformer_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, k, stride):
        super().__init__()
        pad = k // 2
        self.reflection_pad = nn.ReflectionPad2d(pad)
        self.conv2d = nn.Conv2d(in_c, out_c, k, stride)

    def forward(self, x):
        x = self.reflection_pad(x)
        return self.conv2d(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # track_running_stats=True to match old checkpoints
        self.conv1 = ConvLayer(channels, channels, 3, 1)
        self.in1   = nn.InstanceNorm2d(channels, affine=True, track_running_stats=True)
        self.conv2 = ConvLayer(channels, channels, 3, 1)
        self.in2   = nn.InstanceNorm2d(channels, affine=True, track_running_stats=True)
        self.relu  = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + residual

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_c, out_c, k, stride, upsample=None):
        super().__init__()
        self.upsample = upsample
        pad = k // 2
        self.reflection_pad = nn.ReflectionPad2d(pad)
        self.conv2d = nn.Conv2d(in_c, out_c, k, stride)

    def forward(self, x):
        if self.upsample:
            # ensure scale_factor is float for JIT
            x = F.interpolate(x, mode='nearest', scale_factor=float(self.upsample))
        x = self.reflection_pad(x)
        return self.conv2d(x)

class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        # initial convolution layers
        self.conv1 = ConvLayer(3,  32, 9, 1)
        self.in1   = nn.InstanceNorm2d(32, affine=True, track_running_stats=True)
        self.conv2 = ConvLayer(32, 64, 3, 2)
        self.in2   = nn.InstanceNorm2d(64, affine=True, track_running_stats=True)
        self.conv3 = ConvLayer(64,128, 3, 2)
        self.in3   = nn.InstanceNorm2d(128, affine=True, track_running_stats=True)
        self.relu  = nn.ReLU()

        # residual blocks
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # upsampling layers (must be named up1, up2)
        self.up1   = UpsampleConvLayer(128, 64, 3, 1, upsample=2)
        self.in4   = nn.InstanceNorm2d(64, affine=True, track_running_stats=True)
        self.up2   = UpsampleConvLayer(64,  32, 3, 1, upsample=2)
        self.in5   = nn.InstanceNorm2d(32, affine=True, track_running_stats=True)
        # final conv must be named conv4
        self.conv4 = ConvLayer(32,  3, 9, 1)

    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))

        y = self.res1(y); y = self.res2(y)
        y = self.res3(y); y = self.res4(y); y = self.res5(y)

        y = self.relu(self.in4(self.up1(y)))
        y = self.relu(self.in5(self.up2(y)))
        return self.conv4(y)
