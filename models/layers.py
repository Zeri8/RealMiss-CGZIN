import torch.nn as nn
import torch.nn.functional as F

def general_conv3d_prenorm(in_channels, out_channels, k_size=3, stride=1, padding=1, pad_type='reflect'):
    pad = nn.ReflectionPad3d(padding) if pad_type == 'reflect' else nn.ZeroPad3d(padding)
    return nn.Sequential(
        pad,
        nn.Conv3d(in_channels, out_channels, kernel_size=k_size, stride=stride),
        nn.InstanceNorm3d(out_channels),
        nn.GELU()
    )

class fusion_prenorm(nn.Module):
    def __init__(self, in_channel, num_cls):
        super().__init__()
        self.conv = nn.Conv3d(in_channel, in_channel, 1)
        self.norm = nn.InstanceNorm3d(in_channel)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))