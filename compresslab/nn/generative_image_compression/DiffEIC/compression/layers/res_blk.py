import torch.nn as nn
import torch.nn.functional as F
from compresslab.core.layers import conv3x3, conv1x1

class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch, out_ch, stride=2, inplace=False):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=inplace)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        if stride != 1:
            self.downsample = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return out

def subpel_conv1x1(in_ch, out_ch, r=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r)
    )

class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch, out_ch, upsample=2, inplace=False):
        super().__init__()
        self.subpel_conv = subpel_conv1x1(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=inplace)
        self.conv = conv3x3(out_ch, out_ch)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.upsample = subpel_conv1x1(in_ch, out_ch, upsample)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.leaky_relu2(out)
        identity = self.upsample(x)
        out = out + identity
        return out
    
class ResidualBottleneck(nn.Module):
    def __init__(self, N=192, act=nn.GELU) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            conv1x1(N, N // 2),
            act(),
            nn.Conv2d(N // 2, N // 2, kernel_size=3, stride=1, padding=1),
            act(),
            conv1x1(N // 2, N)
        )

    def forward(self, x):
        out = x + self.branch(x)
        return out

class SFT(nn.Module):
    def __init__(self, x_nc, prior_nc, ks=3, nhidden=128):
        super().__init__()
        
        pw = ks // 2
        
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(prior_nc, nhidden, kernel_size=ks, padding=pw),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)
        
    def forward(self, x, ref):
        ref = F.adaptive_avg_pool2d(ref, x.size()[2:])
        actv = self.mlp_shared(ref)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = x * (1 + gamma) + beta
        return out
    
class SFTResidualBlock(nn.Module):
    def __init__(self, x_nc, prior_nc, ks=3):
        super().__init__()
        self.conv_0 = nn.Conv2d(x_nc, x_nc, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(x_nc, x_nc, kernel_size=3, padding=1)
        
        self.norm_0 = SFT(x_nc, prior_nc, ks=ks)
        self.norm_1 = SFT(x_nc, prior_nc, ks=ks)
        
    def forward(self, x, ref):
        dx = self.conv_0(self.actvn(self.norm_0(x, ref)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, ref)))
        out = x + dx
        
        return out
    
    def actvn(self, x):
        return F.leaky_relu(x, 0.2, inplace=True)