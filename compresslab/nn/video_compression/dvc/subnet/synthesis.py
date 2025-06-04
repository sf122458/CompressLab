from compresslab.core.layers import GDN
import math
import torch
import torch.nn as nn

class Synthesis_net(nn.Module):
    """
    Decode residual. [B, out_channel_M, H/16, W/16] -> [B, 3, H, W]
    """
    def __init__(self, out_channel_N=64, out_channel_M=96):
        super().__init__()
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

        self.deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.igdn1 = GDN(out_channel_N, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.igdn2 = GDN(out_channel_N, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.igdn3 = GDN(out_channel_N, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(out_channel_N, 3, 5, stride=2, padding=2, output_padding=1)

        self.initialize()
        

    def initialize(self):
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1 * (self.out_channel_M + self.out_channel_N) / (self.out_channel_M + self.out_channel_M))))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, (math.sqrt(2 * 1 * (self.out_channel_N + 3) / (self.out_channel_N + self.out_channel_N))))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)

    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.igdn3(self.deconv3(x))
        x = self.deconv4(x)
        return x