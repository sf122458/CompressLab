from compresslab.core.layers import GDN
import math
import torch
import torch.nn as nn

class Analysis_net(nn.Module):
    '''
    Compress residual
    [B, 3, H, W] -> [B, out_channel_M, H/16, W/16]
    '''
    def __init__(self, out_channel_N=64, out_channel_M=96):
        super().__init__()
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

        self.conv1 = nn.Conv2d(3, out_channel_N, 5, stride=2, padding=2)
        self.gdn1 = GDN(out_channel_N)
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.gdn2 = GDN(out_channel_N)
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.gdn3 = GDN(out_channel_N)
        self.conv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)
        
        self.initialize()

    def initialize(self):
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + self.out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (self.out_channel_M + self.out_channel_N) / (self.out_channel_N + self.out_channel_N))))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        return self.conv4(x)