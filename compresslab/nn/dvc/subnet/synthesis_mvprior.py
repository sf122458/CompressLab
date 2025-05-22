import math
import torch
import torch.nn as nn


class Synthesis_mvprior_net(nn.Module):
    '''
    Decode residual prior
    '''
    def __init__(self, out_channel_N=64, out_channel_mv=128):
        super().__init__()
        self.out_channel_N = out_channel_N
        self.out_channel_mv = out_channel_mv
        
        self.deconv1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.relu1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.relu2 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_mv, 3, stride=1, padding=1)

        self.initialize()

    def initialize(self):
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, (math.sqrt(2 * 1 * (self.out_channel_mv + self.out_channel_N) / (self.out_channel_N + self.out_channel_N))))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        return torch.exp(self.deconv3(x))