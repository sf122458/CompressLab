import math
import torch
import torch.nn as nn

class Analysis_prior_net(nn.Module):
    '''
    Compress residual prior
    '''
    def __init__(self, out_channel_N=64, out_channel_M=96):
        super().__init__()
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

        self.conv1 = nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        
        self.initialize()

    def initialize(self):
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (self.out_channel_M + self.out_channel_N) / (self.out_channel_M + self.out_channel_M))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)


    def forward(self, x):
        x = torch.abs(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.conv3(x)