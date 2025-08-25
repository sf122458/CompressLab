import torch.nn as nn

class LocalContextEX(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.fusion = nn.Conv2d(in_dim, out_dim, kernel_size=5, stride=1, padding=2)

    def forward(self, local_params):
        local_params = self.fusion(local_params)
        return local_params

class ChannelContextEX(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.ReLU) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, 224, kernel_size=5, stride=1, padding=2),
            act(),
            nn.Conv2d(224, 128, kernel_size=5, stride=1, padding=2),
            act(),
            nn.Conv2d(128, out_dim, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, channel_params):
        """
        Args:
            channel_params(Tensor): [B, C * K, H, W]
        return:
            channel_params(Tensor): [B, C * 2, H, W]
        """
        channel_params = self.fusion(channel_params)

        return channel_params