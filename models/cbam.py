import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Emphasizes informative feature channels using pooled spatial descriptors."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden_dim = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared 1x1 conv bottleneck acts as a lightweight MLP across channels
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_weights = self.shared_mlp(self.avg_pool(x))
        max_weights = self.shared_mlp(self.max_pool(x))
        return x * self.gate(avg_weights + max_weights)


class SpatialAttention(nn.Module):
    """Focuses receptive field on fragmented object perimeters over belt background."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial_desc = torch.cat([avg_map, max_map], dim=1)
        return x * self.gate(self.conv(spatial_desc))


class CBAM(nn.Module):
    """Sequential Channel and Spatial Attention block (Woo et al., ECCV 2018)."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction=reduction)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x