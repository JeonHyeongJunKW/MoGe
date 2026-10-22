"""Optional original-RGB features added to the finer MoGe neck outputs."""
from torch import nn
from torch.nn import functional as F


class RGBDetailBranch(nn.Module):
    def __init__(self, feature_channels, channels=16):
        super().__init__()
        if type(channels) is not int or channels < 1:
            raise ValueError('rgb_detail.channels must be a positive integer')
        if len(feature_channels) < 3:
            raise ValueError('RGB detail fusion requires at least three neck levels')
        # Half, quarter and eighth input resolution; fuse into the three finest
        # neck outputs. No batch statistics, so small training batches are safe.
        self.stages = nn.ModuleList()
        for in_channels in (3, channels, channels):
            self.stages.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1,
                          groups=in_channels),
                nn.Conv2d(in_channels, channels, 1), nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                nn.Conv2d(channels, channels, 1), nn.GELU(),
            ))
        self.levels = list(range(len(feature_channels) - 1, len(feature_channels) - 4, -1))
        self.projections = nn.ModuleList([
            nn.Conv2d(channels, feature_channels[level], 1) for level in self.levels
        ])
        # Preserve the pretrained prediction when enabling a new branch.
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, image, features):
        result = list(features)
        x = image * 2 - 1
        for stage, projection, level in zip(self.stages, self.projections, self.levels):
            x = stage(x)
            detail = F.interpolate(x, size=features[level].shape[-2:],
                                   mode='bilinear', align_corners=False)
            result[level] = features[level] + projection(detail).to(features[level].dtype)
        return result
