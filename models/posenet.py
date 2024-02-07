import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Convolutional Block with Batch Normalization and ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class HourglassModule(nn.Module):
    """A simplified hourglass module for demonstration purposes."""
    def __init__(self, in_channels, out_channels):
        super(HourglassModule, self).__init__()
        # Simplified hourglass: just downsampling then upsampling for illustration
        self.down = ConvBlock(in_channels, out_channels, 3, 2, 1)  # Downsample
        self.up = nn.ConvTranspose2d(out_channels, out_channels, 3, 2, 1, output_padding=1)  # Upsample

    def forward(self, x):
        down = self.down(x)
        return self.up(down)

class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, **kwargs):
        super(PoseNet, self).__init__()
        self.nstack = nstack

        # Initial preprocessing layers
        self.pre = nn.Sequential(
            ConvBlock(1, 64, 7, 2, 3),
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, inp_dim),
        )

        # Stacked hourglass modules for deep feature extraction
        self.hgs = nn.ModuleList([
            HourglassModule(inp_dim, inp_dim) for _ in range(nstack)
        ])

        # Output layers for converting spatial info to coordinate predictions
        self.outs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(inp_dim, oup_dim),
            ) for _ in range(nstack)
        ])

        # Inter-stack feature merging, ensuring dimensional compatibility
        self.merge_features = nn.ModuleList([
            ConvBlock(inp_dim + oup_dim, inp_dim, 1, 1, 0) for _ in range(nstack-1)
        ])

    def forward(self, imgs):
        x = self.pre(imgs)
        combined_preds = []

        for i in range(self.nstack):
            hg = self.hgs[i](x)

            # Transition to predictions
            pooled = F.adaptive_avg_pool2d(hg, (1, 1))
            flat = pooled.view(pooled.size(0), -1)
            preds = self.outs[i](flat)
            combined_preds.append(preds)

            if i < self.nstack - 1:
                # Prepare for feature merging
                preds_reshaped = preds.view(preds.size(0), -1, 1, 1).expand(-1, -1, hg.size(2), hg.size(3))
                merged = torch.cat([hg, preds_reshaped], dim=1)
                x = self.merge_features[i](merged)

        return torch.stack(combined_preds, dim=1)
