
import torch
import torch.nn as nn
import torch.nn.functional as F



# benchmark UNet (simple, depth-configurable)

class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2"""
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upscaling then double conv (with skip connection)."""
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            # after concat we run DoubleConv; reduce channels with 1x1
            self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            # transposed conv reduces channels itself
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            self.reduce = nn.Identity()

        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.reduce(x)

        # Pad x to match skip in case of odd sizes
        diff_y = skip.size(-2) - x.size(-2)
        diff_x = skip.size(-1) - x.size(-1)
        if diff_x != 0 or diff_y != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetSimple(nn.Module):
    """Simple U-Net with configurable depth.

    Args:
        in_channels: input channels (e.g. 1 for grayscale)
        num_classes: output channels/classes (logits)
        base_channels: channels in first level (e.g. 16/32/64)
        depth: number of downsampling steps (>=1). Total levels = depth+1
        bilinear: use bilinear upsample (True) or transposed conv (False)
    """
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32, depth: int = 4, bilinear: bool = True):
        super().__init__()
        if depth < 1:
            raise ValueError('depth must be >= 1')

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.depth = depth
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, base_channels)

        # Encoder
        self.downs = nn.ModuleList()
        ch = base_channels
        for _ in range(depth):
            self.downs.append(Down(ch, ch * 2))
            ch *= 2

        # Decoder: mirror (depth up blocks)
        self.ups = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(Up(ch, ch // 2, bilinear=bilinear))
            ch //= 2

        self.outc = OutConv(base_channels, num_classes)

    def forward(self, x):
        skips = []
        x = self.inc(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = skips.pop()  # bottom
        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip)

        return self.outc(x)


