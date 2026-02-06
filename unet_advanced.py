import torch
from torch import nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    """
    Layer grouping a convolution, normalization and activation funtion
    normalization includes BN and IN
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0,
                groups=1, dilation=1, bias=False, norm=nn.BatchNorm2d, act=nn.ReLU, preact=False):

        super().__init__()
        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        self.conv = nn.Conv2d(
            in_channels=in_ch, 
            out_channels=out_ch, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=bias
        )
        if preact:
            self.norm = norm(in_ch) if norm else nn.Identity()
        else:
            self.norm = norm(out_ch) if norm else nn.Identity()
        self.act = act() if act else nn.Identity()
        self.preact = preact

    def forward(self, x):
        
        if self.preact:
            out = self.conv(self.act(self.norm(x)))
        else:
            out = self.act(self.norm(self.conv(x)))

        return out
    
class BasicBlock(nn.Module):
    """ 
    Block with two ConvNormAct layers and a residual connection. 
    If stride != 1 or in_ch != out_ch, the residual connection is adapted with a ConvNormAct layer. 
    """
    def __init__(self, in_ch, out_ch, stride=1, norm=nn.BatchNorm2d, act=nn.ReLU, preact=True):
        super().__init__()
        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        self.conv1 = ConvNormAct(in_ch, out_ch, 3, stride=stride, padding=1, norm=norm, act=act, preact=preact)
        self.conv2 = ConvNormAct(out_ch, out_ch, 3, stride=1, padding=1, norm=norm, act=act, preact=preact)

        self.shortcut = nn.Sequential()

        # in_ch != out_ch in most of the cases: channel adaptation is needed for the residual connection
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch, out_ch, 3, stride=stride, padding=1, norm=norm, act=act, preact=preact)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += self.shortcut(residual)

        return out
    
class down_block(nn.Module):
    """
    Dpownsampling block with optional pooling followed by multiple BasicBlocks.
    """
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock, pool=False):
        super().__init__()

        block_list = []

        if pool:
            block_list.append(nn.MaxPool2d(2))
            block_list.append(block(in_ch, out_ch))
        else:
            block_list.append(block(in_ch, out_ch, stride=2)) # stride 2 for downsampling without pooling

        for _ in range(num_block-1):
            block_list.append(block(out_ch, out_ch, stride=1))

        self.conv = nn.Sequential(*block_list)
    def forward(self, x): 
        return self.conv(x)

class up_block(nn.Module):
    """
    Upsampling block with bilinear upsampling + channel adaptation, concatenation with skip connection followed by multiple BasicBlocks.
    """
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock):
        super().__init__()

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        block_list = []
        block_list.append(block(2*out_ch, out_ch))


        for _ in range(num_block-1):
            block_list.append(block(out_ch, out_ch))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        # upsampling with bilinear interpolation, then channel adaptation with 1x1 conv
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x1 = self.conv_ch(x1)

        # concatenation and block convolutions
        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)

        return out

class UNetAdvanced(nn.Module):
    """
    U-Net advanced and flexible architecture with BasicBlocks, optional pooling and configurable depth.
    More blocks can be added by modifying the num_block parameter in down_block and up_block instantiations.
    """
    def __init__(self, in_ch, num_classes, base_ch=64, block='BasicBlock', pool=False):
        super().__init__()
        
        assert block in ['BasicBlock']
        block = BasicBlock

        nb = 2 # num basic blocks for each down/up block

        self.inc = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1, bias=False)
        self.inblock =  block(base_ch, base_ch)

        self.down1 = down_block(base_ch, 2*base_ch, num_block=nb, block=block,  pool=pool)
        self.down2 = down_block(2*base_ch, 4*base_ch, num_block=nb, block=block, pool=pool)
        self.down3 = down_block(4*base_ch, 8*base_ch, num_block=nb, block=block, pool=pool)
        self.down4 = down_block(8*base_ch, 16*base_ch, num_block=nb, block=block, pool=pool)

        self.up1 = up_block(16*base_ch, 8*base_ch, num_block=nb, block=block)
        self.up2 = up_block(8*base_ch, 4*base_ch, num_block=nb, block=block)
        self.up3 = up_block(4*base_ch, 2*base_ch, num_block=nb, block=block)
        self.up4 = up_block(2*base_ch, base_ch, num_block=nb, block=block)

        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        #x = torch.cat([img, c], 1)
                
        x1 = self.inc(x) # input conv, no downsampling
        x1 = self.inblock(x1) # initial conv blocks, no downsampling
        x2 = self.down1(x1) # down1 with downsampling
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)


        out = self.up1(x5, x4) 
        out = self.up2(out, x3) 
        out = self.up3(out, x2) 
        out = self.up4(out, x1) 
        out = self.outc(out)

        return out 