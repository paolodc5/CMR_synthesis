import torch
from torch import nn
from torchinfo import summary

# Generator block is a unet advanced modified with dropout

# Repeated code! to be moved in a common script
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
    
class DiscriminatorModel(nn.Module):
    def __init__(self, in_ch=4, base_ch=64, use_fc=True):
        super().__init__()
        self.use_fc = use_fc

        self.conv1 = ConvNormAct(in_ch, base_ch, kernel_size=3, stride=2, padding=1, norm=False, act=nn.GELU)
        self.conv2 = ConvNormAct(base_ch, base_ch*2, kernel_size=3, stride=2, padding=1, norm=nn.BatchNorm2d, act=nn.GELU)
        self.conv3 = ConvNormAct(base_ch*2, base_ch*4, kernel_size=3, stride=2, padding=1, norm=nn.BatchNorm2d, act=nn.GELU)
        self.conv4 = ConvNormAct(base_ch*4, base_ch*8, kernel_size=3, stride=2, padding=1, norm=nn.BatchNorm2d, act=nn.GELU)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_ch*8, 1)
        self.final_conv = nn.Conv2d(base_ch*8, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        if self.use_fc:
            x = self.ap(x)
            x = x.view(x.shape[0], -1)
            out = self.fc(x)
        else:
            out = self.final_conv(x)

        return out



if __name__ == "__main__":
    discr = DiscriminatorModel(in_ch=4, base_ch=64, use_fc=False)
    summary(discr, input_size=(2, 4, 256, 256))

