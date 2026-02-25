import torch
from torch import nn
from torchinfo import summary

from gan_multidiscr import SingleDiscriminator

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


class DiscriminatorPatchGAN(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, n_layers=3):
        super(DiscriminatorPatchGAN, self).__init__()
        
        # Esser et al. (through Isola et al.) they use PatchGAN architecture.
        layers = [
            nn.Conv2d(in_ch, base_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Intermediate layers use istance norm instead of batch norm, as suggested by Isola et al.
        curr_ch = base_ch
        for n in range(1, n_layers):
            prev_ch = curr_ch
            curr_ch = min(base_ch * (2**n), 512)
            layers.extend([
                nn.Conv2d(prev_ch, curr_ch, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(curr_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ] )

        # Penultimate layer: stride 1 to maintain spatial dimensions
        prev_ch = curr_ch
        curr_ch = min(base_ch * (2**n_layers), 512)
        layers.extend([
            nn.Conv2d(prev_ch, curr_ch, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(curr_ch),
            nn.LeakyReLU(0.2, inplace=True)
        ])

        # Final layer: returns prediction map (real/fake) for each patch. No activation function
        layers.append(nn.Conv2d(curr_ch, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    discr = DiscriminatorPatchGAN(in_ch=6, base_ch=64, n_layers=3)
    summary(discr, input_size=(2, 6, 384, 384))

