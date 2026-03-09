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

        if  self.use_fc:
            self.ap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(base_ch*8, 1)
        else:
            self.final_conv = nn.Conv2d(base_ch*8, 1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        if self.use_fc:
            x5 = self.ap(x4)
            x6 = x5.view(x5.shape[0], -1)
            out = self.fc(x6)
        else:
            out = self.final_conv(x4)

        return out


class DiscriminatorPatchGAN(nn.Module):
    def __init__(self, in_ch=6, base_ch=64, n_layers=3):
        super(DiscriminatorPatchGAN, self).__init__()
        self.blocks = nn.ModuleList()

        self.blocks.append(
            nn.Sequential(
                nn.Conv2d(in_ch, base_ch, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )

        curr_ch = base_ch
        for n in range(1, n_layers):
            prev_ch = curr_ch
            curr_ch = min(base_ch * (2**n), 512)
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(prev_ch, curr_ch, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(curr_ch),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        
        prev_ch = curr_ch
        curr_ch = min(base_ch * (2**n_layers), 512)
        self.blocks.append(
            nn.Sequential(
                nn.Conv2d(prev_ch, curr_ch, kernel_size=4, stride=1, padding=1),
                nn.InstanceNorm2d(curr_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )

        self.blocks.append(nn.Conv2d(curr_ch, 1, kernel_size=4, stride=1, padding=1))
    
    def forward(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features[-1], features[:-1]


# if __name__ == "__main__":
#     dummy_input = torch.randn(2, 6, 384, 384)
#     discr = DiscriminatorPatchGAN(in_ch=6, base_ch=64, n_layers=3)
#     out = discr(dummy_input)
#     print(len(out))
#     print(out[0].shape)

#     criterion_l1 = nn.L1Loss()


#     for i in out[1]:
#         print(i.shape)

#     # generate fake output for testing
#     out_fake = []
#     for i in out[1]:
#         out_fake.append(torch.randn_like(i, requires_grad=True))
#         print(out_fake[-1].shape)

#     # out_fake.append(torch.randn_like(out[0], requires_grad=True))

#     perc_loss = compute_perceptual_loss(out[1], out_fake, criterion_l1)
#     print(perc_loss)
#     optimizer = torch.optim.Adam(discr.parameters(), lr=1e-4)
#     optimizer.zero_grad()
#     perc_loss.backward()
#     optimizer.step()

#     print(f"Perceptual Loss: {perc_loss.item()}")
#     print('gradients: ')
#     for name, param in discr.named_parameters():
#         if param.grad is not None:
#             print(f"{name}: {param.grad.norm().item()}")