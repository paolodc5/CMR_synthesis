import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False) # standard normalization

        self.mlp_shared = nn.Sequential(
            spectral_norm(nn.Conv2d(label_nc, 128, kernel_size=3, padding=1)),
            nn.ReLU()
        ) # mask features

        self.mlp_gamma = spectral_norm(nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)) # learned scaling factor
        self.mlp_beta = spectral_norm(nn.Conv2d(128, norm_nc, kernel_size=3, padding=1))

    def forward(self, x, mask):
        normalized = self.param_free_norm(x)

        # resize needed for deeper layers
        mask_resized = F.interpolate(mask, size=x.size()[2:], mode='nearest')

        actv = self.mlp_shared(mask_resized)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        return normalized * (1 + gamma) + beta


class SPADEResBlk(nn.Module):
    def __init__(self, fin, fout, label_nc):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        
        # main branch
        self.norm_0 = SPADE(fin, label_nc)
        self.conv_0 = spectral_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        
        self.norm_1 = SPADE(fmiddle, label_nc)
        self.conv_1 = spectral_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        
        # skip connection
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)
            self.conv_s = spectral_norm(nn.Conv2d(fin, fout, kernel_size=3, padding=1, bias=False))

    def forward(self, x, mask):
        x_s = self.shortcut(x, mask)
        
        dx = self.conv_0(self.actvn(self.norm_0(x, mask)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, mask)))
        
        return x_s + dx

    def shortcut(self, x, mask):
        if self.learned_shortcut:
            x_s = self.conv_s(self.actvn(self.norm_s(x, mask)))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 0.2)


class SPADEGenerator(nn.Module):
    def __init__(self, label_nc, latent_dim=256, output_ch=3):
        super().__init__()
        self.label_nc = label_nc
        self.latent_dim = latent_dim

        self.init_H = 6
        self.init_W = 6
        self.fc = nn.Linear(latent_dim, 1024 * self.init_H * self.init_W)

        self.head_0 = SPADEResBlk(1024, 1024, label_nc)
        self.head_1 = SPADEResBlk(1024, 1024, label_nc)
        self.head_2 = SPADEResBlk(1024, 1024, label_nc)
        self.head_3 = SPADEResBlk(1024, 512, label_nc)
        self.head_4 = SPADEResBlk(512, 256, label_nc)
        self.head_5 = SPADEResBlk(256, 128, label_nc)
        self.head_6 = SPADEResBlk(128, 64, label_nc)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.conv_img = spectral_norm(nn.Conv2d(64, output_ch, kernel_size=3, padding=1))
        self.tanh = nn.Tanh()

    def forward(self, z, mask):
        # z: (Batch, latent_dim), mask: (Batch, label_nc, H, W)
        
        x = self.fc(z)
        x = x.view(-1, 1024, self.init_H, self.init_W) # (1024, 6, 6)

        x = self.head_0(x, mask)
        x = self.up(x)               # (1024, 12, 12)
        
        x = self.head_1(x, mask)
        x = self.up(x)               # (1024, 24, 24)
        
        x = self.head_2(x, mask)
        x = self.up(x)               # (1024, 48, 48)
        
        x = self.head_3(x, mask)
        x = self.up(x)               # (512, 96, 96)
        
        x = self.head_4(x, mask)
        x = self.up(x)               # (256, 192, 192)
        
        x = self.head_5(x, mask)
        x = self.up(x)               # (128, 384, 384)
        
        x = self.head_6(x, mask)     # Keep resolution constant
        
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = self.tanh(x)             # output range [-1, 1]

        return x

if __name__ == "__main__":
    # Test rapido per verificare che il modulo funzioni
    spade = SPADEGenerator(label_nc=10)
    #random noise
    z = torch.randn(5, 256)
    print(z.shape)
    mask = torch.randint(0, 10, (5, 10, 128, 128)).float()  # maschera con 10 classi
    out = spade(z, mask)
    print(out.shape)  # dovrebbe essere (5, 3, 128, 128)