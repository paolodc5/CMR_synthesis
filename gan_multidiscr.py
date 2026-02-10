# Based on the paper from Mok et al. 2018 "Multi-Discriminators GAN for Semi-Supervised Cross-Modality Image Synthesis"
import torch
from torch import nn
from torchinfo import summary
import torch.nn.functional as F


class SingleDiscriminator(nn.Module):
    def __init__(self, in_ch=3):
        super(SingleDiscriminator, self).__init__()
        
        self.layer1 = self._block(in_ch, 64)
        self.layer2 = self._block(64, 128)
        self.layer3 = self._block(128, 256)
        self.layer4 = self._block(256, 512)
        
        self.final = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
    
    def _block(self, in_f, out_f):
        return nn.Sequential(
            nn.Conv2d(in_f, out_f, 4, 2, 1),
            nn.BatchNorm2d(out_f),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        features = [] # Useful for perceptual loss if needed
        
        x = self.layer1(x)
        features.append(x) 
        
        x = self.layer2(x)
        features.append(x) 
        
        x = self.layer3(x)
        features.append(x)  
        
        x = self.layer4(x)
        features.append(x) 
        
        out = self.final(x)
        
        return out, features
    


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_ch=3, n_discriminators=4):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            SingleDiscriminator(in_ch) for _ in range(n_discriminators)
        ])

    def forward(self, x):
        results = []
        input_downsampled = x
        
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                # Bilinear downsapmling
                input_downsampled = F.interpolate(input_downsampled, scale_factor=0.5, 
                                                  mode='bilinear', align_corners=False)
            
            pred, feats = disc(input_downsampled)
            results.append((pred, feats))
            
        return results
    

if __name__ == "__main__":
    msd = MultiScaleDiscriminator(in_ch=16)
    inp = torch.randn(2, 16, 256, 256)
    out = msd(inp)
    
    for i, (pred, feats) in enumerate(out):
        print(f"Discriminator {i}:")
        print(f"    Output shape: {pred.shape}")
        for j, feat in enumerate(feats):
            print(f"    Feature {j} shape: {feat.shape}")      