"""
2D UNet backbone with residual blocks consisting of 2 convolutional operations.
The UNet encoder uses 4 feature map resolutions (64 x 64 -> 32 x 32 -> 16 x 16 -> 8 x 8)
Conditioning: simply the class label for now 
"""

import math
import torch
import torch.nn as nn

# model 
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, d_emb=256):
        super().__init__()
        self.proj_embs = nn.Linear(d_emb, out_ch)
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act1 = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x, lembs):
        # project label embeddings to inject inside res block
        lembs = self.proj_embs(lembs)
        lembs = lembs[:, :, None, None] # (B, out_ch, 1, 1)

        f = x
        f = self.norm1(f)
        f = self.act1(f)
        f = self.conv1(f)
        f = f + lembs
        f = self.norm2(f)
        f = self.act2(f)
        f = self.conv2(f)
        return f + self.skip(x)

class UNet(nn.Module):
    def __init__(self, num_classes=10, in_ch=3, ch=(64, 128, 256, 512), d_emb=256):
        super().__init__()
        self.d_emb = d_emb
        self.num_classes = num_classes

        # class embeddings
        self.label_embedding = nn.Embedding(self.num_classes, self.d_emb)

        # encoder
        self.in_project = nn.Conv2d(in_ch, ch[0], kernel_size=3, padding=1)
        self.enc1 = ResBlock(ch[0], ch[1])
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResBlock(ch[1], ch[2])
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResBlock(ch[2], ch[3])
        self.pool3 = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = ResBlock(ch[3], ch[3])

        # decoder
        self.up1 = nn.ConvTranspose2d(in_channels=ch[3], out_channels=ch[3], kernel_size=2, stride=2)
        self.dec1 = ResBlock(ch[3] * 2, ch[2])

        self.up2 = nn.ConvTranspose2d(in_channels=ch[2], out_channels=ch[2], kernel_size=2, stride=2)
        self.dec2 = ResBlock(ch[2] * 2, ch[1])

        self.up3 = nn.ConvTranspose2d(in_channels=ch[1], out_channels=ch[1], kernel_size=2, stride=2)
        self.dec3 = ResBlock(ch[1] * 2, ch[0])
        self.out_proj = nn.Conv2d(ch[0], in_ch, kernel_size=3, padding=1)

    def forward(self, x, y):
        """Run a forward pass.

        Args:
            x: Noisy image tensor (B, C, H, W). Here, it is (B, 3, 64, 64)
            y: labels tensor (B, 1, 1, 1)
        """
        lembs = self.label_embedding(y)

        # encoder
        x = self.in_project(x)          # (B, 64, 64, 64)
        x = self.enc1(x, lembs)         # (B, 128, 64, 64)
        skip1 = x
        x = self.pool1(x)               # (B, 128, 32, 32)

        x = self.enc2(x, lembs)         # (B, 256, 32, 32)
        skip2 = x
        x = self.pool2(x)               # (B, 256, 16, 16)

        x = self.enc3(x, lembs)         # (B, 512, 16, 16)
        skip3 = x
        x = self.pool3(x)               # (B, 512, 8, 8)  

        # bottleneck
        x = self.bottleneck(x, lembs)   # (B, 512, 8, 8) 

        # decoder
        x = self.up1(x)                             # (B, 512, 16, 16)
        x = torch.cat([x, skip3], dim=1)            # (B, 1024, 16, 16)
        x = self.dec1(x, lembs)                     # (B, 256, 16, 16)

        x = self.up2(x)                             # (B, 256, 32, 32)
        x = torch.cat([x, skip2], dim=1)            # (B, 512, 32, 32)
        x = self.dec2(x, lembs)                     # (B, 128, 32, 32)

        x = self.up3(x)                             # (B, 128, 64, 64)
        x = torch.cat([x, skip1], dim=1)            # (B, 256, 64, 64)
        x = self.dec3(x, lembs)                     # (B, 64, 64, 64)
        x = self.out_proj(x)                        # (B, 3, 64, 64)

        return x

