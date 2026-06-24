import torch
import torch.nn as nn

class DenseDownBlock(nn.Module):
    def __init__(self, in_c, out_c, pool=True):
        super().__init__()
        # Simulating the "Dense Block -> Downsample" from the diagram
        self.dense = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        features = self.dense(x)
        out = self.pool(features)
        return features, out

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # "Concat skip connection -> Dense block -> Upsample block"
        self.dense = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.upsample = nn.ConvTranspose2d(out_c, out_c, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.dense(x)
        x = self.upsample(x)
        return x

class EncoderBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial block (Green arrow): 3x3x16 conv -> Leaky ReLu -> 2x2 avg pool
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2)
        )
        
        # Blue arrows: Dense -> Downsample
        self.down1 = DenseDownBlock(16, 32)
        self.down2 = DenseDownBlock(32, 64)
        # The 2x2 feature map must be pooled once more to produce the
        # 1x1x100 bottleneck shown in the ISNet architecture.
        self.down3 = DenseDownBlock(64, 100)

    def forward(self, x):
        skip0 = x                 # 16x16x1
        out1 = self.init_conv(x)  # 8x8x16
        skip1 = out1
        
        _, out2 = self.down1(out1)     # 4x4x32
        skip2 = out2
        _, out3 = self.down2(out2)     # 2x2x64
        skip3 = out3
        _, out4 = self.down3(out3)     # 1x1x100
        
        return skip0, skip1, skip2, skip3, out4

class ISNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Three parallel encoders
        self.enc_I = EncoderBranch()
        self.enc_Sx = EncoderBranch()
        self.enc_Sy = EncoderBranch()
        
        # Bottleneck Expansion: 300 -> 36 and upsample to 2x2
        self.bottleneck_expand = nn.ConvTranspose2d(300, 36, kernel_size=2, stride=2)
        
        # Decoders (Pink arrows)
        # 2x2 stage: input 36 + skips (64*3) = 228
        self.dec1 = DecoderBlock(228, 36)
        # 4x4 stage: input 36 + skips (32*3) = 132
        self.dec2 = DecoderBlock(132, 36)
        # 8x8 stage: input 36 + skips (16*3) = 84
        self.dec3 = DecoderBlock(84, 36)
        # 16x16 stage: input 36 + skips (1*3) = 39
        self.dec4 = DecoderBlock(39, 36)
        
        # Final output (Red arrow)
        self.final_conv = nn.Conv2d(36, 1, kernel_size=1)

    def forward(self, I, Sx, Sy):
        # Encode I
        I_s0, I_s1, I_s2, I_s3, I_out = self.enc_I(I)
        # Encode Sx
        Sx_s0, Sx_s1, Sx_s2, Sx_s3, Sx_out = self.enc_Sx(Sx)
        # Encode Sy
        Sy_s0, Sy_s1, Sy_s2, Sy_s3, Sy_out = self.enc_Sy(Sy)
        
        # Bottleneck Concat (Dotted black arrow) -> 1x1x300
        bottleneck = torch.cat([I_out, Sx_out, Sy_out], dim=1)
        x = self.bottleneck_expand(bottleneck) # 2x2x36
        
        # Decoding with Skip Concatenations
        x = torch.cat([x, I_s3, Sx_s3, Sy_s3], dim=1) # 2x2
        x = self.dec1(x)                              # outputs 4x4x36
        
        x = torch.cat([x, I_s2, Sx_s2, Sy_s2], dim=1) # 4x4
        x = self.dec2(x)                              # outputs 8x8x36
        
        x = torch.cat([x, I_s1, Sx_s1, Sy_s1], dim=1) # 8x8
        x = self.dec3(x)                              # outputs 16x16x36
        
        x = torch.cat([x, I_s0, Sx_s0, Sy_s0], dim=1) # 16x16
        x = self.dec4(x)                              # outputs 32x32x36
        
        # Output 32x32x1 Phase Map
        out = self.final_conv(x)
        return out
