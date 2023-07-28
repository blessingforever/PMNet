import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.encoder_modules import ResBlock, UpsampleBlock
from modules.deform_conv_v2 import DeformConv2d

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ResBlock(3072, 512)
        self.deform = DeformConv2d(512, 512, 3, padding=1, modulation=True)
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16, f8, f4, guide_feat):
        # f16: [bs,1024,h,w], guide_feat: [bs,2048,h,w]
        f_fuse = torch.cat((f16, guide_feat), 1)#[bs, 3072, h, w]
        x = self.compress(f_fuse)
        x = self.deform(x) # bs, 512, h, w 
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)

        x = self.pred(F.relu(x))
        
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x
