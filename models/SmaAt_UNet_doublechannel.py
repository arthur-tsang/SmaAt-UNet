from torch import nn
from .unet_parts import OutConv
from .unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from .layers import CBAM


class SmaAt_UNet_doublechannel(nn.Module):
    def __init__(self, n_channels, n_classes, kernels_per_layer=2,
                 bilinear=True, reduction_ratio=16, pdrop=0):
        super(SmaAt_UNet_doublechannel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        fac = 2 # channel factor
        
        self.inc = DoubleConvDS(self.n_channels, 64 * fac, kernels_per_layer=kernels_per_layer, pdrop=pdrop)
        self.cbam1 = CBAM(64 * fac, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64 * fac, 128 * fac, kernels_per_layer=kernels_per_layer, pdrop=pdrop)
        self.cbam2 = CBAM(128 * fac, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128 * fac, 256 * fac, kernels_per_layer=kernels_per_layer, pdrop=pdrop)
        self.cbam3 = CBAM(256 * fac, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256 * fac, 512 * fac, kernels_per_layer=kernels_per_layer, pdrop=pdrop)
        self.cbam4 = CBAM(512 * fac, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512 * fac, 1024 * fac // factor, kernels_per_layer=kernels_per_layer, pdrop=pdrop)
        self.cbam5 = CBAM(1024 * fac // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024 * fac, 512 * fac // factor, self.bilinear, kernels_per_layer=kernels_per_layer, pdrop=pdrop)
        self.up2 = UpDS(512 * fac, 256 * fac // factor, self.bilinear, kernels_per_layer=kernels_per_layer, pdrop=pdrop)
        self.up3 = UpDS(256 * fac, 128 * fac // factor, self.bilinear, kernels_per_layer=kernels_per_layer, pdrop=pdrop)
        self.up4 = UpDS(128 * fac, 64 * fac, self.bilinear, kernels_per_layer=kernels_per_layer, pdrop=pdrop)

        self.outc = OutConv(64 * fac, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits
