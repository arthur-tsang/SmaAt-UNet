from torch import nn
from .unet_parts import OutConv
from .unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from .layers import CBAM


class SmaAt_UNet_extralevel(nn.Module):
    def __init__(self, n_channels, n_classes, kernels_per_layer=2,
                 bilinear=True, reduction_ratio=16, pdrop=0,
                 fac=64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        self.fac = fac
        # The original UNet starts at 64, but it also has one fewer level
        
        self.inc = DoubleConvDS(self.n_channels, 1 * fac, kernels_per_layer=kernels_per_layer, pdrop=pdrop)
        self.cbam1 = CBAM(1 * fac, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(1 * fac, 2 * fac, kernels_per_layer=kernels_per_layer, pdrop=pdrop)
        self.cbam2 = CBAM(2 * fac, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(2 * fac, 4 * fac, kernels_per_layer=kernels_per_layer, pdrop=pdrop)
        self.cbam3 = CBAM(4 * fac, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(4 * fac, 8 * fac, kernels_per_layer=kernels_per_layer, pdrop=pdrop)
        self.cbam4 = CBAM(8 * fac, reduction_ratio=reduction_ratio)
        self.down4 = DownDS(8 * fac, 16 * fac, kernels_per_layer=kernels_per_layer, pdrop=pdrop) # had bug earlier: wrote down5
        self.cbam5 = CBAM(16 * fac, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down5 = DownDS(16 * fac, 32 * fac // factor, kernels_per_layer=kernels_per_layer, pdrop=pdrop)
        self.cbam6 = CBAM(32 * fac // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(32 * fac, 16 * fac // factor, self.bilinear, kernels_per_layer=kernels_per_layer, pdrop=pdrop)
        self.up2 = UpDS(16 * fac, 8 * fac // factor, self.bilinear, kernels_per_layer=kernels_per_layer, pdrop=pdrop)
        self.up3 = UpDS(8 * fac, 4 * fac // factor, self.bilinear, kernels_per_layer=kernels_per_layer, pdrop=pdrop)
        self.up4 = UpDS(4 * fac, 2 * fac // factor, self.bilinear, kernels_per_layer=kernels_per_layer, pdrop=pdrop)
        self.up5 = UpDS(2 * fac, 1 * fac, self.bilinear, kernels_per_layer=kernels_per_layer, pdrop=pdrop)

        self.outc = OutConv(1 * fac, self.n_classes)

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
        x6 = self.down5(x5)
        x6Att = self.cbam6(x6)
        x = self.up1(x6Att, x5Att)
        x = self.up2(x, x4Att)
        x = self.up3(x, x3Att)
        x = self.up4(x, x2Att)
        x = self.up5(x, x1Att)
        logits = self.outc(x)
        return logits
