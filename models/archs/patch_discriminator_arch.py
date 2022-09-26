import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from models.archs.attention import NonLocalBlock2D, PyramidAttention

class PatchDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PatchDiscriminator, self).__init__()
        use_bias = False
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        # TODO
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        """Standard forward."""
        return self.model(x)


#PatchGAN Discriminator with Non-Local Attention
class NLPatchDiscriminator(nn.Module):

    def __init__(self, input_nc=3, ndf=64, n_layers=3, ksize=3, ks_padding=1, norm_layer=nn.BatchNorm2d):
        super(NLPatchDiscriminator, self).__init__()
        use_bias = False
        kw = 4
        padw = 1
        sequence = []

        self.conv0 = nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
        self.relu0 = nn.LeakyReLU(0.2, True)
        
        #Non-local attention module
        self.nlb = NonLocalBlock2D(in_channels=64, inter_channels=32)
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)

        x = self.nlb(x)
        
        x = self.model(x)
        return x

# PatchGAN Discriminator with Pyramid Attention
class PAPatchDiscriminator(nn.Module):

    def __init__(self, input_nc=3, ndf=64, n_layers=3, ksize=3, ks_padding=1, norm_layer=nn.BatchNorm2d):
        super(PAPatchDiscriminator, self).__init__()
        use_bias = False
        kw = 4
        padw = 1
        sequence = []

        self.conv0 = nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
        self.relu0 = nn.LeakyReLU(0.2, True)

        #two scale pyramid attention module
        self.pa = PyramidAttention(channel=ndf, reduction=2, scale=[1, 0.5], ksize=ksize, ks_padding=ks_padding)
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)

        x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x = self.pa(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        
        x = self.model(x)
        return x
