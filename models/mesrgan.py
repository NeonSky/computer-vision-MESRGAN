import torch
import torch.nn as nn

#################
### Generator ###
#################

# Based on the following two papers:
# * ESRGAN: https://arxiv.org/pdf/1809.00219v2.pdf
# * SRGAN: https://arxiv.org/pdf/1609.04802.pdf

# Residual Dense Block (RDB)
# The params are only specified on the GitHub repostitory of ESRGAN: https://github.com/xinntao/ESRGAN/blob/master/RRDBNet_arch.py

class MobileConv(nn.Module):
    def __init__(self, C_in, C_out, t):
        super().__init__()
        
        k = int(C_in*t)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.conv1x1 = nn.Conv2d(C_in, k, 1, 1, 0, bias=True)
        self.conv3x3 = nn.Conv2d(k, k, 3, 1, 1, groups=k, bias=True)
        self.conv_linear = nn.Conv2d(k, C_out, 1, 1, 0, bias=True)
        
    def forward(self, x):
        x = self.lrelu(self.conv1x1(x))
        x = self.lrelu(self.conv3x3(x))
        x = self.conv_linear(x)
        return x

    
class RDB(nn.Module):

    def __init__(self, n_filters=64, growth_channel=32, t=1):
        super().__init__()

        nf = n_filters
        gc = growth_channel

        self.residual_scaling_param = 0.2
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv1 = MobileConv(nf, gc, t)
        self.conv2 = MobileConv(nf + gc, gc, t)
        self.conv3 = MobileConv(nf + 2 * gc, gc, t)
        self.conv4 = MobileConv(nf + 3 * gc, gc, t)
        self.conv5 = MobileConv(nf + 4 * gc, nf, t)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x + x5 * self.residual_scaling_param

# Residual-in-Residual Dense Block (RRDB)
class RRDB(nn.Module):

    def __init__(self, n_blocks, t=2):
        super().__init__()

        self.residual_scaling_param = 0.2
        self.rdbs = nn.Sequential(*[RDB(t=t) for _ in range(n_blocks)])

    def forward(self, x):
        x_og = x.clone()

        x = self.rdbs(x)

        return x_og + self.residual_scaling_param * x

# Based on the enhanced version of SRResNet (i.e. the generator of SRGAN).
class Generator(nn.Module):

    def __init__(self, n_channels=3, n_rrdbs=16, n_upscales=2, t=2):
        super().__init__()

        self.prelu = nn.PReLU()

        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=9, stride=1, padding=4, bias=True)

        self.rrdbs = RRDB(16, t)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

        def upsample_layer():
            return nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(upscale_factor=2), # c_out = out_channels / upscale_factor^2. In our case, 256 / 2^2 = 64
                self.prelu
            )
        self.upsamples = nn.Sequential(*[upsample_layer() for _ in range(n_upscales)])

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x_pre_rrdbs = x.clone()

        x = self.rrdbs(x)
        x = self.conv2(x)

        x = x + x_pre_rrdbs

        x = self.upsamples(x)
        x = self.prelu(self.conv3(x))
        x = self.conv4(x)

        return x
    
#####################
### Discriminator ###
#####################

def conv2d(
    in_channels, 
    out_channels, 
    stride=1, 
    kernel_size=3,
    padding=1, 
    bias=False):

    return nn.Conv2d(
        in_channels, 
        out_channels,
        stride=stride,
        kernel_size=kernel_size,
        bias=bias)

def conv_bn_relu_block(C_in, C_out, *args, **kwargs):
    return nn.Sequential(
        conv2d(C_in, C_out, *args, **kwargs),
        nn.BatchNorm2d(C_out),
        nn.RReLU()
    )

def stride_pair_block(C_in, C_out):
    return nn.Sequential(
        conv_bn_relu_block(C_in, C_out, stride=1),
        conv_bn_relu_block(C_out, C_out, stride=2)
    )
        
class Discriminator(nn.Module):
    def __init__(self, patch_size=128, C_in=3):
        super().__init__()
        self.conv1 = conv2d(C_in, 64, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        
        self.block1 = conv_bn_relu_block(64, 64, stride=2)
        self.block2 = stride_pair_block(64, 128)
        self.block3 = stride_pair_block(128, 256)
        self.block4 = stride_pair_block(256, 512)
        
        in_dim = self._forward_conv(torch.zeros(1, 3, patch_size, patch_size)).size()[1] # 0th dim is batch size
        
        self.dense1 = nn.Linear(in_dim, 1024)
        self.dense2 = nn.Linear(1024, 1)

    def _forward_conv(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)

        x = self.lrelu(self.dense1(x))
        x = self.dense2(x)

        return x