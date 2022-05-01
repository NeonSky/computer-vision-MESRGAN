import torch
import torch.nn as nn

#################
### Generator ###
#################

class ResidualBlock(nn.Module):

    def __init__(self):
        super().__init__()
        
        # No conv biases since batch norms directly follow
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64)
        )

    def forward(self, x):
        x_og = x.clone()
        ################
        x = self.layers(x)
        ################
        x = x + x_og
        return x

class Generator(nn.Module):

    def __init__(self, n_rrdbs=16, n_upscales=2, n_channels=3):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=9, stride=1, padding=4, bias=True),
            nn.PReLU()
        )

        self.rrdbs = nn.Sequential(*[ResidualBlock() for _ in range(n_rrdbs)])

        # No conv bias since a batch norm follows
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64)
        )

        def upsample_layer():
            return nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(upscale_factor=2), # c_out = out_channels / upscale_factor^2. In our case, 256 / 2^2 = 64
                nn.PReLU()
            )
        self.upsamples = nn.Sequential(*[upsample_layer() for _ in range(n_upscales)])

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        
        x_og = x.clone()
        ################
        x = self.rrdbs(x)
        x = self.conv2(x)
        ################
        x = x + x_og

        x = self.upsamples(x)
        x = self.conv3(x)

        return x

#####################
### Discriminator ###
#####################
    
def conv2d(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
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
    def __init__(self, patch_size=96, n_channels=3):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            conv2d(n_channels, 64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        self.block1 = conv_bn_relu_block(64, 64, stride=2)
        self.block2 = stride_pair_block(64, 128)
        self.block3 = stride_pair_block(128, 256)
        self.block4 = stride_pair_block(256, 512)
        
        in_dim = self._forward_conv(torch.zeros(1, 3, patch_size, patch_size)).size()[1] # 0th dim is batch size
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def _forward_conv(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = self.classifier(x)
        return x