from torch import nn
import torch.nn.functional as F
from models.blocks import ConvBlock
class EncoderStyle(nn.Module):
    def __init__(self, config):
        super(EncoderStyle, self).__init__()
        channels = config.enc_cl_channels
        channels[0] = config.style_channel_3d

        kernel_size = config.enc_cl_kernel_size
        stride = config.enc_cl_stride

        self.global_pool = F.max_pool1d

        layers = []
        n_convs = config.enc_cl_down_n      # 2

        for i in range(n_convs):
            layers += ConvBlock(kernel_size, channels[i], channels[i + 1], stride=stride, norm='none', acti='lrelu')

        self.conv_model = nn.Sequential(*layers)
        self.channels = channels

    def forward(self, x):   # [20, 21, 3, 60]
        x = x.transpose(1, 2)   # [20, 3, 21, 60]
        x = x.reshape([x.shape[0], 63, x.shape[3]])   # [20, 63, 60]
        x = self.conv_model(x)
        kernel_size = x.shape[-1]
        x = self.global_pool(x, kernel_size)
        return x