from torch import nn
from models.blocks import get_norm_layer, BottleNeckResBlock, ConvBlock, ResBlock, Upsample

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        channels = config.dec_channels
        kernel_size = config.dec_kernel_size
        stride = config.dec_stride

        res_norm = 'none'       # no adain in res
        norm = 'none'
        pad_type = 'reflect'
        acti = 'lrelu'

        layers = []
        n_resblk = config.dec_resblks   # 1
        n_conv = config.dec_up_n    # 1
        bt_channel = config.dec_bt_channel  # #channels at the bottleneck 144

        self.linear = nn.Linear(256, 63)        # 84->63  kit_ml
        layers += get_norm_layer('adain', channels[0])    # adain before everything

        for i in range(n_resblk):
            layers.append(BottleNeckResBlock(kernel_size, channels[0], bt_channel, channels[0], pad_type=pad_type, norm=res_norm, acti=acti))

        for i in range(n_conv):
            layers.append(Upsample(scale_factor=2, mode='nearest'))
            cur_acti = 'none' if i == n_conv - 1 else acti
            cur_norm = 'none' if i == n_conv - 1 else norm
            layers += ConvBlock(kernel_size, channels[i], channels[i + 1], stride=stride, pad_type=pad_type, norm=cur_norm, acti=cur_acti)

        self.model = nn.Sequential(*layers)
        self.channels = channels    # [144, 63]

        # contentEncoder
        kernel_size = config.enc_co_kernel_size  # 8
        stride = config.enc_co_stride  # 2

        layers0 = []
        n_convs = config.enc_co_down_n  # 1
        n_resblk = config.enc_co_resblks  # 1
        acti = 'lrelu'
        assert n_convs + 1 == len(channels)

        for i in range(n_convs):
            layers0 += ConvBlock(kernel_size, channels[i+1], channels[i], stride=stride, norm='in', acti=acti)

        for i in range(n_resblk):
            layers0.append(ResBlock(kernel_size, channels[0], stride=1, pad_type='reflect', norm='in', acti=acti))

        self.conv_model = nn.Sequential(*layers0)

    def forward(self, x):       # [bs, 63, 60]
        x = self.conv_model(x)      # [128, 144, 30]
        x = self.model(x)       # [128, 63, 60]
        return x