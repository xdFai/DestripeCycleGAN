import torch.nn as nn
import torch
from pytorch_wavelets import DWTForward, DWTInverse
from torchvision import transforms
import os
import torch.nn.functional as F
import matplotlib.pylab as plt
import torchvision
import os

"""
The Writeness of DWT has been changed
"""

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class single_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(single_conv, self).__init__()
        self.s_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.s_conv(x)
        return x

class conv11(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv11, self).__init__()
        self.s_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.s_conv(x)
        return x


class conv33(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv33, self).__init__()
        self.s_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.s_conv(x)
        return x

class conv55(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv55, self).__init__()
        self.s_conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.s_conv(x)
        return x

class conv77(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv77, self).__init__()
        self.s_conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)

    def forward(self, x):
        x = self.s_conv(x)
        return x

class _DCR_block(nn.Module):
    def __init__(self, channel_in):
        super(_DCR_block, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in / 2.), kernel_size=3, stride=1,
                                padding=1)
        self.relu1 = nn.LeakyReLU()
        self.conv_2 = nn.Conv2d(in_channels=int(channel_in * 3 / 2.), out_channels=int(channel_in / 2.), kernel_size=3,
                                stride=1, padding=1)
        self.relu2 = nn.LeakyReLU()
        self.conv_3 = nn.Conv2d(in_channels=channel_in * 2, out_channels=channel_in, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.LeakyReLU()

    def forward(self, x):
        residual = x
        out = self.relu1(self.conv_1(x))
        conc = torch.cat([x, out], 1)
        out = self.relu2(self.conv_2(conc))
        conc = torch.cat([conc, out], 1)
        out = self.relu3(self.conv_3(conc))
        out = torch.add(out, residual)
        return out

##########################################################################
class ChannelPool(nn.Module):
    def forward(self, x):
        # 将maxpooling 与 global average pooling 结果拼接在一起
        return torch.cat((torch.max(x, 1) [0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = conv77(2,1)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()

        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()#H*C*W
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()#C*H*W
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()#W*H*C
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()#C*H*W
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class group_aggregation_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            conv11(group_size,group_size),
            TripletAttention()
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            conv33(group_size,group_size),
            TripletAttention()
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            conv55(group_size,group_size),
            TripletAttention()
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size, data_format='channels_first'),
            conv77(group_size,group_size),
            TripletAttention()
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2, dim_xl, 1)
        )
    def forward(self, xh, xl):
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode ='bilinear', align_corners=True)
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)
        x0 = self.g0(torch.cat((xh[0], xl[0]), dim=1))
        x1 = self.g1(torch.cat((xh[1], xl[1]), dim=1))
        x2 = self.g2(torch.cat((xh[2], xl[2]), dim=1))
        x3 = self.g3(torch.cat((xh[3], xl[3]), dim=1))
        x = torch.cat((x0,x1,x2,x3), dim=1)
        x = self.tail_conv(x)
        return x

class MWUNet(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(MWUNet, self).__init__()
        self.features = []

        # encoder*****************************************************
        self.head = single_conv(in_ch, 32)
        self.dconv_encode0 = nn.Sequential(single_conv(32, 32), _DCR_block(32))  # → har
        self.DWT = DWTForward(J=1, wave='haar').cuda()
        self.dconv_encode1 = nn.Sequential(single_conv(128, 64), _DCR_block(64))  # → har
        self.DWT = DWTForward(J=1, wave='haar').cuda()
        self.dconv_encode2 = nn.Sequential(single_conv(256, 128), _DCR_block(128))  # → pool
        self.maxpool = nn.MaxPool2d(2)
        self.mid = nn.Sequential(single_conv(512, 256), _DCR_block(256),
                                 single_conv(256, 512))

        # upsample*****************************************************
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
        )

        self.upsample0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.IDWT = DWTInverse(wave='haar').cuda()

        # skip*****************************************************
        self.GAB1 = group_aggregation_bridge(256, 64)
        self.GAB2 = group_aggregation_bridge(512, 128)

        # decoder*****************************************************
        self.dconv_decode2 = nn.Sequential(single_conv(128 + 128, 128), _DCR_block(128),single_conv(128, 256))

        self.dconv_decode1 = nn.Sequential(single_conv(64 + 64, 64), _DCR_block(64),single_conv(64, 128))

        self.dconv_decode0 = nn.Sequential(single_conv(64, 32), _DCR_block(32),single_conv(32, 32))
        self.tail = nn.Sequential(nn.Conv2d(32, out_ch, 1), nn.Tanh())

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:, :, i, :, :])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)

    def _Itransformer(self, out):
        yh = []
        C = int(out.shape[1] / 4)
        yl = out[:, 0:C, :, :]
        y1 = out[:, C:2 * C, :, :].unsqueeze(2)
        y2 = out[:, 2 * C:3 * C, :, :].unsqueeze(2)
        y3 = out[:, 3 * C:4 * C, :, :].unsqueeze(2)
        final = torch.cat([y1, y2, y3], 2)
        yh.append(final)
        return yl, yh

    def forward(self, x):
        input = x
        # x = torch.cat((x, mask), 1)
        # *****************************************************************************
        # head +encoder
        x0 = self.dconv_encode0(self.head(x))
        res0 = x0
        # *****************************************************************************
        # har
        DMT1_yl, DMT1_yh = self.DWT(x0)
        DMT1 = self._transformer(DMT1_yl, DMT1_yh)
        x1 = self.dconv_encode1(DMT1)
        res1 = x1
        # *****************************************************************************
        # har
        DMT1_yl, DMT1_yh = self.DWT(x1)
        DMT2 = self._transformer(DMT1_yl, DMT1_yh)
        x2 = self.dconv_encode2(DMT2)
        res2 = x2
        # *****************************************************************************
        # pool
        DMT1_yl, DMT1_yh = self.DWT(x2)
        DMT3 = self._transformer(DMT1_yl, DMT1_yh)
        x3 = self.mid(DMT3)
        # *****************************************************************************

        x2 = self.GAB2(x3,x2)
        x = self._Itransformer(x3)

        x = self.IDWT(x)
        # *****************************************************************************
        x = self.dconv_decode2(torch.cat([x, x2], dim=1))
        # *****************************************************************************
        x1 = self.GAB1(x, x1)
        x = self._Itransformer(x)
        x = self.IDWT(x)
        # *****************************************************************************
        x = self.dconv_decode1(torch.cat([x, x1], dim=1))
        # *****************************************************************************
        x = self._Itransformer(x)
        x = self.IDWT(x)
        # *****************************************************************************
        x = self.dconv_decode0(torch.cat([x, x0], dim=1))
        x = self.tail(x)
        # *****************************************************************************
        out = x + input

        return out

if __name__ == '__main__':
    net = MWUNet(3, 3).cuda()
    input = torch.zeros((1, 3, 64, 64), dtype=torch.float32).cuda()
    output = net(input)
    # print(net.features)
    print(output.shape)