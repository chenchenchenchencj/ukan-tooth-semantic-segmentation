import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *

try:
    import timm
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_
except Exception:
    timm = None
    def to_2tuple(x):
        return (x, x) if not isinstance(x, tuple) else x
    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.):
            super().__init__()
            self.drop_prob = drop_prob
        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor
    def trunc_normal_(tensor, mean=0., std=1.):
        return nn.init.trunc_normal_(tensor, mean=mean, std=std)
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
from pdb import set_trace as st

from kan import KANLinear, KAN
from torch.nn import init
__all__ = [
    "KANLayer", "KANBlock", "DWConv", "DW_bn_relu", "PatchEmbed",
    "ConvLayer", "D_ConvLayer", "UKAN", "UKAN_MSAG", "UKAN_EGMS",
    "PlainUNet", "ResUNet", "NestedUNetPP", "DeepLabV3PlusLite",
    "SegFormerMini"
]

class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]

        if not no_kan:
            self.fc1 = KANLinear(
                in_features,
                hidden_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.fc2 = KANLinear(
                hidden_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.fc3 = KANLinear(
                hidden_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            # # TODO
            # self.fc4 = KANLinear(
            #             hidden_features,
            #             out_features,
            #             grid_size=grid_size,
            #             spline_order=spline_order,
            #             scale_noise=scale_noise,
            #             scale_base=scale_base,
            #             scale_spline=scale_spline,
            #             base_activation=base_activation,
            #             grid_eps=grid_eps,
            #             grid_range=grid_range,
            #         )

        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        # TODO
        # self.fc1 = nn.Linear(in_features, hidden_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)

        # # TODO
        # self.dwconv_4 = DW_bn_relu(hidden_features)

        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_3(x, H, W)

        # # TODO
        # x = x.reshape(B,N,C).contiguous()
        # x = self.dwconv_4(x, H, W)

        return x


class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                              no_kan=no_kan)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class ChannelSpatialGate(nn.Module):
    """Lightweight channel-spatial attention for panoramic tooth foreground cues."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x * self.channel_gate(x)
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.spatial_gate(torch.cat([avg, mx], dim=1))


class MultiScaleDilatedContext(nn.Module):
    """ASPP-like context block for elongated teeth and variable root scale."""
    def __init__(self, channels, rates=(1, 3, 5)):
        super().__init__()
        branch_ch = max(channels // len(rates), 8)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, branch_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.ReLU(inplace=True),
            ) for r in rates
        ])
        self.project = nn.Sequential(
            nn.Conv2d(branch_ch * len(rates), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.attn = ChannelSpatialGate(channels)

    def forward(self, x):
        feat = torch.cat([b(x) for b in self.branches], dim=1)
        feat = self.project(feat)
        return x + self.attn(feat)


class EdgeGuidedRefine(nn.Module):
    """Predicts a soft boundary map and uses it to sharpen decoder features."""
    def __init__(self, channels):
        super().__init__()
        mid = max(channels // 2, 8)
        self.edge = nn.Sequential(
            nn.Conv2d(channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            ChannelSpatialGate(channels),
        )

    def forward(self, x):
        edge = self.edge(x)
        return x + self.refine(torch.cat([x, edge], dim=1))


class UKAN(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[256, 320, 512], no_kan=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]

        self.encoder1 = ConvLayer(3, kan_input_dim // 8)
        self.encoder2 = ConvLayer(kan_input_dim // 8, kan_input_dim // 4)
        self.encoder3 = ConvLayer(kan_input_dim // 4, kan_input_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1],
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
        )])

        self.block2 = nn.ModuleList([KANBlock(
            dim=embed_dims[2],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
        )])

        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1],
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
        )])

        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[0],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
        )])

        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2])

        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0] // 4)
        self.decoder4 = D_ConvLayer(embed_dims[0] // 4, embed_dims[0] // 8)
        self.decoder5 = D_ConvLayer(embed_dims[0] // 8, embed_dims[0] // 8)

        self.final = nn.Conv2d(embed_dims[0] // 8, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out

        ### Tokenized KAN Stage
        ### Stage 4

        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))

        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return self.final(out)

class UKAN_MSAG(UKAN):
    """Multi-Scale Attention-Gated U-KAN without boundary refinement."""
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[256, 320, 512], no_kan=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], **kwargs):
        super().__init__(num_classes, input_channels, deep_supervision, img_size, patch_size, in_chans,
                         embed_dims, no_kan, drop_rate, drop_path_rate, norm_layer, depths, **kwargs)
        self.ms_context = MultiScaleDilatedContext(embed_dims[2], rates=(1, 2, 4))
        self.skip4_attn = ChannelSpatialGate(embed_dims[1])
        self.skip3_attn = ChannelSpatialGate(embed_dims[0])
        self.skip2_attn = ChannelSpatialGate(embed_dims[0] // 4)
        self.skip1_attn = ChannelSpatialGate(embed_dims[0] // 8)

    def forward(self, x):
        B = x.shape[0]
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2)); t1 = self.skip1_attn(out)
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2)); t2 = self.skip2_attn(out)
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2)); t3 = self.skip3_attn(out)
        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = self.skip4_attn(out)
        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.ms_context(out)
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H, W)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))
        return self.final(out)


class UKAN_EGMS(UKAN):
    """Edge-Guided Multi-Scale U-KAN for tooth foreground semantic segmentation."""
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[256, 320, 512], no_kan=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], **kwargs):
        super().__init__(num_classes, input_channels, deep_supervision, img_size, patch_size, in_chans,
                         embed_dims, no_kan, drop_rate, drop_path_rate, norm_layer, depths, **kwargs)
        self.ms_context = MultiScaleDilatedContext(embed_dims[2], rates=(1, 2, 4))
        self.skip4_attn = ChannelSpatialGate(embed_dims[1])
        self.skip3_attn = ChannelSpatialGate(embed_dims[0])
        self.skip2_attn = ChannelSpatialGate(embed_dims[0] // 4)
        self.skip1_attn = ChannelSpatialGate(embed_dims[0] // 8)
        self.edge_refine = EdgeGuidedRefine(embed_dims[0] // 8)

    def forward(self, x):
        B = x.shape[0]
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2)); t1 = self.skip1_attn(out)
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2)); t2 = self.skip2_attn(out)
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2)); t3 = self.skip3_attn(out)
        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = self.skip4_attn(out)
        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.ms_context(out)
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H, W)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))
        out = self.edge_refine(out)
        return self.final(out)

class HaarFrequencyEnhance(nn.Module):
    """Learnable high-frequency enhancement for weak tooth boundaries in panoramic X-rays."""
    def __init__(self, channels=3, strength=0.35):
        super().__init__()
        self.strength = strength
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        low = F.avg_pool2d(x, kernel_size=2, stride=2)
        low = F.interpolate(low, size=x.shape[-2:], mode='bilinear', align_corners=False)
        high = x - low
        return x + self.strength * self.gate(high) * self.proj(high)


class LargeKernelContext(nn.Module):
    """SegNeXt/LKM-style large-kernel depthwise context for long tooth-arch structure."""
    def __init__(self, channels):
        super().__init__()
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.dw7d3 = nn.Conv2d(channels, channels, 7, padding=9, dilation=3, groups=channels, bias=False)
        self.pw = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        attn = self.pw(self.dw7d3(self.dw5(x)))
        return x + self.out(x * attn)


class AxialGlobalContext(nn.Module):
    """Coordinate/axial global context gate for upper-lower tooth row consistency."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )
        self.conv_h = nn.Conv2d(mid, channels, 1)
        self.conv_w = nn.Conv2d(mid, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        ph = F.adaptive_avg_pool2d(x, (h, 1))
        pw = F.adaptive_avg_pool2d(x, (1, w)).transpose(2, 3)
        y = torch.cat([ph, pw], dim=2)
        y = self.conv1(y)
        yh, yw = torch.split(y, [h, w], dim=2)
        yw = yw.transpose(2, 3)
        gate = torch.sigmoid(self.conv_h(yh)) * torch.sigmoid(self.conv_w(yw))
        return x + x * gate


class UKAN_WEG(UKAN_EGMS):
    """G: Wavelet/high-frequency edge-guided UKAN."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq_enhance = HaarFrequencyEnhance(3, strength=0.35)

    def forward(self, x):
        return super().forward(self.freq_enhance(x))


class UKAN_LKA(UKAN_EGMS):
    """H: Large-kernel context UKAN for long-range tooth arch modeling."""
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[256, 320, 512], no_kan=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], **kwargs):
        super().__init__(num_classes, input_channels, deep_supervision, img_size, patch_size, in_chans,
                         embed_dims, no_kan, drop_rate, drop_path_rate, norm_layer, depths, **kwargs)
        self.ms_context = nn.Sequential(
            MultiScaleDilatedContext(embed_dims[2], rates=(1, 2, 4)),
            LargeKernelContext(embed_dims[2]),
        )


class UKAN_GlobalLite(UKAN_EGMS):
    """I: Axial global-context UKAN, a lightweight Mamba-style long-context surrogate."""
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[256, 320, 512], no_kan=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], **kwargs):
        super().__init__(num_classes, input_channels, deep_supervision, img_size, patch_size, in_chans,
                         embed_dims, no_kan, drop_rate, drop_path_rate, norm_layer, depths, **kwargs)
        self.ms_context = nn.Sequential(
            MultiScaleDilatedContext(embed_dims[2], rates=(1, 2, 4)),
            AxialGlobalContext(embed_dims[2]),
        )


class UKAN_Proposed(UKAN_EGMS):
    """J: Proposed full model: high-frequency input, large-kernel context, axial global gate and edge refinement."""
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[256, 320, 512], no_kan=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], **kwargs):
        super().__init__(num_classes, input_channels, deep_supervision, img_size, patch_size, in_chans,
                         embed_dims, no_kan, drop_rate, drop_path_rate, norm_layer, depths, **kwargs)
        self.freq_enhance = HaarFrequencyEnhance(3, strength=0.35)
        self.ms_context = nn.Sequential(
            MultiScaleDilatedContext(embed_dims[2], rates=(1, 2, 4)),
            LargeKernelContext(embed_dims[2]),
            AxialGlobalContext(embed_dims[2]),
        )

    def forward(self, x):
        return super().forward(self.freq_enhance(x))
class WaveletHighFreqFusion(nn.Module):
    """Multi-branch Haar-like high-frequency fusion for dental boundaries at each scale."""
    def __init__(self, channels):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            ChannelSpatialGate(channels),
        )

    def forward(self, x):
        low2 = F.avg_pool2d(x, 2, 2)
        low2 = F.interpolate(low2, size=x.shape[-2:], mode='bilinear', align_corners=False)
        high2 = x - low2
        low4 = F.avg_pool2d(x, 4, 4)
        low4 = F.interpolate(low4, size=x.shape[-2:], mode='bilinear', align_corners=False)
        high4 = x - low4
        feat = self.fuse(torch.cat([self.local(x), high2, high4], dim=1))
        return x + feat


class FullScaleToothContext(nn.Module):
    """Heavy context block: multi-dilation context + large-kernel attention + axial global gate."""
    def __init__(self, channels):
        super().__init__()
        self.ms = MultiScaleDilatedContext(channels, rates=(1, 2, 4, 6))
        self.lka1 = LargeKernelContext(channels)
        self.lka2 = LargeKernelContext(channels)
        self.axial = AxialGlobalContext(channels, reduction=12)
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            ChannelSpatialGate(channels),
        )

    def forward(self, x):
        y = self.ms(x)
        y = self.lka1(y)
        y = self.lka2(y)
        y = self.axial(y)
        return x + self.out(y)


class SkipBoundaryFusion(nn.Module):
    """Strong skip fusion block combining frequency, large-kernel and edge refinement."""
    def __init__(self, channels):
        super().__init__()
        self.freq = WaveletHighFreqFusion(channels)
        self.context = FullScaleToothContext(channels)
        self.edge = EdgeGuidedRefine(channels)

    def forward(self, x):
        return self.edge(self.context(self.freq(x)))


class UKAN_ProposedXL(UKAN_EGMS):
    """Final heavy UKAN variant for paper experiments.
    Multi-scale high-frequency enhancement is applied to input and every skip,
    full-scale large-kernel context is used at token bottleneck, and decoder output
    is edge-refined. This is intentionally stronger than the lightweight screening modules.
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[256, 320, 512], no_kan=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], **kwargs):
        super().__init__(num_classes, input_channels, deep_supervision, img_size, patch_size, in_chans,
                         embed_dims, no_kan, drop_rate, drop_path_rate, norm_layer, depths, **kwargs)
        self.input_freq = HaarFrequencyEnhance(3, strength=0.45)
        self.skip1_attn = SkipBoundaryFusion(embed_dims[0] // 8)
        self.skip2_attn = SkipBoundaryFusion(embed_dims[0] // 4)
        self.skip3_attn = SkipBoundaryFusion(embed_dims[0])
        self.skip4_attn = SkipBoundaryFusion(embed_dims[1])
        self.ms_context = FullScaleToothContext(embed_dims[2])
        self.decoder_context3 = FullScaleToothContext(embed_dims[0])
        self.decoder_context2 = SkipBoundaryFusion(embed_dims[0] // 4)
        self.decoder_context1 = SkipBoundaryFusion(embed_dims[0] // 8)

    def forward(self, x):
        x = self.input_freq(x)
        B = x.shape[0]
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2)); t1 = self.skip1_attn(out)
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2)); t2 = self.skip2_attn(out)
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2)); t3 = self.skip3_attn(out)
        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = self.skip4_attn(out)
        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.ms_context(out)
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H, W)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = self.decoder_context3(out)
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        out = self.decoder_context2(torch.add(out, t2))
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        out = self.decoder_context1(torch.add(out, t1))
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))
        out = self.edge_refine(out)
        return self.final(out)

class _ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=False), nn.BatchNorm2d(out_ch)]
        if act:
            layers.append(nn.SiLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            _ConvBNAct(in_ch, out_ch),
            _ConvBNAct(out_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class _ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = _ConvBNAct(in_ch, out_ch)
        self.conv2 = _ConvBNAct(out_ch, out_ch, act=False)
        self.short = nn.Identity() if in_ch == out_ch else _ConvBNAct(in_ch, out_ch, k=1, p=0, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.short(x))


class _UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, block=_DoubleConv):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv = block(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class PlainUNet(nn.Module):
    """Conservative U-Net baseline for 2D tooth foreground segmentation."""
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, base_ch=32, **kwargs):
        super().__init__()
        c = base_ch
        self.e1 = _DoubleConv(input_channels, c)
        self.e2 = _DoubleConv(c, c * 2)
        self.e3 = _DoubleConv(c * 2, c * 4)
        self.e4 = _DoubleConv(c * 4, c * 8)
        self.b = _DoubleConv(c * 8, c * 16)
        self.d4 = _UpBlock(c * 16, c * 8, c * 8)
        self.d3 = _UpBlock(c * 8, c * 4, c * 4)
        self.d2 = _UpBlock(c * 4, c * 2, c * 2)
        self.d1 = _UpBlock(c * 2, c, c)
        self.out = nn.Conv2d(c, num_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        b = self.b(F.max_pool2d(e4, 2))
        x = self.d4(b, e4)
        x = self.d3(x, e3)
        x = self.d2(x, e2)
        x = self.d1(x, e1)
        return self.out(x)


class ResUNet(nn.Module):
    """Residual U-Net baseline, usually stronger and more stable than plain U-Net."""
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, base_ch=32, **kwargs):
        super().__init__()
        c = base_ch
        self.e1 = _ResidualBlock(input_channels, c)
        self.e2 = _ResidualBlock(c, c * 2)
        self.e3 = _ResidualBlock(c * 2, c * 4)
        self.e4 = _ResidualBlock(c * 4, c * 8)
        self.b = _ResidualBlock(c * 8, c * 16)
        self.d4 = _UpBlock(c * 16, c * 8, c * 8, _ResidualBlock)
        self.d3 = _UpBlock(c * 8, c * 4, c * 4, _ResidualBlock)
        self.d2 = _UpBlock(c * 4, c * 2, c * 2, _ResidualBlock)
        self.d1 = _UpBlock(c * 2, c, c, _ResidualBlock)
        self.out = nn.Conv2d(c, num_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        b = self.b(F.max_pool2d(e4, 2))
        x = self.d4(b, e4)
        x = self.d3(x, e3)
        x = self.d2(x, e2)
        x = self.d1(x, e1)
        return self.out(x)


class NestedUNetPP(nn.Module):
    """UNet++ style nested skip architecture for dense multi-scale fusion."""
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, base_ch=32, **kwargs):
        super().__init__()
        nb = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0_0 = _DoubleConv(input_channels, nb[0])
        self.conv1_0 = _DoubleConv(nb[0], nb[1])
        self.conv2_0 = _DoubleConv(nb[1], nb[2])
        self.conv3_0 = _DoubleConv(nb[2], nb[3])
        self.conv4_0 = _DoubleConv(nb[3], nb[4])
        self.conv0_1 = _DoubleConv(nb[0] + nb[1], nb[0])
        self.conv1_1 = _DoubleConv(nb[1] + nb[2], nb[1])
        self.conv2_1 = _DoubleConv(nb[2] + nb[3], nb[2])
        self.conv3_1 = _DoubleConv(nb[3] + nb[4], nb[3])
        self.conv0_2 = _DoubleConv(nb[0] * 2 + nb[1], nb[0])
        self.conv1_2 = _DoubleConv(nb[1] * 2 + nb[2], nb[1])
        self.conv2_2 = _DoubleConv(nb[2] * 2 + nb[3], nb[2])
        self.conv0_3 = _DoubleConv(nb[0] * 3 + nb[1], nb[0])
        self.conv1_3 = _DoubleConv(nb[1] * 3 + nb[2], nb[1])
        self.conv0_4 = _DoubleConv(nb[0] * 4 + nb[1], nb[0])
        self.out = nn.Conv2d(nb[0], num_classes, 1)

    def _up(self, x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(x1_0, x0_0)], 1))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(x2_0, x1_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(x1_1, x0_0)], 1))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(x3_0, x2_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(x2_1, x1_0)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2, x0_0)], 1))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(x4_0, x3_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(x3_1, x2_0)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2, x1_0)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0)], 1))
        return self.out(x0_4)


class _ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            _ConvBNAct(in_ch, out_ch, k=1, p=0) if r == 1 else _ConvBNAct(in_ch, out_ch, k=3, p=r)
            for r in rates
        ])
        for branch, r in zip(self.branches, rates):
            conv = branch.net[0]
            if r != 1:
                conv.dilation = (r, r)
                conv.padding = (r, r)
        self.project = _ConvBNAct(out_ch * len(rates), out_ch, k=1, p=0)

    def forward(self, x):
        return self.project(torch.cat([b(x) for b in self.branches], dim=1))


class DeepLabV3PlusLite(nn.Module):
    """DeepLabV3+ style ASPP baseline with a lightweight CNN encoder."""
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, base_ch=32, **kwargs):
        super().__init__()
        c = base_ch
        self.stem = _DoubleConv(input_channels, c)
        self.e2 = _ResidualBlock(c, c * 2)
        self.e3 = _ResidualBlock(c * 2, c * 4)
        self.e4 = _ResidualBlock(c * 4, c * 8)
        self.aspp = _ASPP(c * 8, c * 4)
        self.low = _ConvBNAct(c, c, k=1, p=0)
        self.dec = _DoubleConv(c * 5, c * 2)
        self.out = nn.Conv2d(c * 2, num_classes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        low = self.stem(x)
        x = self.e2(F.max_pool2d(low, 2))
        x = self.e3(F.max_pool2d(x, 2))
        x = self.e4(F.max_pool2d(x, 2))
        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec(torch.cat([x, self.low(low)], 1))
        x = F.interpolate(self.out(x), size=size, mode='bilinear', align_corners=False)
        return x


class _MixFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hidden = dim * mlp_ratio
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class SegFormerMini(nn.Module):
    """SegFormer-inspired hierarchical MLP decoder baseline without external downloads."""
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, base_ch=32, **kwargs):
        super().__init__()
        c = base_ch
        self.s1 = nn.Sequential(_ConvBNAct(input_channels, c, 7, 2, 3), _MixFFN(c))
        self.s2 = nn.Sequential(_ConvBNAct(c, c * 2, 3, 2, 1), _MixFFN(c * 2))
        self.s3 = nn.Sequential(_ConvBNAct(c * 2, c * 4, 3, 2, 1), _MixFFN(c * 4), _MixFFN(c * 4))
        self.s4 = nn.Sequential(_ConvBNAct(c * 4, c * 8, 3, 2, 1), _MixFFN(c * 8), _MixFFN(c * 8))
        dec = c * 2
        self.p1 = _ConvBNAct(c, dec, 1, p=0)
        self.p2 = _ConvBNAct(c * 2, dec, 1, p=0)
        self.p3 = _ConvBNAct(c * 4, dec, 1, p=0)
        self.p4 = _ConvBNAct(c * 8, dec, 1, p=0)
        self.fuse = _DoubleConv(dec * 4, dec)
        self.out = nn.Conv2d(dec, num_classes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        target = f1.shape[-2:]
        feats = [
            self.p1(f1),
            F.interpolate(self.p2(f2), size=target, mode='bilinear', align_corners=False),
            F.interpolate(self.p3(f3), size=target, mode='bilinear', align_corners=False),
            F.interpolate(self.p4(f4), size=target, mode='bilinear', align_corners=False),
        ]
        x = self.fuse(torch.cat(feats, dim=1))
        return F.interpolate(self.out(x), size=size, mode='bilinear', align_corners=False)
