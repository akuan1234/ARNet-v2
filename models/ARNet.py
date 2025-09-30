import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

import numpy as np
from smt import smt_b

from thop import profile
from torch import Tensor
from typing import List
from einops import rearrange

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.relu(x)
        return x


class FAP(nn.Module):
    def __init__(self, in_c, num_groups=4, hidden_dim=None, in_ch=-1):
        super().__init__()
        self.num_groups = num_groups

        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups
        self.expand_conv = BasicConv2d(in_c, expand_dim, 1)
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_groups * hidden_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, num_groups * hidden_dim, 1),
            nn.Softmax(dim=1),
        )

        self.interact = nn.ModuleDict()
        self.interact["0"] = BasicConv2d(hidden_dim, 2 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = BasicConv2d(2 * hidden_dim, 3 * hidden_dim, 3, 1, 1)
        self.conv1 = BasicConv2d(4 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
        self.conv2 = BasicConv2d(4 * hidden_dim, 4 * hidden_dim, 3, 1, 1)
        self.interact["2"] = BasicConv2d(4 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
        self.interact["1"] = BasicConv2d(hidden_dim, 2 * hidden_dim, 3, 1, 1)
        self.interact["3"] = BasicConv2d(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)

        self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
        self.final_relu = nn.ReLU(True)

    def forward(self, x):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)

        outs = []
        # x1 = xs[0]
        branch_out0 = self.interact["0"](xs[0])

        # outs.append(branch_out.chunk(3, dim=1))
        # x01, x02, x03 = branch_out0.chunk(3, dim=1)

        branch_out1 = self.interact["1"](xs[1])
        branch_out2 = self.interact["1"](xs[2])
        branch_out3 = self.interact["1"](xs[3])
        branch_out01 = torch.cat((branch_out0, branch_out1), 1)
        # print(branch_out01.shape[1])
        branch_out01 = self.interact["2"](branch_out01)
        branch_out012 = torch.cat((branch_out01, branch_out2), 1)
        branch_out012 = self.interact["2"](branch_out012)
        branch_out0123 = torch.cat((branch_out012, branch_out3), 1)
        branch_out0123 = self.interact["2"](branch_out0123)

        out21 = torch.cat((branch_out01, branch_out0), 1)
        out21 = self.conv1(out21)
        out22 = torch.cat((branch_out01, branch_out012), 1)
        out22 = self.conv1(out22)
        out23 = torch.cat((branch_out0123, branch_out012), 1)
        out23 = self.conv1(out23)
        branch_out22 = torch.cat((out21, out22), 1)
        branch_out22 = self.conv1(branch_out22)
        branch_out23 = torch.cat((branch_out22, out23), 1)
        branch_out23 = self.conv1(branch_out23)

        out31 = torch.cat((branch_out22, out21), 1)
        out31 = self.conv1(out31)
        out32 = torch.cat((branch_out23, branch_out22), 1)
        out32 = self.conv1(out32)
        branch_out32 = torch.cat((out31, out32), 1)
        branch_out32 = self.conv1(branch_out32)

        out = torch.cat((out31, branch_out32), 1)
        out = self.conv2(out)

        gate = self.gate_genator(out)
        out = self.fuse(out * gate)
        return self.final_relu(out + x)

class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class MSE(nn.Module):
    expansion = 1  # 类属性，目前设定为1，但没有在代码中直接使用。这个属性在某些网络结构中用于标识特征图扩展的倍数。

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(MSE, self).__init__()
        # branch1
        self.atrConv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, dilation=3, padding=3, stride=1), nn.BatchNorm2d(planes),
            nn.PReLU()
        )
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1,  # 使用3x3的卷积核，保持输入输出尺寸不变
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)  # 批量归一化，稳定和加速训练
        self.relu = nn.ReLU(inplace=True)  # 在地方使用，增加非线性
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample1 = upsample
        self.stride1 = stride
        # barch2
        self.conv3 = nn.Conv2d(inplanes, inplanes, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.relu3 = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv4 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv4 = nn.Conv2d(inplanes, planes, kernel_size=5, stride=stride,
                                   padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv_cat = BasicConv2d(3 * inplanes, inplanes, 3, padding=1)
        self.upsample2 = upsample
        self.stride2 = stride

    def forward(self, x):
        residual = x

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)

        if self.upsample1 is not None:
            residual = self.upsample1(x)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out2 = self.relu3(out2)

        out2 = self.conv4(out2)
        out2 = self.bn4(out2)
        out3 = self.atrConv(x)
        if self.upsample2 is not None:
            residual = self.upsample2(x)
        out = self.conv_cat(torch.cat((out1, out2, out3), 1))
        out += residual
        out = self.relu(out)

        return out

def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input

def get_sobel(in_chan, out_chan):
    '''
    filter_x = np.array([
        [3, 0, -3],
        [10, 0, -10],
        [3, 0, -3],
    ]).astype(np.float32)
    filter_y = np.array([
        [3, 10, 3],
        [0, 0, 0],
        [-3, -10, -3],
    ]).astype(np.float32)
    '''
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)
    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y
class BEM(nn.Module):
    def __init__(self):
        super(BEM, self).__init__()

        self.sobel_x4, self.sobel_y4 = get_sobel(64, 1)
        self.block = nn.Sequential(
            ConvBNR(128, 64, 3),
            ConvBNR(64, 64, 3),
            nn.Conv2d(64, 1, 1))

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = run_sobel(self.sobel_x4, self.sobel_y4, x1)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out

class DWFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(DWFFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class SGA(nn.Module):
    def __init__(self, dim=128, num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super(SGA, self).__init__()
        self.norm1 = Layernorm(dim, LayerNorm_type)
        self.attn = SelfAttention(dim, num_heads, bias)
        self.norm2 = Layernorm(dim, LayerNorm_type)
        self.ffn = DWFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

class Layernorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(Layernorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        q = self.qkv1conv(self.qkv_0(x))
        k = self.qkv2conv(self.qkv_1(x))
        v = self.qkv3conv(self.qkv_2(x))
        if mask is not None:
            q = q * mask
            k = k * mask

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class ARNet(nn.Module):
    def __init__(self, channel=64):
        super(ARNet, self).__init__()

        self.smt = smt_b()

        self.Translayer2_1 = BasicConv2d(128, 64, 1)
        self.Translayer3_1 = BasicConv2d(256, 64, 1)
        self.Translayer4_1 = BasicConv2d(512, 64, 1)

        self.MSE = MSE(64, 64)

        self.BEM = BEM()
        self.sobel_x4, self.sobel_y4 = get_sobel(64, 1)

        self.CIIM1 = nn.Sequential(FAP(64, num_groups=4, hidden_dim=32))
        self.CIIM2 = nn.Sequential(FAP(64, num_groups=4, hidden_dim=32))
        self.CIIM3 = nn.Sequential(FAP(64, num_groups=4, hidden_dim=32))

        self.uconv1 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1, relu=True)
        self.uconv3 = BasicConv2d(128, 64, kernel_size=3, stride=1, padding=1, relu=True)

        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.attn = SGA(dim=64, num_heads=8)
        self.attn1 = SGA(dim=64, num_heads=8)
        self.attn2 = SGA(dim=64, num_heads=8)
        self.attn3 = SGA(dim=64, num_heads=8)
        self.attn4 = SGA(dim=64, num_heads=8)
        self.attn5 = SGA(dim=64, num_heads=8)

        self.predtrans1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):

        rgb_list = self.smt(x)

        r1 = rgb_list[3]  # 512,12
        r2 = rgb_list[2]  # 256,24
        r3 = rgb_list[1]  # 128,48
        r4 = rgb_list[0]  # 64,96,96

        r3 = self.Translayer2_1(r3)  # [1, 64, 44, 44]
        r2 = self.Translayer3_1(r2)
        r1 = self.Translayer4_1(r1)  # 都变为64的通道

        edge = self.BEM(r1, r4)
        edge_att = torch.sigmoid(edge)

        r1_ = self.up1(r1)  # 将图像大小扩大一倍
        r12_ = torch.cat((r2, r1_), 1)
        r12_ = self.uconv3(r12_)
        r12_ = self.up1(r12_)
        r123_ = torch.cat((r3, r12_), 1)
        r123_ = self.up1(r123_)
        r123_ = self.uconv3(r123_)
        r1234_ = torch.cat((r4, r123_), 1)
        r1234_ = self.uconv3(r1234_)

        x1 = self.CIIM1(r2 + r1_)
        r1234_1 = F.interpolate(r1234_, size=26, mode='bilinear')
        xg1 = F.interpolate(edge_att, size=26, mode='bilinear')
        x1_b = self.attn(x1, xg1)
        x1_g = self.attn1(x1, r1234_1)
        x1 = x1 + x1_b + x1_g
        x1 = self.uconv1(x1)
        x1 = self.MSE(x1)

        x1 = self.up1(x1)
        x12 = self.CIIM2(x1 + r3)
        r1234_2 = F.interpolate(r1234_, size=52, mode='bilinear')
        xg2 = F.interpolate(edge_att, size=52, mode='bilinear')
        x12_b = self.attn2(x12, xg2)
        x12_g = self.attn3(x12, r1234_2)
        x12 = x12 + x12_b + x12_g
        x12 = self.uconv1(x12)
        x12 = self.MSE(x12)

        x12 = self.up1(x12)
        x123 = self.CIIM3(x12 + r4)
        x123_b = self.attn4(x123, edge_att)
        x123_g = self.attn5(x123, r1234_)
        x123 = x123 + x123_b + x123_g
        x123 = self.uconv1(x123)
        x123 = self.MSE(x123)

        r123 = F.interpolate(self.predtrans1(x123), size=416, mode='bilinear')
        r12 = F.interpolate(self.predtrans2(x12), size=416, mode='bilinear')
        r1 = F.interpolate(self.predtrans3(x1), size=416, mode='bilinear')
        edge_att = F.interpolate(edge_att, size=416, mode='bilinear')

        return r123, r12, r1, edge_att

    def load_pre(self, pre_model):
        self.smt.load_state_dict(torch.load(pre_model)['model'])
        print(f"loading pre_model ${pre_model}")


if __name__ == '__main__':
    x = torch.randn(1, 3, 416, 416)
    flops, params = profile(ARNet(x), (x,))
    print('flops: %.2f G, parms: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
