import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F


model_urls = {
    # 'res2net50_26w_4s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_26w_4s-06e79181.pth',
    'res2net50_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_4s-06e79181.pth',
    'res2net50_48w_2s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_48w_2s-afed724a.pth',
    'res2net50_14w_8s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_14w_8s-6527dddc.pth',
    'res2net50_26w_6s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_26w_6s-19041792.pth',
    'res2net50_26w_8s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net50_26w_8s-2c7c9f12.pth',
    'res2net101_26w_4s': 'http://mc.nankai.edu.cn/projects/res2net/pretrainmodels/res2net101_26w_4s-02a759a1.pth',
}

class MEModule(nn.Module):
    """ Motion exciation module
    
    :param reduction=16
    :param n_segment=8/16
    """
    def __init__(self, channel, reduction=16, n_segment_H=16, n_segment_M=8, n_segment_L=4):
        super(MEModule, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.n_segment_H = n_segment_H
        self.n_segment_M = n_segment_M
        self.n_segment_L = n_segment_L
        self.conv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel//self.reduction,
            kernel_size=1,
            bias=False)
        self.bn1_4 = nn.BatchNorm2d(num_features=self.channel//self.reduction)
        self.bn1_8 = nn.BatchNorm2d(num_features=self.channel//self.reduction)
        self.bn1_16 = nn.BatchNorm2d(num_features=self.channel//self.reduction)

        self.conv2 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel//self.reduction,
            kernel_size=3,
            padding=1,
            groups=channel//self.reduction,
            bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)

        self.conv3 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel,
            kernel_size=1,
            bias=False)
        self.bn3_4 = nn.BatchNorm2d(num_features=self.channel)
        self.bn3_8 = nn.BatchNorm2d(num_features=self.channel)
        self.bn3_16 = nn.BatchNorm2d(num_features=self.channel)

        self.identity = nn.Identity()

    def forward(self, x_4, x_8, x_16, training=True):
        if not training:
            self.n_segment_L = self.n_segment_H
            self.n_segment_M = self.n_segment_H

        if x_4 is not None:
            x = x_4
            nt, c, h, w = x.size()
            bottleneck = self.conv1(x) # nt, c//r, h, w
            bottleneck = self.bn1_4(bottleneck) # nt, c//r, h, w
            reshape_bottleneck = bottleneck.view((-1, self.n_segment_L) + bottleneck.size()[1:])  # n, t, c//r, h, w
            t_fea, __ = reshape_bottleneck.split([self.n_segment_L-1, 1], dim=1) # n, t-1, c//r, h, w
            conv_bottleneck = self.conv2(bottleneck)  # nt, c//r, h, w
            reshape_conv_bottleneck = conv_bottleneck.view((-1, self.n_segment_L) + conv_bottleneck.size()[1:])
            __, tPlusone_fea = reshape_conv_bottleneck.split([1, self.n_segment_L-1], dim=1)  # n, t-1, c//r, h, w
            diff_fea = tPlusone_fea - t_fea # n, t-1, c//r, h, w
            diff_fea_pluszero = F.pad(diff_fea, self.pad, mode="constant", value=0)  # n, t, c//r, h, w
            diff_fea_pluszero = diff_fea_pluszero.view((-1,) + diff_fea_pluszero.size()[2:])  #nt, c//r, h, w
            y = self.avg_pool(diff_fea_pluszero)  # nt, c//r, 1, 1
            y = self.conv3(y)  # nt, c, 1, 1
            y = self.bn3_4(y)  # nt, c, 1, 1
            y = self.sigmoid(y)  # nt, c, 1, 1
            y = y - 0.5
            output = x + x * y.expand_as(x)
        elif x_8 is not None:
            x = x_8
            nt, c, h, w = x.size()
            bottleneck = self.conv1(x) # nt, c//r, h, w
            bottleneck = self.bn1_8(bottleneck) # nt, c//r, h, w
            reshape_bottleneck = bottleneck.view((-1, self.n_segment_M) + bottleneck.size()[1:])  # n, t, c//r, h, w
            t_fea, __ = reshape_bottleneck.split([self.n_segment_M-1, 1], dim=1) # n, t-1, c//r, h, w
            conv_bottleneck = self.conv2(bottleneck)  # nt, c//r, h, w
            reshape_conv_bottleneck = conv_bottleneck.view((-1, self.n_segment_M) + conv_bottleneck.size()[1:])
            __, tPlusone_fea = reshape_conv_bottleneck.split([1, self.n_segment_M-1], dim=1)  # n, t-1, c//r, h, w
            diff_fea = tPlusone_fea - t_fea # n, t-1, c//r, h, w
            diff_fea_pluszero = F.pad(diff_fea, self.pad, mode="constant", value=0)  # n, t, c//r, h, w
            diff_fea_pluszero = diff_fea_pluszero.view((-1,) + diff_fea_pluszero.size()[2:])  #nt, c//r, h, w
            y = self.avg_pool(diff_fea_pluszero)  # nt, c//r, 1, 1
            y = self.conv3(y)  # nt, c, 1, 1
            y = self.bn3_8(y)  # nt, c, 1, 1
            y = self.sigmoid(y)  # nt, c, 1, 1
            y = y - 0.5
            output = x + x * y.expand_as(x)
        else:
            x = x_16
            nt, c, h, w = x.size()
            bottleneck = self.conv1(x) # nt, c//r, h, w
            bottleneck = self.bn1_16(bottleneck) # nt, c//r, h, w
            reshape_bottleneck = bottleneck.view((-1, self.n_segment_H) + bottleneck.size()[1:])  # n, t, c//r, h, w
            t_fea, __ = reshape_bottleneck.split([self.n_segment_H-1, 1], dim=1) # n, t-1, c//r, h, w
            conv_bottleneck = self.conv2(bottleneck)  # nt, c//r, h, w
            reshape_conv_bottleneck = conv_bottleneck.view((-1, self.n_segment_H) + conv_bottleneck.size()[1:])
            __, tPlusone_fea = reshape_conv_bottleneck.split([1, self.n_segment_H-1], dim=1)  # n, t-1, c//r, h, w
            diff_fea = tPlusone_fea - t_fea # n, t-1, c//r, h, w
            diff_fea_pluszero = F.pad(diff_fea, self.pad, mode="constant", value=0)  # n, t, c//r, h, w
            diff_fea_pluszero = diff_fea_pluszero.view((-1,) + diff_fea_pluszero.size()[2:])  #nt, c//r, h, w
            y = self.avg_pool(diff_fea_pluszero)  # nt, c//r, 1, 1
            y = self.conv3(y)  # nt, c, 1, 1
            y = self.bn3_16(y)  # nt, c, 1, 1
            y = self.sigmoid(y)  # nt, c, 1, 1
            y = y - 0.5
            output = x + x * y.expand_as(x)
        
        return output

class ShiftModule(nn.Module):
    """1D Temporal convolutions, the convs are initialized to act as the "Part shift" layer
    """

    def __init__(self, input_channels, n_segment_H=16, n_segment_M=8, n_segment_L=4, n_div=8, mode='shift'):
        super(ShiftModule, self).__init__()
        self.input_channels = input_channels
        self.n_segment_H = n_segment_H
        self.n_segment_M = n_segment_M
        self.n_segment_L = n_segment_L
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div
        self.conv = nn.Conv1d(
            2*self.fold, 2*self.fold,
            kernel_size=3, padding=1, groups=2*self.fold,
            bias=False)
        # weight_size: (2*self.fold, 1, 3)
        if mode == 'shift':
            # import pdb; pdb.set_trace()
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1 # shift left
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right
            if 2*self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1 # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1 # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True

    def forward(self, x_4, x_8, x_16, training=True):
        # shift by conv
        # import pdb; pdb.set_trace()
        if not training:
            self.n_segment_L = self.n_segment_H
            self.n_segment_M = self.n_segment_H

        if x_4 is not None:
            x = x_4
            nt, c, h, w = x.size()
            n_batch = nt // self.n_segment_L
            x = x.view(n_batch, self.n_segment_L, c, h, w)
            x = x.permute([0, 3, 4, 2, 1])  # (n_batch, h, w, c, n_segment)
            x = x.contiguous().view(n_batch*h*w, c, self.n_segment_L)
            x = self.conv(x)  # (n_batch*h*w, c, n_segment)
            x = x.view(n_batch, h, w, c, self.n_segment_L)
            x = x.permute([0, 4, 3, 1, 2])  # (n_batch, n_segment, c, h, w)
            x = x.contiguous().view(nt, c, h, w)
        elif x_8 is not None:
            x = x_8
            nt, c, h, w = x.size()
            n_batch = nt // self.n_segment_M
            x = x.view(n_batch, self.n_segment_M, c, h, w)
            x = x.permute([0, 3, 4, 2, 1])  # (n_batch, h, w, c, n_segment)
            x = x.contiguous().view(n_batch*h*w, c, self.n_segment_M)
            x = self.conv(x)  # (n_batch*h*w, c, n_segment)
            x = x.view(n_batch, h, w, c, self.n_segment_M)
            x = x.permute([0, 4, 3, 1, 2])  # (n_batch, n_segment, c, h, w)
            x = x.contiguous().view(nt, c, h, w)
        else:
            x = x_16
            nt, c, h, w = x.size()
            n_batch = nt // self.n_segment_H
            x = x.view(n_batch, self.n_segment_H, c, h, w)
            x = x.permute([0, 3, 4, 2, 1])  # (n_batch, h, w, c, n_segment)
            x = x.contiguous().view(n_batch*h*w, c, self.n_segment_H)
            x = self.conv(x)  # (n_batch*h*w, c, n_segment)
            x = x.view(n_batch, h, w, c, self.n_segment_H)
            x = x.permute([0, 4, 3, 1, 2])  # (n_batch, n_segment, c, h, w)
            x = x.contiguous().view(nt, c, h, w)
        return x

class Bottle2neckShift(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample_conv=None, downsample_bn_4=None, downsample_bn_8=None, downsample_bn_16=None, 
    baseWidth=26, scale=4, num_segments_H=16, num_segments_M=8, num_segments_L=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neckShift, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))

        self.me = MEModule(width*scale, reduction=16, n_segment_H=num_segments_H, n_segment_M=num_segments_M, n_segment_L=num_segments_L)

        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(width*scale)
        self.bn1_8 = nn.BatchNorm2d(width*scale)
        self.bn1_16 = nn.BatchNorm2d(width*scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns_4 = []
        bns_8 = []
        bns_16 = []
        shifts = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride,
                padding=1, bias=False))
            bns_4.append(nn.BatchNorm2d(width))
            bns_8.append(nn.BatchNorm2d(width))
            bns_16.append(nn.BatchNorm2d(width))
            shifts.append(ShiftModule(width, n_segment_H=num_segments_H, n_segment_M=num_segments_M, n_segment_L=num_segments_L, n_div=2, mode='fixed'))
        shifts.append(ShiftModule(width, n_segment_H=num_segments_H, n_segment_M=num_segments_M, n_segment_L=num_segments_L, n_div=2, mode='shift'))

        self.convs = nn.ModuleList(convs)
        self.bns_4 = nn.ModuleList(bns_4)
        self.bns_8 = nn.ModuleList(bns_8)
        self.bns_16 = nn.ModuleList(bns_16)
        self.shifts = nn.ModuleList(shifts)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion,
                   kernel_size=1, bias=False)
        self.bn3_4 = nn.BatchNorm2d(planes * self.expansion)
        self.bn3_8 = nn.BatchNorm2d(planes * self.expansion)
        self.bn3_16 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample_conv = downsample_conv
        self.downsample_bn_4 = downsample_bn_4
        self.downsample_bn_8 = downsample_bn_8
        self.downsample_bn_16 = downsample_bn_16
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x_4, x_8, x, training=True):
        if training == True:
            residual_4 = x_4
            residual_8 = x_8
            residual_16 = x

            out_4 = self.conv1(x_4)
            out_4 = self.bn1_4(out_4)
            out_4 = self.relu(out_4)
            out_4 = self.me(out_4, None, None, True)
            spx_4 = torch.split(out_4, self.width, 1)  # 4*(nt, c/4, h, w)
            for i in range(self.nums):
                if i == 0 or self.stype == 'stage':
                    sp_4 = spx_4[i]
                else:
                    sp_4 = sp_4 + spx_4[i]
                sp_4 = self.shifts[i](sp_4, None, None, True)
                sp_4 = self.convs[i](sp_4)
                sp_4 = self.relu(self.bns_4[i](sp_4))
                if i == 0:
                    out_4 = sp_4
                else:
                    out_4 = torch.cat((out_4, sp_4), 1)
            last_sp_4 = spx_4[self.nums]
            last_sp_4 = self.shifts[self.nums](last_sp_4, None, None, True)
            if self.scale != 1 and self.stype == 'normal':
                out_4 = torch.cat((out_4, last_sp_4), 1)
            elif self.scale != 1 and self.stype == 'stage':
                out_4 = torch.cat((out_4, self.pool(last_sp_4)), 1)
            out_4 = self.conv3(out_4)
            out_4 = self.bn3_4(out_4)


            out_8 = self.conv1(x_8)
            out_8 = self.bn1_8(out_8)
            out_8 = self.relu(out_8)
            out_8 = self.me(None, out_8, None, True)
            spx_8 = torch.split(out_8, self.width, 1)  # 4*(nt, c/4, h, w)
            for i in range(self.nums):
                if i == 0 or self.stype == 'stage':
                    sp_8 = spx_8[i]
                else:
                    sp_8 = sp_8 + spx_8[i]
                sp_8 = self.shifts[i](None, sp_8, None, True)
                sp_8 = self.convs[i](sp_8)
                sp_8 = self.relu(self.bns_8[i](sp_8))
                if i == 0:
                    out_8 = sp_8
                else:
                    out_8 = torch.cat((out_8, sp_8), 1)
            last_sp_8 = spx_8[self.nums]
            last_sp_8 = self.shifts[self.nums](None, last_sp_8, None, True)
            if self.scale != 1 and self.stype == 'normal':
                out_8 = torch.cat((out_8, last_sp_8), 1)
            elif self.scale != 1 and self.stype == 'stage':
                out_8 = torch.cat((out_8, self.pool(last_sp_8)), 1)
            out_8 = self.conv3(out_8)
            out_8 = self.bn3_8(out_8)


            out_16 = self.conv1(x)
            out_16 = self.bn1_16(out_16)
            out_16 = self.relu(out_16)
            out_16 = self.me(None, None, out_16, True)
            spx_16 = torch.split(out_16, self.width, 1)  # 4*(nt, c/4, h, w)
            for i in range(self.nums):
                if i == 0 or self.stype == 'stage':
                    sp_16 = spx_16[i]
                else:
                    sp_16 = sp_16 + spx_16[i]
                sp_16 = self.shifts[i](None, None, sp_16, True)
                sp_16 = self.convs[i](sp_16)
                sp_16 = self.relu(self.bns_16[i](sp_16))
                if i == 0:
                    out_16 = sp_16
                else:
                    out_16 = torch.cat((out_16, sp_16), 1)
            last_sp_16 = spx_16[self.nums]
            last_sp_16 = self.shifts[self.nums](None, None, last_sp_16, True)
            if self.scale != 1 and self.stype == 'normal':
                out_16 = torch.cat((out_16, last_sp_16), 1)
            elif self.scale != 1 and self.stype == 'stage':
                out_16 = torch.cat((out_16, self.pool(last_sp_16)), 1)
            out_16 = self.conv3(out_16)
            out_16 = self.bn3_16(out_16)


            if self.downsample_conv is not None:
                residual_4 = self.downsample_bn_4(self.downsample_conv(x_4))
                residual_8 = self.downsample_bn_8(self.downsample_conv(x_8))
                residual_16 = self.downsample_bn_16(self.downsample_conv(x))
            out_4 += residual_4
            out_4 = self.relu(out_4)
            out_8 += residual_8
            out_8 = self.relu(out_8)
            out_16 += residual_16
            out_16 = self.relu(out_16)

            return out_4, out_8, out_16
        else:
            if x_4 is not None:
                residual = x_4
                out = self.conv1(x_4)
                out = self.bn1_4(out)
                out = self.relu(out)
                out = self.me(out, None, None, False)
                spx_4 = torch.split(out, self.width, 1)  # 4*(nt, c/4, h, w)
                for i in range(self.nums):
                    if i == 0 or self.stype == 'stage':
                        sp_4 = spx_4[i]
                    else:
                        sp_4 = sp_4 + spx_4[i]
                    sp_4 = self.shifts[i](sp_4, None, None, False)
                    sp_4 = self.convs[i](sp_4)
                    sp_4 = self.relu(self.bns_4[i](sp_4))
                    if i == 0:
                        out = sp_4
                    else:
                        out = torch.cat((out, sp_4), 1)
                last_sp_4 = spx_4[self.nums]
                last_sp_4 = self.shifts[self.nums](last_sp_4, None, None, False)
                if self.scale != 1 and self.stype == 'normal':
                    out = torch.cat((out, last_sp_4), 1)
                elif self.scale != 1 and self.stype == 'stage':
                    out = torch.cat((out, self.pool(last_sp_4)), 1)
                out = self.conv3(out)
                out = self.bn3_4(out)
                if self.downsample_conv is not None:
                    residual = self.downsample_bn_4(self.downsample_conv(x_4))
            elif x_8 is not None:
                residual = x_8
                out = self.conv1(x_8)
                out = self.bn1_8(out)
                out = self.relu(out)
                out = self.me(None, out, None, False)
                spx_8 = torch.split(out, self.width, 1)  # 4*(nt, c/4, h, w)
                for i in range(self.nums):
                    if i == 0 or self.stype == 'stage':
                        sp_8 = spx_8[i]
                    else:
                        sp_8 = sp_8 + spx_8[i]
                    sp_8 = self.shifts[i](None, sp_8, None, False)
                    sp_8 = self.convs[i](sp_8)
                    sp_8 = self.relu(self.bns_8[i](sp_8))
                    if i == 0:
                        out = sp_8
                    else:
                        out = torch.cat((out, sp_8), 1)
                last_sp_8 = spx_8[self.nums]
                last_sp_8 = self.shifts[self.nums](None, last_sp_8, None, False)
                if self.scale != 1 and self.stype == 'normal':
                    out = torch.cat((out, last_sp_8), 1)
                elif self.scale != 1 and self.stype == 'stage':
                    out = torch.cat((out, self.pool(last_sp_8)), 1)
                out = self.conv3(out)
                out = self.bn3_8(out)
                if self.downsample_conv is not None:
                    residual = self.downsample_bn_8(self.downsample_conv(x_8))
            else:
                residual = x
                out = self.conv1(x)
                out = self.bn1_16(out)
                out = self.relu(out)
                out = self.me(None, None, out, False)
                spx_16 = torch.split(out, self.width, 1)  # 4*(nt, c/4, h, w)
                for i in range(self.nums):
                    if i == 0 or self.stype == 'stage':
                        sp_16 = spx_16[i]
                    else:
                        sp_16 = sp_16 + spx_16[i]
                    sp_16 = self.shifts[i](None, None, sp_16, False)
                    sp_16 = self.convs[i](sp_16)
                    sp_16 = self.relu(self.bns_16[i](sp_16))
                    if i == 0:
                        out = sp_16
                    else:
                        out = torch.cat((out, sp_16), 1)
                last_sp_16 = spx_16[self.nums]
                last_sp_16 = self.shifts[self.nums](None, None, last_sp_16, False)
                if self.scale != 1 and self.stype == 'normal':
                    out = torch.cat((out, last_sp_16), 1)
                elif self.scale != 1 and self.stype == 'stage':
                    out = torch.cat((out, self.pool(last_sp_16)), 1)
                out = self.conv3(out)
                out = self.bn3_16(out)
                if self.downsample_conv is not None:
                    residual = self.downsample_bn_16(self.downsample_conv(x))
            
            out += residual
            out = self.relu(out)
            
            return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_segments_H=16, num_segments_M=8, num_segments_L=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1_4 = nn.BatchNorm2d(64)
        self.bn1_8 = nn.BatchNorm2d(64)
        self.bn1_16 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], num_segments_H=num_segments_H, num_segments_M=num_segments_M, num_segments_L=num_segments_L)
        self.layer2 = self._make_layer(block, 128, layers[1], num_segments_H=num_segments_H, num_segments_M=num_segments_M, num_segments_L=num_segments_L, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], num_segments_H=num_segments_H, num_segments_M=num_segments_M, num_segments_L=num_segments_L, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], num_segments_H=num_segments_H, num_segments_M=num_segments_M, num_segments_L=num_segments_L, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, num_segments_H=16, num_segments_M=8, num_segments_L=4, stride=1):
        downsample_conv = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample_conv = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
            downsample_bn_4 = nn.BatchNorm2d(planes * block.expansion)
            downsample_bn_8 = nn.BatchNorm2d(planes * block.expansion)
            downsample_bn_16 = nn.BatchNorm2d(planes * block.expansion)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample_conv, downsample_bn_4, downsample_bn_8, downsample_bn_16,
            stype='stage', baseWidth = self.baseWidth, scale=self.scale, num_segments_H=num_segments_H, num_segments_M=num_segments_M, num_segments_L=num_segments_L))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth = self.baseWidth, scale=self.scale, num_segments_H=num_segments_H, num_segments_M=num_segments_M, num_segments_L=num_segments_L))

        return nn.Sequential(*layers)

    def forward(self, x_4, x_8, x_16, training=True):
        if training == True:
            x_4 = self.conv1(x_4)
            x_4 = self.bn1_4(x_4)
            x_4 = self.relu(x_4)
            x_4 = self.maxpool(x_4)

            x_8 = self.conv1(x_8)
            x_8 = self.bn1_8(x_8)
            x_8 = self.relu(x_8)
            x_8 = self.maxpool(x_8)

            x_16 = self.conv1(x_16)
            x_16 = self.bn1_16(x_16)
            x_16 = self.relu(x_16)
            x_16 = self.maxpool(x_16)

            for i in range(len(self.layer1)):
                x_4, x_8, x_16 = self.layer1[i](x_4, x_8, x_16, True)
            for i in range(len(self.layer2)):
                x_4, x_8, x_16 = self.layer2[i](x_4, x_8, x_16, True)
            for i in range(len(self.layer3)):
                x_4, x_8, x_16 = self.layer3[i](x_4, x_8, x_16, True)
            for i in range(len(self.layer4)):
                x_4, x_8, x_16 = self.layer4[i](x_4, x_8, x_16, True)

            x_4 = self.avgpool(x_4)    
            x_4 = x_4.view(x_4.size(0), -1)
            x_4 = self.fc(x_4)

            x_8 = self.avgpool(x_8)    
            x_8 = x_8.view(x_8.size(0), -1)
            x_8 = self.fc(x_8)

            x_16 = self.avgpool(x_16)    
            x_16 = x_16.view(x_16.size(0), -1)
            x_16 = self.fc(x_16)
        
            return x_4, x_8, x_16
        else:
            if x_4 is not None:
                x = self.conv1(x_4)
                x = self.bn1_4(x)
                x = self.relu(x)
                x = self.maxpool(x)
                for i in range(len(self.layer1)):
                    x = self.layer1[i](x, None, None, False)
                for i in range(len(self.layer2)):
                    x = self.layer2[i](x, None, None, False)
                for i in range(len(self.layer3)):
                    x = self.layer3[i](x, None, None, False)
                for i in range(len(self.layer4)):
                    x = self.layer4[i](x, None, None, False)
                x = self.avgpool(x)    
                x = x.view(x.size(0), -1)
                x = self.fc(x)
            elif x_8 is not None:
                x = self.conv1(x_8)
                x = self.bn1_8(x)
                x = self.relu(x)
                x = self.maxpool(x)
                for i in range(len(self.layer1)):
                    x = self.layer1[i](None, x, None, False)
                for i in range(len(self.layer2)):
                    x = self.layer2[i](None, x, None, False)
                for i in range(len(self.layer3)):
                    x = self.layer3[i](None, x, None, False)
                for i in range(len(self.layer4)):
                    x = self.layer4[i](None, x, None, False)
                x = self.avgpool(x)    
                x = x.view(x.size(0), -1)
                x = self.fc(x)
            else:
                x = self.conv1(x_16)
                x = self.bn1_16(x)
                x = self.relu(x)
                x = self.maxpool(x)
                for i in range(len(self.layer1)):
                    x = self.layer1[i](None, None, x, False)
                for i in range(len(self.layer2)):
                    x = self.layer2[i](None, None, x, False)
                for i in range(len(self.layer3)):
                    x = self.layer3[i](None, None, x, False)
                for i in range(len(self.layer4)):
                    x = self.layer4[i](None, None, x, False)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
            return x



def resnet50(pretrained=False, shift='TSM', num_segments_H=16, num_segments_M=8, num_segments_L=4, reduction=1, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neckShift, [3, 4, 6, 3], baseWidth = 26, scale = 4, num_segments_H=num_segments_H, num_segments_M=num_segments_M, num_segments_L=num_segments_L, **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['res2net50_26w_4s'])
        new_state_dict =  model.state_dict()
        for k, v in new_state_dict.items():
            if 'downsample_conv' in k:
                new_state_dict.update({k:pretrained_dict[k.replace('_conv', '.0')]})
            elif 'downsample_bn_4' in k and 'num_batches_tracked' not in k:
                new_state_dict.update({k:pretrained_dict[k.replace('_bn_4', '.1')]})
            elif 'downsample_bn_8' in k and 'num_batches_tracked' not in k:
                new_state_dict.update({k:pretrained_dict[k.replace('_bn_8', '.1')]})
            elif 'downsample_bn_16' in k and 'num_batches_tracked' not in k:
                new_state_dict.update({k:pretrained_dict[k.replace('_bn_16', '.1')]})
            elif (k.replace('_4', '') in pretrained_dict):
                new_state_dict.update({k:pretrained_dict[k.replace('_4', '')]})
            elif (k.replace('_8', '') in pretrained_dict):
                new_state_dict.update({k:pretrained_dict[k.replace('_8', '')]})
            elif (k.replace('_16', '') in pretrained_dict):
                new_state_dict.update({k:pretrained_dict[k.replace('_16', '')]})
            elif (k in pretrained_dict):
                new_state_dict.update({k:pretrained_dict[k]})
        model.load_state_dict(new_state_dict)

    return model


if __name__ == '__main__':
    images = torch.rand(8, 3, 224, 224)
    model = tea50_8f(pretrained=True)
    output = model(images)
    print(output.size())