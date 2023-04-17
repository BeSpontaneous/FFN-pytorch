#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import torch
import torch.nn as nn

from slowfast.models.common import drop_path
from slowfast.models.nonlocal_helper import Nonlocal
from slowfast.models.operators import SE, Swish


def get_trans_func(name):
    """
    Retrieves the transformation module by name.
    """
    trans_funcs = {
        "bottleneck_transform_FFN": BottleneckTransform_FFN,
        "basic_transform": BasicTransform,
        "x3d_transform": X3DTransform,
    }
    assert (
        name in trans_funcs.keys()
    ), "Transformation function '{}' not supported".format(name)
    return trans_funcs[name]


class BasicTransform(nn.Module):
    """
    Basic transformation: Tx3x3, 1x3x3, where T is the size of temporal kernel.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner=None,
        num_groups=1,
        stride_1x1=None,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm3d,
        block_idx=0,
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the first
                convolution in the basic block.
            stride (int): the stride of the bottleneck.
            dim_inner (None): the inner dimension would not be used in
                BasicTransform.
            num_groups (int): number of groups for the convolution. Number of
                group is always 1 for BasicTransform.
            stride_1x1 (None): stride_1x1 will not be used in BasicTransform.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(BasicTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._construct(dim_in, dim_out, stride, dilation, norm_module)

    def _construct(self, dim_in, dim_out, stride, dilation, norm_module):
        # Tx3x3, BN, ReLU.
        self.a = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=[self.temp_kernel_size, 3, 3],
            stride=[1, stride, stride],
            padding=[int(self.temp_kernel_size // 2), 1, 1],
            bias=False,
        )
        self.a_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)
        # 1x3x3, BN.
        self.b = nn.Conv3d(
            dim_out,
            dim_out,
            kernel_size=[1, 3, 3],
            stride=[1, 1, 1],
            padding=[0, dilation, dilation],
            dilation=[1, dilation, dilation],
            bias=False,
        )

        self.b.final_conv = True

        self.b_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )

        self.b_bn.transform_final_bn = True

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)

        x = self.b(x)
        x = self.b_bn(x)
        return x


class X3DTransform(nn.Module):
    """
    X3D transformation: 1x1x1, Tx3x3 (channelwise, num_groups=dim_in), 1x1x1,
        augmented with (optional) SE (squeeze-excitation) on the 3x3x3 output.
        T is the temporal kernel size (defaulting to 3)
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner,
        num_groups,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm3d,
        se_ratio=0.0625,
        swish_inner=True,
        block_idx=0,
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            se_ratio (float): if > 0, apply SE to the Tx3x3 conv, with the SE
                channel dimensionality being se_ratio times the Tx3x3 conv dim.
            swish_inner (bool): if True, apply swish to the Tx3x3 conv, otherwise
                apply ReLU to the Tx3x3 conv.
        """
        super(X3DTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._se_ratio = se_ratio
        self._swish_inner = swish_inner
        self._stride_1x1 = stride_1x1
        self._block_idx = block_idx
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            dilation,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        dilation,
        norm_module,
    ):
        (str1x1, str3x3) = (stride, 1) if self._stride_1x1 else (1, stride)

        # 1x1x1, BN, ReLU.
        self.a = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=[1, 1, 1],
            stride=[1, str1x1, str1x1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.a_bn = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)

        # Tx3x3, BN, ReLU.
        self.b = nn.Conv3d(
            dim_inner,
            dim_inner,
            [self.temp_kernel_size, 3, 3],
            stride=[1, str3x3, str3x3],
            padding=[int(self.temp_kernel_size // 2), dilation, dilation],
            groups=num_groups,
            bias=False,
            dilation=[1, dilation, dilation],
        )
        self.b_bn = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )

        # Apply SE attention or not
        use_se = True if (self._block_idx + 1) % 2 else False
        if self._se_ratio > 0.0 and use_se:
            self.se = SE(dim_inner, self._se_ratio)

        if self._swish_inner:
            self.b_relu = Swish()
        else:
            self.b_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x1x1, BN.
        self.c = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.c_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.c_bn.transform_final_bn = True

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x



class BottleneckTransform_FFN(nn.Module):
    """
    Bottleneck transformation: Tx1x1, 1x3x3, 1x1x1, where T is the size of
        temporal kernel.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner,
        num_groups,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm3d,
        block_idx=0,
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the first
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(BottleneckTransform_FFN, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._stride_1x1 = stride_1x1
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            dilation,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        dilation,
        norm_module,
    ):
        (str1x1, str3x3) = (stride, 1) if self._stride_1x1 else (1, stride)

        # Tx1x1, BN, ReLU.
        self.a = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=[self.temp_kernel_size, 1, 1],
            stride=[1, str1x1, str1x1],
            padding=[int(self.temp_kernel_size // 2), 0, 0],
            bias=False,
        )
        self.a_bn_4 = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_bn_8 = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_bn_16 = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x3x3, BN, ReLU.
        self.b = nn.Conv3d(
            dim_inner,
            dim_inner,
            [1, 3, 3],
            stride=[1, str3x3, str3x3],
            padding=[0, dilation, dilation],
            groups=num_groups,
            bias=False,
            dilation=[1, dilation, dilation],
        )
        self.b_bn_4 = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.b_bn_8 = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.b_bn_16 = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )
        self.b_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x1x1, BN.
        self.c = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.c.final_conv = True

        self.c_bn_4 = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.c_bn_8 = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.c_bn_16 = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.c_bn_4.transform_final_bn = True
        self.c_bn_8.transform_final_bn = True
        self.c_bn_16.transform_final_bn = True

        self.adaconv_4 = nn.Conv3d(dim_in, dim_in, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=dim_in, bias=False)
        self.adaconv_8 = nn.Conv3d(dim_in, dim_in, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=dim_in, bias=False)
        self.adaconv_16 = nn.Conv3d(dim_in, dim_in, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=dim_in, bias=False)
        self.adaconv_4.weight.data.normal_(0,1e-3)
        self.adaconv_8.weight.data.normal_(0,1e-3)
        self.adaconv_16.weight.data.normal_(0,1e-3)

    def forward(self, x_4, x_8, x_16, training=False):
        # Explicitly forward every layer.
        # Branch2a.
        if training == True:
            iden_4 = x_4
            x_4 = self.adaconv_4(x_4)
            x_4 = x_4 + iden_4

            x_4 = self.a(x_4)
            x_4 = self.a_bn_4(x_4)
            x_4 = self.a_relu(x_4)
            
            x_4 = self.b(x_4)
            x_4 = self.b_bn_4(x_4)
            x_4 = self.b_relu(x_4)
            
            x_4 = self.c(x_4)
            x_4 = self.c_bn_4(x_4)
            
            
            
            iden_8 = x_8
            x_8 = self.adaconv_8(x_8)
            x_8 = x_8 + iden_8
            
            x_8 = self.a(x_8)
            x_8 = self.a_bn_8(x_8)
            x_8 = self.a_relu(x_8)
            
            x_8 = self.b(x_8)
            x_8 = self.b_bn_8(x_8)
            x_8 = self.b_relu(x_8)
            
            x_8 = self.c(x_8)
            x_8 = self.c_bn_8(x_8)
            
            

            iden_16 = x_16
            x_16 = self.adaconv_16(x_16)
            x_16 = x_16 + iden_16
            
            x_16 = self.a(x_16)
            x_16 = self.a_bn_16(x_16)
            x_16 = self.a_relu(x_16)
            
            x_16 = self.b(x_16)
            x_16 = self.b_bn_16(x_16)
            x_16 = self.b_relu(x_16)
            
            x_16 = self.c(x_16)
            x_16 = self.c_bn_16(x_16)
            
            return x_4, x_8, x_16
        else:
            if x_4 is not None:
                iden_4 = x_4
                x_4 = self.adaconv_4(x_4)
                x_4 = x_4 + iden_4

                x = self.a(x_4)
                x = self.a_bn_4(x)
                x = self.a_relu(x)

                # Branch2b.
                x = self.b(x)
                x = self.b_bn_4(x)
                x = self.b_relu(x)

                # Branch2c
                x = self.c(x)
                x = self.c_bn_4(x)
            elif x_8 is not None:
                iden_8 = x_8
                x_8 = self.adaconv_8(x_8)
                x_8 = x_8 + iden_8
                
                x = self.a(x_8)
                x = self.a_bn_8(x)
                x = self.a_relu(x)

                # Branch2b.
                x = self.b(x)
                x = self.b_bn_8(x)
                x = self.b_relu(x)

                # Branch2c
                x = self.c(x)
                x = self.c_bn_8(x)
            else:
                iden_16 = x_16
                x_16 = self.adaconv_16(x_16)
                x_16 = x_16 + iden_16

                x = self.a(x_16)
                x = self.a_bn_16(x)
                x = self.a_relu(x)

                # Branch2b.
                x = self.b(x)
                x = self.b_bn_16(x)
                x = self.b_relu(x)

                # Branch2c
                x = self.c(x)
                x = self.c_bn_16(x)
            
            return x



class ResBlock_FFN(nn.Module):
    """
    Residual block.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        trans_func,
        dim_inner,
        num_groups=1,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm3d,
        block_idx=0,
        drop_connect_rate=0.0,
    ):
        """
        ResBlock class constructs redisual blocks. More details can be found in:
            Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
            "Deep residual learning for image recognition."
            https://arxiv.org/abs/1512.03385
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            trans_func (string): transform function to be used to construct the
                bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            drop_connect_rate (float): basic rate at which blocks are dropped,
                linearly increases from input to output blocks.
        """
        super(ResBlock_FFN, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._drop_connect_rate = drop_connect_rate
        self._construct(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            trans_func,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation,
            norm_module,
            block_idx,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        trans_func,
        dim_inner,
        num_groups,
        stride_1x1,
        inplace_relu,
        dilation,
        norm_module,
        block_idx,
    ):
        # Use skip connection with projection if dim or res change.
        if (dim_in != dim_out) or (stride != 1):
            self.branch1 = nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size=1,
                stride=[1, stride, stride],
                padding=0,
                bias=False,
                dilation=1,
            )
            self.branch1_bn_4 = norm_module(
                num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
            )
            self.branch1_bn_8 = norm_module(
                num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
            )
            self.branch1_bn_16 = norm_module(
                num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
            )
        self.branch2 = trans_func(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner,
            num_groups,
            stride_1x1=stride_1x1,
            inplace_relu=inplace_relu,
            dilation=dilation,
            norm_module=norm_module,
            block_idx=block_idx,
        )
        self.relu = nn.ReLU(self._inplace_relu)

    def forward(self, x_4, x_8, x_16, training=False):
        if training == True:
            f_x_4, f_x_8, f_x_16 = self.branch2(x_4, x_8, x_16, True)
            if self.training and self._drop_connect_rate > 0.0:
                f_x_4 = drop_path(f_x_4, self._drop_connect_rate)
                f_x_8 = drop_path(f_x_8, self._drop_connect_rate)
                f_x_16 = drop_path(f_x_16, self._drop_connect_rate)
            if hasattr(self, "branch1"):
                x_4 = self.branch1_bn_4(self.branch1(x_4)) + f_x_4
                x_8 = self.branch1_bn_8(self.branch1(x_8)) + f_x_8
                x_16 = self.branch1_bn_16(self.branch1(x_16)) + f_x_16
            else:
                x_4 = x_4 + f_x_4
                x_8 = x_8 + f_x_8
                x_16 = x_16 + f_x_16
            x_4 = self.relu(x_4)
            x_8 = self.relu(x_8)
            x_16 = self.relu(x_16)
            
            return x_4, x_8, x_16
        else:
            if x_4 is not None:
                x = x_4
                f_x = self.branch2(x, None, None, False)
                if self.training and self._drop_connect_rate > 0.0:
                    f_x = drop_path(f_x, self._drop_connect_rate)
                if hasattr(self, "branch1"):
                    x = self.branch1_bn_4(self.branch1(x)) + f_x
                else:
                    x = x + f_x
                x = self.relu(x)
            elif x_8 is not None:
                x = x_8
                f_x = self.branch2(None, x, None, False)
                if self.training and self._drop_connect_rate > 0.0:
                    f_x = drop_path(f_x, self._drop_connect_rate)
                if hasattr(self, "branch1"):
                    x = self.branch1_bn_8(self.branch1(x)) + f_x
                else:
                    x = x + f_x
                x = self.relu(x)
            else:
                x = x_16
                f_x = self.branch2(None, None, x, False)
                if self.training and self._drop_connect_rate > 0.0:
                    f_x = drop_path(f_x, self._drop_connect_rate)
                if hasattr(self, "branch1"):
                    x = self.branch1_bn_16(self.branch1(x)) + f_x
                else:
                    x = x + f_x
                x = self.relu(x)
            
            return x


class ResStage_FFN(nn.Module):
    """
    Stage of 3D ResNet. It expects to have one or more tensors as input for
        single pathway (C2D, I3D, Slow), and multi-pathway (SlowFast) cases.
        More details can be found here:

        Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
        "SlowFast networks for video recognition."
        https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        stride,
        temp_kernel_sizes,
        num_blocks,
        dim_inner,
        num_groups,
        num_block_temp_kernel,
        nonlocal_inds,
        nonlocal_group,
        nonlocal_pool,
        dilation,
        instantiation="softmax",
        trans_func_name="bottleneck_transform",
        stride_1x1=False,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
        drop_connect_rate=0.0,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.
        ResStage builds p streams, where p can be greater or equal to one.
        Args:
            dim_in (list): list of p the channel dimensions of the input.
                Different channel dimensions control the input dimension of
                different pathways.
            dim_out (list): list of p the channel dimensions of the output.
                Different channel dimensions control the input dimension of
                different pathways.
            temp_kernel_sizes (list): list of the p temporal kernel sizes of the
                convolution in the bottleneck. Different temp_kernel_sizes
                control different pathway.
            stride (list): list of the p strides of the bottleneck. Different
                stride control different pathway.
            num_blocks (list): list of p numbers of blocks for each of the
                pathway.
            dim_inner (list): list of the p inner channel dimensions of the
                input. Different channel dimensions control the input dimension
                of different pathways.
            num_groups (list): list of number of p groups for the convolution.
                num_groups=1 is for standard ResNet like networks, and
                num_groups>1 is for ResNeXt like networks.
            num_block_temp_kernel (list): extent the temp_kernel_sizes to
                num_block_temp_kernel blocks, then fill temporal kernel size
                of 1 for the rest of the layers.
            nonlocal_inds (list): If the tuple is empty, no nonlocal layer will
                be added. If the tuple is not empty, add nonlocal layers after
                the index-th block.
            dilation (list): size of dilation for each pathway.
            nonlocal_group (list): list of number of p nonlocal groups. Each
                number controls how to fold temporal dimension to batch
                dimension before applying nonlocal transformation.
                https://github.com/facebookresearch/video-nonlocal-net.
            instantiation (string): different instantiation for nonlocal layer.
                Supports two different instantiation method:
                    "dot_product": normalizing correlation matrix with L2.
                    "softmax": normalizing correlation matrix with Softmax.
            trans_func_name (string): name of the the transformation function apply
                on the network.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            drop_connect_rate (float): basic rate at which blocks are dropped,
                linearly increases from input to output blocks.
        """
        super(ResStage_FFN, self).__init__()
        assert all(
            (
                num_block_temp_kernel[i] <= num_blocks[i]
                for i in range(len(temp_kernel_sizes))
            )
        )
        self.num_blocks = num_blocks
        self.nonlocal_group = nonlocal_group
        self._drop_connect_rate = drop_connect_rate
        self.temp_kernel_sizes = [
            (temp_kernel_sizes[i] * num_blocks[i])[: num_block_temp_kernel[i]]
            + [1] * (num_blocks[i] - num_block_temp_kernel[i])
            for i in range(len(temp_kernel_sizes))
        ]
        assert (
            len(
                {
                    len(dim_in),
                    len(dim_out),
                    len(temp_kernel_sizes),
                    len(stride),
                    len(num_blocks),
                    len(dim_inner),
                    len(num_groups),
                    len(num_block_temp_kernel),
                    len(nonlocal_inds),
                    len(nonlocal_group),
                }
            )
            == 1
        )
        self.num_pathways = len(self.num_blocks)
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            trans_func_name,
            stride_1x1,
            inplace_relu,
            nonlocal_inds,
            nonlocal_pool,
            instantiation,
            dilation,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        trans_func_name,
        stride_1x1,
        inplace_relu,
        nonlocal_inds,
        nonlocal_pool,
        instantiation,
        dilation,
        norm_module,
    ):
        for pathway in range(self.num_pathways):
            for i in range(self.num_blocks[pathway]):
                # Retrieve the transformation function.
                trans_func = get_trans_func(trans_func_name)
                # Construct the block.
                res_block = ResBlock_FFN(
                    dim_in[pathway] if i == 0 else dim_out[pathway],
                    dim_out[pathway],
                    self.temp_kernel_sizes[pathway][i],
                    stride[pathway] if i == 0 else 1,
                    trans_func,
                    dim_inner[pathway],
                    num_groups[pathway],
                    stride_1x1=stride_1x1,
                    inplace_relu=inplace_relu,
                    dilation=dilation[pathway],
                    norm_module=norm_module,
                    block_idx=i,
                    drop_connect_rate=self._drop_connect_rate,
                )
                self.add_module("pathway{}_res{}".format(pathway, i), res_block)
                if i in nonlocal_inds[pathway]:
                    nln = Nonlocal(
                        dim_out[pathway],
                        dim_out[pathway] // 2,
                        nonlocal_pool[pathway],
                        instantiation=instantiation,
                        norm_module=norm_module,
                    )
                    self.add_module(
                        "pathway{}_nonlocal{}".format(pathway, i), nln
                    )

    def forward(self, inputs_4, inputs_8, inputs_16, training=False):
        if training == True:
            output_4 = []
            output_8 = []
            output_16 = []
            for pathway in range(self.num_pathways):
                x_4 = inputs_4[pathway]
                x_8 = inputs_8[pathway]
                x_16 = inputs_16[pathway]
                for i in range(self.num_blocks[pathway]):
                    m = getattr(self, "pathway{}_res{}".format(pathway, i))
                    x_4, x_8, x_16 = m(x_4, x_8, x_16, True)
                    if hasattr(self, "pathway{}_nonlocal{}".format(pathway, i)):
                        nln = getattr(
                            self, "pathway{}_nonlocal{}".format(pathway, i)
                        )
                        b, c, t, h, w = x.shape
                        if self.nonlocal_group[pathway] > 1:
                            # Fold temporal dimension into batch dimension.
                            x = x.permute(0, 2, 1, 3, 4)
                            x = x.reshape(
                                b * self.nonlocal_group[pathway],
                                t // self.nonlocal_group[pathway],
                                c,
                                h,
                                w,
                            )
                            x = x.permute(0, 2, 1, 3, 4)
                        x = nln(x)
                        if self.nonlocal_group[pathway] > 1:
                            # Fold back to temporal dimension.
                            x = x.permute(0, 2, 1, 3, 4)
                            x = x.reshape(b, t, c, h, w)
                            x = x.permute(0, 2, 1, 3, 4)
                output_4.append(x_4)
                output_8.append(x_8)
                output_16.append(x_16)
                
            return output_4, output_8, output_16
        else:
            if inputs_4 is not None:
                output = []
                for pathway in range(self.num_pathways):
                    x = inputs_4[pathway]
                    for i in range(self.num_blocks[pathway]):
                        m = getattr(self, "pathway{}_res{}".format(pathway, i))
                        x = m(x, None, None, False)
                        if hasattr(self, "pathway{}_nonlocal{}".format(pathway, i)):
                            nln = getattr(
                                self, "pathway{}_nonlocal{}".format(pathway, i)
                            )
                            b, c, t, h, w = x.shape
                            if self.nonlocal_group[pathway] > 1:
                                # Fold temporal dimension into batch dimension.
                                x = x.permute(0, 2, 1, 3, 4)
                                x = x.reshape(
                                    b * self.nonlocal_group[pathway],
                                    t // self.nonlocal_group[pathway],
                                    c,
                                    h,
                                    w,
                                )
                                x = x.permute(0, 2, 1, 3, 4)
                            x = nln(x)
                            if self.nonlocal_group[pathway] > 1:
                                # Fold back to temporal dimension.
                                x = x.permute(0, 2, 1, 3, 4)
                                x = x.reshape(b, t, c, h, w)
                                x = x.permute(0, 2, 1, 3, 4)
                    output.append(x)
            elif inputs_8 is not None:
                output = []
                for pathway in range(self.num_pathways):
                    x = inputs_8[pathway]
                    for i in range(self.num_blocks[pathway]):
                        m = getattr(self, "pathway{}_res{}".format(pathway, i))
                        x = m(None, x, None, False)
                        if hasattr(self, "pathway{}_nonlocal{}".format(pathway, i)):
                            nln = getattr(
                                self, "pathway{}_nonlocal{}".format(pathway, i)
                            )
                            b, c, t, h, w = x.shape
                            if self.nonlocal_group[pathway] > 1:
                                # Fold temporal dimension into batch dimension.
                                x = x.permute(0, 2, 1, 3, 4)
                                x = x.reshape(
                                    b * self.nonlocal_group[pathway],
                                    t // self.nonlocal_group[pathway],
                                    c,
                                    h,
                                    w,
                                )
                                x = x.permute(0, 2, 1, 3, 4)
                            x = nln(x)
                            if self.nonlocal_group[pathway] > 1:
                                # Fold back to temporal dimension.
                                x = x.permute(0, 2, 1, 3, 4)
                                x = x.reshape(b, t, c, h, w)
                                x = x.permute(0, 2, 1, 3, 4)
                    output.append(x)
            else:
                output = []
                for pathway in range(self.num_pathways):
                    x = inputs_16[pathway]
                    for i in range(self.num_blocks[pathway]):
                        m = getattr(self, "pathway{}_res{}".format(pathway, i))
                        x = m(None, None, x, False)
                        if hasattr(self, "pathway{}_nonlocal{}".format(pathway, i)):
                            nln = getattr(
                                self, "pathway{}_nonlocal{}".format(pathway, i)
                            )
                            b, c, t, h, w = x.shape
                            if self.nonlocal_group[pathway] > 1:
                                # Fold temporal dimension into batch dimension.
                                x = x.permute(0, 2, 1, 3, 4)
                                x = x.reshape(
                                    b * self.nonlocal_group[pathway],
                                    t // self.nonlocal_group[pathway],
                                    c,
                                    h,
                                    w,
                                )
                                x = x.permute(0, 2, 1, 3, 4)
                            x = nln(x)
                            if self.nonlocal_group[pathway] > 1:
                                # Fold back to temporal dimension.
                                x = x.permute(0, 2, 1, 3, 4)
                                x = x.reshape(b, t, c, h, w)
                                x = x.permute(0, 2, 1, 3, 4)
                    output.append(x)

            return output