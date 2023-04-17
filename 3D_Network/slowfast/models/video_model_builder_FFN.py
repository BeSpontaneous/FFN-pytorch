# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


"""Video models."""

import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

import slowfast.utils.logging as logging
import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.attention import MultiScaleBlock
from slowfast.models.batchnorm_helper import get_norm
from slowfast.models.common import TwoStreamFusion
from slowfast.models.reversible_mvit import ReversibleMViT
from slowfast.models.utils import (
    calc_mvit_feature_geometry,
    get_3d_sincos_pos_embed,
    round_width,
    validate_checkpoint_wrapper_import,
)

from . import head_helper, operators, resnet_helper_FFN, stem_helper_FFN  # noqa
from .build import MODEL_REGISTRY

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None


logger = logging.get_logger(__name__)

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {18: (2, 2, 2, 2), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "slow_c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow_i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast_FFN": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}

_POOL1 = {
    "2d": [[1, 1, 1]],
    "c2d": [[2, 1, 1]],
    "slow_c2d": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "slow_i3d": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast_FFN": [[1, 1, 1], [1, 1, 1]],
    "x3d": [[1, 1, 1]],
}


class FuseFastToSlow_FFN(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow_FFN, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn_4 = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.bn_8 = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.bn_16 = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x_4, x_8, x_16, training=False):
        if training == True:
            x_s_4 = x_4[0]
            x_f_4 = x_4[1]
            fuse_4 = self.conv_f2s(x_f_4)
            fuse_4 = self.bn_4(fuse_4)
            fuse_4 = self.relu(fuse_4)
            x_s_fuse_4 = torch.cat([x_s_4, fuse_4], 1)
            
            x_s_8 = x_8[0]
            x_f_8 = x_8[1]
            fuse_8 = self.conv_f2s(x_f_8)
            fuse_8 = self.bn_8(fuse_8)
            fuse_8 = self.relu(fuse_8)
            x_s_fuse_8 = torch.cat([x_s_8, fuse_8], 1)
            
            x_s_16 = x_16[0]
            x_f_16 = x_16[1]
            fuse_16 = self.conv_f2s(x_f_16)
            fuse_16 = self.bn_16(fuse_16)
            fuse_16 = self.relu(fuse_16)
            x_s_fuse_16 = torch.cat([x_s_16, fuse_16], 1)
            
            return [x_s_fuse_4, x_f_4], [x_s_fuse_8, x_f_8], [x_s_fuse_16, x_f_16]
        else:
            if x_4 is not None:
                x_s = x_4[0]
                x_f = x_4[1]
                fuse = self.conv_f2s(x_f)
                fuse = self.bn_4(fuse)
                fuse = self.relu(fuse)
                x_s_fuse = torch.cat([x_s, fuse], 1)
            elif x_8 is not None:
                x_s = x_8[0]
                x_f = x_8[1]
                fuse = self.conv_f2s(x_f)
                fuse = self.bn_8(fuse)
                fuse = self.relu(fuse)
                x_s_fuse = torch.cat([x_s, fuse], 1)
            else:
                x_s = x_16[0]
                x_f = x_16[1]
                fuse = self.conv_f2s(x_f)
                fuse = self.bn_16(fuse)
                fuse = self.relu(fuse)
                x_s_fuse = torch.cat([x_s, fuse], 1)
            return [x_s_fuse, x_f]


@MODEL_REGISTRY.register()
class SlowFast_FFN(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast_FFN, self).__init__()
        self.norm_module = get_norm(cfg)
        self.cfg = cfg
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2
        self._construct_network(cfg)
        init_helper.init_weights(
            self,
            cfg.MODEL.FC_INIT_STD,
            cfg.RESNET.ZERO_INIT_FINAL_BN,
            cfg.RESNET.ZERO_INIT_FINAL_CONV,
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper_FFN.VideoModelStem_FFN(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module,
        )
        self.s1_fuse = FuseFastToSlow_FFN(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper_FFN.ResStage_FFN(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = FuseFastToSlow_FFN(
            width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper_FFN.ResStage_FFN(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastToSlow_FFN(
            width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper_FFN.ResStage_FFN(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastToSlow_FFN(
            width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper_FFN.ResStage_FFN(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if cfg.DETECTION.ENABLE:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        1,
                        1,
                    ],
                    [cfg.DATA.NUM_FRAMES // pool_size[1][0], 1, 1],
                ],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2,
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
                detach_final_fc=cfg.MODEL.DETACH_FINAL_FC,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None],
                # if cfg.MULTIGRID.SHORT_CYCLE
                # or cfg.MODEL.MODEL_NAME == "ContrastiveModel"
                # else [
                #     [
                #         cfg.DATA.NUM_FRAMES
                #         // cfg.SLOWFAST.ALPHA
                #         // pool_size[0][0],
                #         cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][1],
                #         cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[0][2],
                #     ],
                #     [
                #         cfg.DATA.NUM_FRAMES // pool_size[1][0],
                #         cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[1][1],
                #         cfg.DATA.TRAIN_CROP_SIZE // 32 // pool_size[1][2],
                #     ],
                # ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                detach_final_fc=cfg.MODEL.DETACH_FINAL_FC,
                cfg=cfg,
            )

    def forward(self, x_4, x_8, x_16, training=False, bboxes=None):
        if training == True:
            x_4 = x_4[:]  # avoid pass by reference
            x_8 = x_8[:]  # avoid pass by reference
            x_16 = x_16[:]  # avoid pass by reference
            x_4, x_8, x_16 = self.s1(x_4, x_8, x_16, True)
            x_4, x_8, x_16 = self.s1_fuse(x_4, x_8, x_16, True)
            x_4, x_8, x_16 = self.s2(x_4, x_8, x_16, True)
            x_4, x_8, x_16 = self.s2_fuse(x_4, x_8, x_16, True)
            for pathway in range(self.num_pathways):
                pool = getattr(self, "pathway{}_pool".format(pathway))
                x_4[pathway] = pool(x_4[pathway])
                x_8[pathway] = pool(x_8[pathway])
                x_16[pathway] = pool(x_16[pathway])
            x_4, x_8, x_16 = self.s3(x_4, x_8, x_16, True)
            x_4, x_8, x_16 = self.s3_fuse(x_4, x_8, x_16, True)
            x_4, x_8, x_16 = self.s4(x_4, x_8, x_16, True)
            x_4, x_8, x_16 = self.s4_fuse(x_4, x_8, x_16, True)
            x_4, x_8, x_16 = self.s5(x_4, x_8, x_16, True)
            if self.enable_detection:
                x_4 = self.head(x_4, bboxes)
            else:
                x_4 = self.head(x_4)
                x_8 = self.head(x_8)
                x_16 = self.head(x_16)
            
            return x_4, x_8, x_16
        else:
            if x_4 is not None:
                x = x_4[:]  # avoid pass by reference
                x = self.s1(x, None, None, False)
                x = self.s1_fuse(x, None, None, False)
                x = self.s2(x, None, None, False)
                x = self.s2_fuse(x, None, None, False)
                for pathway in range(self.num_pathways):
                    pool = getattr(self, "pathway{}_pool".format(pathway))
                    x[pathway] = pool(x[pathway])
                x = self.s3(x, None, None, False)
                x = self.s3_fuse(x, None, None, False)
                x = self.s4(x, None, None, False)
                x = self.s4_fuse(x, None, None, False)
                x = self.s5(x, None, None, False)
                if self.enable_detection:
                    x = self.head(x, bboxes)
                else:
                    x = self.head(x)
            elif x_8 is not None:
                x = x_8[:]  # avoid pass by reference
                x = self.s1(None, x, None, False)
                x = self.s1_fuse(None, x, None, False)
                x = self.s2(None, x, None, False)
                x = self.s2_fuse(None, x, None, False)
                for pathway in range(self.num_pathways):
                    pool = getattr(self, "pathway{}_pool".format(pathway))
                    x[pathway] = pool(x[pathway])
                x = self.s3(None, x, None, False)
                x = self.s3_fuse(None, x, None, False)
                x = self.s4(None, x, None, False)
                x = self.s4_fuse(None, x, None, False)
                x = self.s5(None, x, None, False)
                if self.enable_detection:
                    x = self.head(x, bboxes)
                else:
                    x = self.head(x)
            else:
                x = x_16[:]  # avoid pass by reference
                x = self.s1(None, None, x, False)
                x = self.s1_fuse(None, None, x, False)
                x = self.s2(None, None, x, False)
                x = self.s2_fuse(None, None, x, False)
                for pathway in range(self.num_pathways):
                    pool = getattr(self, "pathway{}_pool".format(pathway))
                    x[pathway] = pool(x[pathway])
                x = self.s3(None, None, x, False)
                x = self.s3_fuse(None, None, x, False)
                x = self.s4(None, None, x, False)
                x = self.s4_fuse(None, None, x, False)
                x = self.s5(None, None, x, False)
                if self.enable_detection:
                    x = self.head(x, bboxes)
                else:
                    x = self.head(x)
            
            return x