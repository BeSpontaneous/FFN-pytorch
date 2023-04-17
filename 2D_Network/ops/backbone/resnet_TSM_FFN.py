"""
An example combining `Temporal Shift Module` with `ResNet`. This implementation
is based on `Temporal Segment Networks`, which merges temporal dimension into
batch, i.e. inputs [N*T, C, H, W]. Here we show the case with residual connections
and zero padding with 8 frames as input.
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
from .temporal_shift import TemporalShift
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet50', 'resnet101']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)



class TSM(nn.Module):
    def __init__(self, n_segment=3):
        super(TSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = 8

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div)
        return x

    @staticmethod
    def shift(x, n_segment, fold_div=3):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)

       
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, num_segments_H, stride=1, downsample_conv=None, downsample_bn_4=None, downsample_bn_8=None, downsample_bn_16=None, remainder=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1_4 = nn.BatchNorm2d(planes)
        self.bn1_8 = nn.BatchNorm2d(planes)
        self.bn1_16 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2_4 = nn.BatchNorm2d(planes)
        self.bn2_8 = nn.BatchNorm2d(planes)
        self.bn2_16 = nn.BatchNorm2d(planes)
        self.tsm = TSM(num_segments_H)
        self.downsample_conv = downsample_conv
        self.downsample_bn_4 = downsample_bn_4
        self.downsample_bn_8 = downsample_bn_8
        self.downsample_bn_16 = downsample_bn_16
        self.stride = stride
        self.remainder= remainder

        self.adaconv_4 = nn.Conv2d(inplanes, inplanes, kernel_size=1, groups=inplanes, bias=False)
        self.adaconv_8 = nn.Conv2d(inplanes, inplanes, kernel_size=1, groups=inplanes, bias=False)
        self.adaconv_16 = nn.Conv2d(inplanes, inplanes, kernel_size=1, groups=inplanes, bias=False)
        self.adaconv_4.weight.data.normal_(0,1e-3)
        self.adaconv_8.weight.data.normal_(0,1e-3)
        self.adaconv_16.weight.data.normal_(0,1e-3)  

    def forward(self, x_4, x_8, x, training=True):
        if training == True:
            identity_4 = x_4
            identity_8 = x_8
            identity_16 = x

            out_4 = self.tsm(x_4)
            iden_4 = out_4
            out_4 = self.adaconv_4(out_4)
            out_4 = out_4 + iden_4
            out_4 = self.conv1(out_4)
            out_4 = self.bn1_4(out_4)
            out_4 = self.relu(out_4)
            out_4 = self.conv2(out_4)
            out_4 = self.bn2_4(out_4)


            out_8 = self.tsm(x_8)
            iden_8 = out_8
            out_8 = self.adaconv_8(out_8)
            out_8 = out_8 + iden_8
            out_8 = self.conv1(out_8)
            out_8 = self.bn1_8(out_8)
            out_8 = self.relu(out_8)
            out_8 = self.conv2(out_8)
            out_8 = self.bn2_8(out_8)

            out_16 = self.tsm(x)
            iden_16 = out_16
            out_16 = self.adaconv_16(out_16)
            out_16 = out_16 + iden_16
            out_16 = self.conv1(out_16)
            out_16 = self.bn1_16(out_16)
            out_16 = self.relu(out_16)
            out_16 = self.conv2(out_16)
            out_16 = self.bn2_16(out_16)

            if self.downsample_conv is not None:
                identity_4 = self.downsample_bn_4(self.downsample_conv(x_4))
                identity_8 = self.downsample_bn_8(self.downsample_conv(x_8))
                identity_16 = self.downsample_bn_16(self.downsample_conv(x))
            
            out_4 += identity_4
            out_4 = self.relu(out_4)
            out_8 += identity_8
            out_8 = self.relu(out_8)
            out_16 += identity_16
            out_16 = self.relu(out_16)
            
            return out_4, out_8, out_16
        else:
            if x_4 is not None:
                identity = x_4
                out = self.tsm(x_4)

                iden_4 = out
                out = self.adaconv_4(out)
                out = out + iden_4

                out = self.conv1(out)
                out = self.bn1_4(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2_4(out)
                if self.downsample_conv is not None:
                    identity = self.downsample_bn_4(self.downsample_conv(x_4))
            elif x_8 is not None:
                identity = x_8
                out = self.tsm(x_8)

                iden_8 = out
                out = self.adaconv_8(out)
                out = out + iden_8

                out = self.conv1(out)
                out = self.bn1_8(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2_8(out)
                if self.downsample_conv is not None:
                    identity = self.downsample_bn_8(self.downsample_conv(x_8))
            else:
                identity = x
                out = self.tsm(x)

                iden_16 = out
                out = self.adaconv_16(out)
                out = out + iden_16
                
                out = self.conv1(out)
                out = self.bn1_16(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2_16(out)
                if self.downsample_conv is not None:
                    identity = self.downsample_bn_16(self.downsample_conv(x))
            
            out += identity
            out = self.relu(out)
            
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, num_segments_H, stride=1, downsample_conv=None, downsample_bn_4=None, downsample_bn_8=None, downsample_bn_16=None, remainder=0):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1_4 = nn.BatchNorm2d(planes)
        self.bn1_8 = nn.BatchNorm2d(planes)
        self.bn1_16 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2_4 = nn.BatchNorm2d(planes)
        self.bn2_8 = nn.BatchNorm2d(planes)
        self.bn2_16 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3_4 = nn.BatchNorm2d(planes * self.expansion)
        self.bn3_8 = nn.BatchNorm2d(planes * self.expansion)
        self.bn3_16 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.tsm = TSM(num_segments_H)
        self.downsample_conv = downsample_conv
        self.downsample_bn_4 = downsample_bn_4
        self.downsample_bn_8 = downsample_bn_8
        self.downsample_bn_16 = downsample_bn_16
        self.stride = stride
        self.remainder= remainder

        self.adaconv_4 = nn.Conv2d(inplanes, inplanes, kernel_size=1, groups=inplanes, bias=False)
        self.adaconv_8 = nn.Conv2d(inplanes, inplanes, kernel_size=1, groups=inplanes, bias=False)
        self.adaconv_16 = nn.Conv2d(inplanes, inplanes, kernel_size=1, groups=inplanes, bias=False)
        self.adaconv_4.weight.data.normal_(0,1e-3)
        self.adaconv_8.weight.data.normal_(0,1e-3)
        self.adaconv_16.weight.data.normal_(0,1e-3)                 

    def forward(self, x_4, x_8, x, training=True):
        if training == True:
            identity_4 = x_4
            identity_8 = x_8
            identity_16 = x

            out_4 = self.tsm(x_4)

            iden_4 = out_4
            out_4 = self.adaconv_4(out_4)
            out_4 = out_4 + iden_4

            out_4 = self.conv1(out_4)
            out_4 = self.bn1_4(out_4)
            out_4 = self.relu(out_4)
            out_4 = self.conv2(out_4)
            out_4 = self.bn2_4(out_4)
            out_4 = self.relu(out_4)
            out_4 = self.conv3(out_4)
            out_4 = self.bn3_4(out_4)

            out_8 = self.tsm(x_8)

            iden_8 = out_8
            out_8 = self.adaconv_8(out_8)
            out_8 = out_8 + iden_8

            out_8 = self.conv1(out_8)
            out_8 = self.bn1_8(out_8)
            out_8 = self.relu(out_8)
            out_8 = self.conv2(out_8)
            out_8 = self.bn2_8(out_8)
            out_8 = self.relu(out_8)
            out_8 = self.conv3(out_8)
            out_8 = self.bn3_8(out_8)

            out_16 = self.tsm(x)

            iden_16 = out_16
            out_16 = self.adaconv_16(out_16)
            out_16 = out_16 + iden_16

            out_16 = self.conv1(out_16)
            out_16 = self.bn1_16(out_16)
            out_16 = self.relu(out_16)
            out_16 = self.conv2(out_16)
            out_16 = self.bn2_16(out_16)
            out_16 = self.relu(out_16)
            out_16 = self.conv3(out_16)
            out_16 = self.bn3_16(out_16)

            if self.downsample_conv is not None:
                identity_4 = self.downsample_bn_4(self.downsample_conv(x_4))
                identity_8 = self.downsample_bn_8(self.downsample_conv(x_8))
                identity_16 = self.downsample_bn_16(self.downsample_conv(x))

            out_4 += identity_4
            out_4 = self.relu(out_4)
            out_8 += identity_8
            out_8 = self.relu(out_8)
            out_16 += identity_16
            out_16 = self.relu(out_16)
            
            return out_4, out_8, out_16
        else:
            if x_4 is not None:
                identity = x_4
                out = self.tsm(x_4)

                iden_4 = out
                out = self.adaconv_4(out)
                out = out + iden_4

                out = self.conv1(out)
                out = self.bn1_4(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2_4(out)
                out = self.relu(out)
                out = self.conv3(out)
                out = self.bn3_4(out)
                if self.downsample_conv is not None:
                    identity = self.downsample_bn_4(self.downsample_conv(x_4))
            elif x_8 is not None:
                identity = x_8
                out = self.tsm(x_8)

                iden_8 = out
                out = self.adaconv_8(out)
                out = out + iden_8

                out = self.conv1(out)
                out = self.bn1_8(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2_8(out)
                out = self.relu(out)
                out = self.conv3(out)
                out = self.bn3_8(out)
                if self.downsample_conv is not None:
                    identity = self.downsample_bn_8(self.downsample_conv(x_8))
            else:
                identity = x
                out = self.tsm(x)

                iden_16 = out
                out = self.adaconv_16(out)
                out = out + iden_16
                
                out = self.conv1(out)
                out = self.bn1_16(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2_16(out)
                out = self.relu(out)
                out = self.conv3(out)
                out = self.bn3_16(out)
                if self.downsample_conv is not None:
                    identity = self.downsample_bn_16(self.downsample_conv(x))
            out += identity
            out = self.relu(out)
            
            return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_segments_H, num_segments_M, num_segments_L, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_4 = nn.BatchNorm2d(64)
        self.bn1_8 = nn.BatchNorm2d(64)
        self.bn1_16 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)       
        self.num_segments_H = num_segments_H 

        self.layer1 = self._make_layer(block, 64, layers[0], num_segments_H=num_segments_H)
        self.layer2 = self._make_layer(block, 128, layers[1], num_segments_H=num_segments_H, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], num_segments_H=num_segments_H, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], num_segments_H=num_segments_H, stride=2)       
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():       
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, num_segments_H, stride=1):       
        downsample_conv = None    
        downsample_bn_4 = None
        downsample_bn_8 = None
        downsample_bn_16 = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample_conv = conv1x1(self.inplanes, planes * block.expansion, stride)
            downsample_bn_4 = nn.BatchNorm2d(planes * block.expansion)
            downsample_bn_8 = nn.BatchNorm2d(planes * block.expansion)
            downsample_bn_16 = nn.BatchNorm2d(planes * block.expansion)
        layers = []
        layers.append(block(self.inplanes, planes, num_segments_H, stride, downsample_conv, downsample_bn_4, downsample_bn_8, downsample_bn_16))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            remainder =int( i % 3)
            layers.append(block(self.inplanes, planes, num_segments_H, remainder=remainder))
            
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


def resnet18(pretrained=False, shift='TSM', num_segments_H=16, num_segments_M=8, num_segments_L=4, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):  
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_segments_H=num_segments_H, num_segments_M=num_segments_M, num_segments_L=num_segments_L, **kwargs)  
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
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


def resnet50(pretrained=False, shift='TSM', num_segments_H=16, num_segments_M=8, num_segments_L=4, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):    
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_segments_H=num_segments_H, num_segments_M=num_segments_M, num_segments_L=num_segments_L, **kwargs)          
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
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


def resnet101(pretrained=False, shift='TSM', num_segments_H=16, num_segments_M=8, num_segments_L=4, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):    
        model = ResNet(Bottleneck, [3, 4, 23, 3], num_segments_H=num_segments_H, num_segments_M=num_segments_M, num_segments_L=num_segments_L, **kwargs)          
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
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