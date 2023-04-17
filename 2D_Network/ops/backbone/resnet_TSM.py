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

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


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

    def __init__(self, inplanes, planes, num_segments, stride=1, downsample=None, remainder=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.tsm = TSM(num_segments)
        self.downsample = downsample
        self.stride = stride
        self.remainder= remainder

    def forward(self, x):
        identity = x  
        out = self.tsm(x)   
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
       
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,num_segments, stride=1, downsample=None, remainder=0):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.tsm = TSM(num_segments)
        self.downsample = downsample
        self.stride = stride
        self.remainder= remainder              

    def forward(self, x):
        identity = x
        out = self.tsm(x) 
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    
class ResNet(nn.Module):

    def __init__(self, block, layers, num_segments, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.inplanes_mask = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)       
        self.num_segments = num_segments 

        self.layer1 = self._make_layer(block, 64, layers[0], num_segments=num_segments)
        self.layer2 = self._make_layer(block, 128, layers[1],  num_segments=num_segments, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2],  num_segments=num_segments, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3],  num_segments=num_segments, stride=2)       
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)                   
        # self.fc = nn.Conv1d(512*block.expansion, num_classes, kernel_size=1, stride=1, padding=0,bias=True)         
        
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

    def _make_layer(self, block, planes, blocks, num_segments, stride=1):       
        downsample = None        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, num_segments, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            remainder =int( i % 3)
            layers.append(block(self.inplanes, planes, num_segments, remainder=remainder))
            
        return nn.Sequential(*layers)            
        
        
    def forward(self, x):  
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)                             
        x = self.layer2(x)          
        x = self.layer3(x)                           
        x = self.layer4(x)
        x = self.avgpool(x)    
        x = x.view(x.size(0), -1)  
                       
        x = self.fc(x)      
        return x


def resnet18(pretrained=False, shift='TSM', num_segments=8, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):  
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_segments=num_segments, **kwargs)  
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        new_state_dict =  model.state_dict()
        for k, v in pretrained_dict.items():
            if (k in new_state_dict):
                new_state_dict.update({k:v})      
#                 print ("%s layer has pretrained weights" % k)
        model.load_state_dict(new_state_dict)
    return model


def resnet34(pretrained=False, shift='TSM',num_segments=8, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):
        model = ResNet(BasicBlock, BasicBlock, [3, 4, 6, 3],num_segments=num_segments, **kwargs)        
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        new_state_dict =  model.state_dict()
        for k, v in pretrained_dict.items():
            if (k in new_state_dict):
                new_state_dict.update({k:v})      
#                 print ("%s layer has pretrained weights" % k)
        model.load_state_dict(new_state_dict)
    return model


def resnet50(pretrained=False, shift='TSM', num_segments=8, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):    
        model = ResNet(Bottleneck, [3, 4, 6, 3],num_segments=num_segments, **kwargs)          
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        new_state_dict =  model.state_dict()
        for k, v in pretrained_dict.items():
            if (k in new_state_dict):
                new_state_dict.update({k:v})
        #         print(k)
        #     else:
        #         print(k,'not initialized!!!!!!!!!!')
        # assert(1==0)
        model.load_state_dict(new_state_dict)
    return model


def resnet101(pretrained=False, shift='TSM', num_segments=8, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):    
        model = ResNet(Bottleneck, [3, 4, 23, 3], num_segments=num_segments, **kwargs)          
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        new_state_dict =  model.state_dict()
        for k, v in pretrained_dict.items():
            if (k in new_state_dict):
                new_state_dict.update({k:v})      
#                 print ("%s layer has pretrained weights" % k)
        model.load_state_dict(new_state_dict)
    return model


def resnet152(pretrained=False, shift='TSM', num_segments=8, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (shift =='TSM'):    
        model = ResNet(Bottleneck, [3, 8, 36, 3],num_segments=num_segments, **kwargs)          
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        new_state_dict =  model.state_dict()
        for k, v in pretrained_dict.items():
            if (k in new_state_dict):
                new_state_dict.update({k:v})      
#                 print ("%s layer has pretrained weights" % k)
        model.load_state_dict(new_state_dict)
    return model