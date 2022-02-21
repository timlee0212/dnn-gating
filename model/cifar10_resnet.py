'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.pg_utils as q

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = q.PGConv2d(in_planes, planes, kernel_size=3, 
                                stride=stride, padding=1, bias=False, 
                                wbits=kwargs['wbits'], abits=kwargs['abits'],
                                pgabits=kwargs['pgabits'],
                                sparse_bp=kwargs['sparse_bp'],
                                threshold=kwargs['th'])
        self.conv2 = q.PGConv2d(planes, planes, kernel_size=3, 
                                stride=1, padding=1, bias=False, 
                                wbits=kwargs['wbits'], abits=kwargs['abits'],
                                pgabits=kwargs['pgabits'],
                                sparse_bp=kwargs['sparse_bp'],
                                threshold=kwargs['th'])
        self.relu = q.PactReLU() if kwargs['pact'] else nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = q.PGConv2d(in_planes, self.expansion * planes, 
                                   kernel_size=1, stride=stride, bias=False, 
                                   wbits=kwargs['wbits'], abits=kwargs['abits'],
                                   pgabits=kwargs['pgabits'],
                                   sparse_bp=kwargs['sparse_bp'],
                                   threshold=kwargs['th'])

    def forward(self, x):
        "x is float value"
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, **kwargs):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = q.PGConv2d(3, 16, 
                               kernel_size=3, stride=1, bias=False, 
                               wbits=kwargs['wbits'], abits=kwargs['abits'],
                               pgabits=kwargs['pgabits'],
                               sparse_bp=kwargs['sparse_bp'],
                               threshold = 1.0)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, **kwargs)
        self.linear = nn.Linear(64, num_classes)
        self.relu = q.PactReLU() if kwargs['pact'] else nn.ReLU()

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, **kwargs):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, **kwargs))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, **kwargs)

def resnet32(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [3, 6, 6], num_classes=num_classes, **kwargs)


