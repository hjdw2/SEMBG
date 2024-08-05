import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import sys, math
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,groups=1, bias=True)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1,  groups=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, groups=groups, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.in_planes = nStages[0]
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class Wide_ResNet_SEMBG(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, g0, g1, g2, g3):
        super(Wide_ResNet_SEMBG, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        g = g0 * g1 * g2 * g3

        self.conv1 = conv3x3(3,int(nStages[0]/g0)*g0)
        self.in_planes = int(nStages[0]/g0)*g0
        self.layer1 = self._wide_layer(wide_basic, int(nStages[1]*math.sqrt(g0)/g)*g, n, dropout_rate, stride=1, groups=g0)

        self.layer2_1 = self._wide_layer(wide_basic, int(nStages[2]*math.sqrt(g1)/g1/math.sqrt(3.0))*g1, n, dropout_rate, stride=2, groups=g1)
        self.layer3_1 = self._wide_layer(wide_basic, int(nStages[3]*math.sqrt(g1)/g1/math.sqrt(3.0))*g1, n, dropout_rate, stride=2, groups=g1)
        self.bn_1 = nn.BatchNorm2d(int(nStages[3]*math.sqrt(g1)/g1/math.sqrt(3.0))*g1, momentum=0.9)
        self.linear_1 = nn.Linear(int(nStages[3]*math.sqrt(g1)/g1/math.sqrt(3.0))*g1, num_classes)

        self.in_planes = int(nStages[1]*math.sqrt(g0)/g)*g
        self.layer2_2 = self._wide_layer(wide_basic, int(nStages[2]*math.sqrt(g2)/g2/math.sqrt(3.0))*g2, n, dropout_rate, stride=2, groups=g2)
        self.layer3_2 = self._wide_layer(wide_basic, int(nStages[3]*math.sqrt(g2)/g2/math.sqrt(3.0))*g2, n, dropout_rate, stride=2, groups=g2)
        self.bn_2 = nn.BatchNorm2d(int(nStages[3]*math.sqrt(g2)/g2/math.sqrt(3.0))*g2, momentum=0.9)
        self.linear_2 = nn.Linear(int(nStages[3]*math.sqrt(g2)/g2/math.sqrt(3.0))*g2, num_classes)

        self.in_planes = int(nStages[1]*math.sqrt(g0)/g)*g
        self.layer2_3 = self._wide_layer(wide_basic, int(nStages[2]*math.sqrt(g3)/g3/math.sqrt(3.0))*g3, n, dropout_rate, stride=2, groups=g3)
        self.layer3_3 = self._wide_layer(wide_basic, int(nStages[3]*math.sqrt(g3)/g3/math.sqrt(3.0))*g3, n, dropout_rate, stride=2, groups=g3)
        self.bn_3 = nn.BatchNorm2d(int(nStages[3]*math.sqrt(g3)/g3/math.sqrt(3.0))*g3, momentum=0.9)
        self.linear_3 = nn.Linear(int(nStages[3]*math.sqrt(g3)/g3/math.sqrt(3.0))*g3, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, groups):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, groups))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)

        out1 = self.layer2_1(out)
        out1 = self.layer3_1(out1)
        out1 = F.relu(self.bn_1(out1))
        out1 = F.avg_pool2d(out1, 8)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.linear_1(out1)

        out2 = self.layer2_2(out)
        out2 = self.layer3_2(out2)
        out2 = F.relu(self.bn_2(out2))
        out2 = F.avg_pool2d(out2, 8)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.linear_2(out2)

        out3 = self.layer2_3(out)
        out3 = self.layer3_3(out3)
        out3 = F.relu(self.bn_3(out3))
        out3 = F.avg_pool2d(out3, 8)
        out3 = out3.view(out3.size(0), -1)
        out3 = self.linear_3(out3)
        return out1, out2, out3
