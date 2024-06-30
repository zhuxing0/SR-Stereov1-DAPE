import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import pdb

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)
    
class UnNet1(nn.Module):
    def __init__(self, in_channel=162):
        super(UnNet1, self).__init__() # python 2, 'super().__init__()' and 'super(xxx, self).__init__()' both are also ok in python 3.

        self.norm_fn = 'batch'

        self.in_planes = in_channel
        self.layer1 = self._make_layer(32, stride=1)

        self.conv3 = nn.Conv2d(self.in_planes, 1, 1, 1)
        self.act = nn.Sigmoid() # nn.Sigmoid() nn.Tanh()

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, 64, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(64, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)
    
    def forward(self, geo_feat): # default, CGEV特征, torch.Size([4, 162, 80, 184]))

        y = self.layer1(geo_feat)
        y = self.conv3(y)
        x = self.act(y)
        return x, y
    
class EENet1(nn.Module):
    def __init__(self, norm_fn = 'batch'):
        super(EENet1, self).__init__() # python 2, 'super().__init__()' and 'super(xxx, self).__init__()' both are also ok in python 3.

        self.norm_fn = norm_fn

        # self.layer1 = ResidualBlock(1, 16, self.norm_fn, stride=1)
        # self.layer2 = ResidualBlock(3, 16, self.norm_fn, stride=1)
        # self.layer3 = ResidualBlock(32, 16, self.norm_fn, stride=1)

        self.layer1 = ResidualBlock(1, 29, self.norm_fn, stride=1)
        self.layer2 = ResidualBlock(32, 16, self.norm_fn, stride=1)

        self.conv3 = nn.Conv2d(16, 1, 1, 1)
        self.act = nn.Sigmoid() 

    def forward(self, disp, image): # default, CGEV特征, torch.Size([4, 162, 80, 184]))
        
        # # 2_ori: (disp: 1-29 + image): 32-16-1
        # disp = self.layer1(disp)
        # image = self.layer2(image)
        # edge = self.layer3(torch.cat([disp, image], 1))
        # edge = self.conv3(edge)
        # edge = self.act(edge)

        disp = self.layer1(disp)

        edge = self.layer2(torch.concat([disp, image], 1))

        edge = self.act(self.conv3(edge))

        return edge, image
