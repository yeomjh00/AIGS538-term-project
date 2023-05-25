import torch
from torch import nn
from torch.nn import functional as F

import torchvision
import torchvision.transforms as transforms

from augmentations import InstaHide, CutMix

# Scratch conv net which will be replaced into other architectures
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, optimizer, augmentation, criterion, batch_size):
        super(ResNet, self).__init__()
        pass

    def forward(self, x, y):
        pass
    
    def __str__(self):
        return "single client of federated learning system"

    def __repr__(self):
        # TODO: return architecture explanation
        pass

    def train(self):
        pass
    
    def test(self):
        pass

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)            
        )
        self.conv2 = nn.Sequential(
            ResidualBlock(64, 16, 16),
            ResidualBlock(16, 16, 16),
        )
        self.conv3 = nn.Sequential(
            ResidualBlock(16, 32, 32),
            ResidualBlock(32, 32, 32, 2, 1, _downsample=True),
        )
        self.conv4 = nn.Sequential(
            ResidualBlock(32, 64, 64),
            ResidualBlock(64, 64, 64, 2, 1, _downsample=True),
        )
        self.conv5 = nn.Sequential(
            ResidualBlock(64, 128, 128),
            ResidualBlock(128, 128, 128, 2, 1, _downsample=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, _in_channels, _hidden_channels, _out_channels, \
                _stride_1=1, _stride_2=1, \
                _kernel_size_1=3, _kernel_size_2=3, \
                _padding_1=1, _padding_2=1, \
                _downsample=None, _bias=False):
        super(ResidualBlock, self).__init__()

        skip_stride = 1
        if _downsample:
            skip_stride = 2

        self.main_layer = nn.Sequential (
            nn.Conv2d(in_channels=_in_channels, out_channels=_hidden_channels, \
                      kernel_size=_kernel_size_1, \
                      stride=_stride_1, padding=_padding_1, bias=_bias),
            nn.BatchNorm2d(_hidden_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=_hidden_channels, out_channels=_out_channels, \
                      kernel_size=_kernel_size_2, \
                      stride=_stride_2, padding=_padding_2, bias=_bias),
            nn.BatchNorm2d(_out_channels),
        )

        self.skip_layers = nn.Sequential(
            nn.Conv2d(in_channels=_in_channels, out_channels=_out_channels, \
                      kernel_size=1, stride=skip_stride, \
                      padding=0, bias=_bias),
            nn.BatchNorm2d(_out_channels)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        Z = self.main_layer(x)
        Z += self.skip_layers(x)
        Z = self.relu(Z)
        return Z