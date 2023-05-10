import torch
from torch import nn
from torch.nn import functional as F

import torchvision
import torchvision.transforms as transforms

import cutmix
import instahide

# Scratch conv net which will be replaced into other architectures
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, optimizer, augmentation, criterion, batch_size):
        super(Classifier, self).__init__()
        # TODO: change architecture by following attack paper.
        self.conv1 = nn.Conv2d(input_size, hidden_size)
        self.conv2 = nn.Conv2d(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.augmentation = augmentation


    def forward(self, x):
        x = self.dropout(F.relu(self.conv1(x)))
        x = self.conv2(x)
        return x
    
    def __str__(self):
        return "single client of federated learning system"

    def __repr__(self):
        # TODO: return architecture explanation
        pass

    def train(self):
        pass

    def test(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

