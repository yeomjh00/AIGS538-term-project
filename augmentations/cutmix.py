import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torchvision
import torchvision.transforms as transforms
import random


"""
https://github.com/ildoonet/cutmix
"""

class CutMix:
    def __init__(self, args, input_num, output_num):
        self.beta = args.beta
        self.cutmix_prob = args.cutmix_prob
        self.mix = args.mix
        self.input_num = input_num
        self.output_num = output_num
    
    def __call__(self, x, y, attack=False):
        assert len(x.size()) == 4, "input tensor must be [Batch] * [channel] * [height] * [width] sized tensor"
        images = x.chunk(x.size(0), dim=0)
        labels = y.chunk(y.size(0), dim=0)

        assert len(images) == len(labels), "images and labels must have same batch size"
        sample_list = [i in range(len(x))]
        aug_images = []
        for i in range(self.output_num - self.input_num):
            s1, s2 = random.sanmple(sample_list, 2)
            lam = np.random.beta(self.beta, self.beta)

