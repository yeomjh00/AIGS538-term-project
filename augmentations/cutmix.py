import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
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
    
    def __call__(self, x, y):
        pass
