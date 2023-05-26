import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.distributions.beta import Beta
import math
import random


"""
https://github.com/ildoonet/cutmix
"""

class CutMix:
    def __init__(self, args, input_batch, auged_batch, w=32, h=32):
        self.alpha = args.alpha
        self.beta = args.beta
        self.mix = args.mix
        self.device = args.device
        self.input_batch = input_batch
        self.auged_batch = auged_batch
        self.w = w
        self.h = h
    
    def __call__(self, x, y, attack=False):
        assert len(x.size()) == 4, "input tensor must be [Batch] * [channel] * [height] * [width] sized tensor"
        assert x.size()[0] == self.input_batch, "input tensor must be fixed sized tensor"
        images = x.chunk(x.size(0), dim=0)
        labels = y.chunk(y.size(0), dim=0)
        aug_size = (self.auged_batch - self.input_batch)
        
        h_edges = torch.randint(10, self.h, (aug_size,)).to(self.device)
        w_edges = torch.randint(10, self.w, (aug_size,)).to(self.device)

        top = torch.randint(0, self.h - 10, (aug_size,)).to(self.device)
        left = torch.randint(0, self.w - 10, (aug_size,)).to(self.device)

        end_h = torch.minimum(top + h_edges, torch.ones_like(top) * self.h)
        end_w = torch.minimum(left + w_edges, torch.ones_like(left) * self.w)
        
        actual_h_edge = end_h - top
        actual_w_edge = end_w - left

        lam = (actual_w_edge * actual_h_edge) / (self.w * self.h)
        
        assert len(images) == len(labels), "images and labels must have same batch size"
        perm_length = (aug_size // self.input_batch + 1) * self.input_batch
        perm = torch.concat([torch.randperm(self.input_batch) for i in range(perm_length)], dim=0)

        cropped_images = []
        cropped_labels = []
        for i in range(aug_size):
            img = images[perm[i]].squeeze().clone()
            next_img = images[perm[i+1]].squeeze().clone()
            crop =  next_img[:, top[i]:end_h[i], left[i]:end_w[i]].clone().detach()
            img[:, top[i]:end_h[i], left[i]:end_w[i]] = crop[:,:,:]
            
            cropped_images.append(img.unsqueeze(0))
            cropped_labels.append(labels[perm[i]] * lam[i] + labels[perm[i + 1]] * (1 - lam[i]))
        
        aug_images = torch.cat(cropped_images, dim=0)
        aug_labels = torch.cat(cropped_labels, dim=0)
        
        return torch.cat((x.clone(), aug_images), dim=0), torch.cat((y.clone(), aug_labels), dim=0)