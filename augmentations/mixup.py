import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.distributions.beta import Beta
import math
import random

class MixUp(Dataset):
    def __init__(self, dataset, args, edge=False): 
        self.dataset = dataset
        self.num_class = args.num_class
        self.mix_num = args.mixup_mix_num
        self.beta = args.mixup_beta
        self.prob = args.mixup_prob
        self.edge = edge

        if self.edge:
            self.mix_num = 2
            self.prob = 1.0

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        tlb = torch.zeros(self.num_class)
        tlb[lb] = 1

        for _ in range(self.mix_num):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            lamb = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            
            tlb2 = torch.zeros(self.num_class)
            tlb2[lb2] = 1
            
            img = img * lamb + img2 * (1. - lamb)
            tlb = tlb * lamb + tlb2 * (1. - lamb)

        return img, tlb

    def __len__(self):
        return len(self.dataset)