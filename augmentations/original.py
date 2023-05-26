import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.distributions.beta import Beta
import math
import random
import numpy as np

class Original(Dataset):
    def __init__(self, dataset, args, edge=False): 
        self.dataset = dataset
        self.num_class = args.num_class
        self.mix_num = args.cutmix_mix_num
        self.beta = args.cutmix_beta
        self.prob = args.cutmix_prob
        self.edge = edge

        if self.edge:
            self.mix_num = 2
            self.prob = 1.0

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb = torch.Tensor([lb])
        onehot = F.one_hot(self.num_class, lb)

        for _ in range(self.mix_num):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            lamb = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = F.one_hot(self.num_class, lb2)

            bbx1, bby1, bbx2, bby2 = self.__rand_bbox(img.size(), lamb)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lamb = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            lb_onehot = lb_onehot * lamb + lb2_onehot * (1. - lamb)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)
    
    def __rand_bbox(size, lam):
        if len(size) == 4:
            W = size[2]
            H = size[3]
        elif len(size) == 3:
            W = size[1]
            H = size[2]
        else:
            raise Exception

        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2