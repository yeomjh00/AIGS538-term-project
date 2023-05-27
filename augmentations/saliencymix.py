import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.distributions.beta import Beta
import math
import random
import cv2

class SaliencyMix(Dataset):
    def __init__(self, dataset, args, edge=False): 
        self.dataset = dataset
        self.num_class = args.num_class
        self.mix_num = args.saliencymix_mix_num
        self.beta = args.saliencymix_beta
        self.prob = args.saliencymix_prob
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
            bbx1, bby1, bbx2, bby2 = self.__saliency_bbox(img2, lamb)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lamb = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            tlb = tlb * lamb + tlb2 * (1. - lamb)

        return img, tlb

    def __len__(self):
        return len(self.dataset)
    
    def __saliency_bbox(self, img, lam):
        size = img.shape
        if len(size) == 4:
            W = size[2]
            H = size[3]
            temp_img = img.cpu().numpy().transpose(2, 3, 0, 1)
        elif len(size) == 3:
            W = size[1]
            H = size[2]
            temp_img = img.cpu().numpy().transpose(1, 2, 0)
        else:
            raise Exception

        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int64(W * cut_rat)
        cut_h = np.int64(H * cut_rat)

        # initialize OpenCV's static fine grained saliency detector and
        # compute the saliency map
        saliency = cv2.StaticSaliencySpectralResidual_create()
        (success, saliencyMap) = saliency.computeSaliency(temp_img)
        saliencyMap = (saliencyMap * 255).astype("uint8")

        maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
        cx = maximum_indices[0]
        cy = maximum_indices[1]

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
