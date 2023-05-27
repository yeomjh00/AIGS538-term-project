import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import  Dataset
import random
import cv2 as cv
import numpy as np

class Original(Dataset):
    def __init__(self, dataset, args, edge=False): 
        self.dataset = dataset
        self.num_class = args.num_class
        self.mix_num = args.original_mix_num
        self.beta = args.original_beta
        self.prob = args.original_prob
        self.saliency_ratio = args.original_saliency_ratio
        self.basic_mix_ratio = args.original_basic_mix_ratio
        self.edge = edge
        self.noise = args.original_noise

        if self.edge:
            self.mix_num = 2
            self.prob = 1.0

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        # print(f"lb from index: {lb}")
        tlb = torch.zeros(self.num_class)
        tlb[lb] = 1.

        for _ in range(self.mix_num):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            lamb = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            tlb2 = torch.zeros(self.num_class)
            tlb2[lb2] = 1.

            bbx1, bby1, bbx2, bby2 = self.__saliency_bbox(img, lamb)

            x,  y = bbx2-bbx1, bby2 - bby1
            noise = torch.normal(0, self.noise, (3, x, y)) * 0.1

            res_img = img * self.basic_mix_ratio \
                    + img2 * (1. - self.basic_mix_ratio)
            res_img[:, bbx1:bbx2, bby1:bby2] = res_img[:, bbx1:bbx2, bby1:bby2] \
                                                - img[:, bbx1:bbx2, bby1:bby2] * self.saliency_ratio \
                                                + img2[:, bbx1:bbx2, bby1:bby2] * self.saliency_ratio \
                                                + noise
            img = res_img
            lamb = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            tlb = tlb * (self.basic_mix_ratio - self.saliency_ratio * lamb) \
                    + tlb2 * (1. - self.basic_mix_ratio + self.saliency_ratio * lamb)
            
        return img, tlb

    def __len__(self):
        return len(self.dataset)
    
    def __saliency_bbox(self, img, lam):
            if len(img.size()) == 4:
                W = img.size(2)
                H = img.size(3)
            elif len(img.size()) == 3:
                W = img.size(1)
                H = img.size(2)
            else:
                raise Exception
            
            cut_rat = np.sqrt(1. - lam)
            cut_w = np.int64(W * cut_rat)
            cut_h = np.int64(H * cut_rat)

            temp_img = img.cpu().numpy().transpose(1, 2, 0)
            saliency = cv.saliency.StaticSaliencySpectralResidual_create()
            _, saliencyMap = saliency.computeSaliency(temp_img)
            saliencyMap = (saliencyMap * 255).astype("uint8")

            x, y = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)

            bbx1 = np.clip(x - cut_w // 2, 0, W)
            bby1 = np.clip(y - cut_h // 2, 0, H)
            bbx2 = np.clip(x + cut_w // 2, 0, W)
            bby2 = np.clip(y + cut_h // 2, 0, H)

            return bbx1, bby1, bbx2, bby2