import torch
import torchvision
import torchvision.transforms as transforms

class InstaHide:
    def __init__(self, mix: int, mix_ratio: list):
        self.mix = mix
        self.mix_ratio = mix_ratio
        assert mix == len(mix_ratio), "total number of images to be mixed and length of mix ratio list should be same"
        assert sum(mix_ratio), "sum of mix ratio should be 1"
        
        return
    
    def __call__(self):
        pass