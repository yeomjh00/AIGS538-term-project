from .instahide import InstaHide
from .saliencymix import SaliencyMix
from .cutmix import CutMix
from .original import Original
from torch.utils.data.dataset import Dataset
from typing import Callable
    
def load_augmentation(dataset, args, edge=False) -> Dataset:
    aug_type = args.aug_type
    if aug_type is None:
        return dataset
    elif aug_type == "instahide":
        return InstaHide(dataset, args)
    elif aug_type == "saliencymix":
        return SaliencyMix(dataset, args)
    elif aug_type == "cutmix":
        return CutMix(dataset, args)
    elif aug_type == "original":
        return Original(dataset, args)
    else:
        raise ValueError("augmentation type is not supported")