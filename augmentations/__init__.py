from .saliencymix import SaliencyMix
from .cutmix import CutMix
from .mixup import MixUp
from .original import Original
from torch.utils.data.dataset import Dataset
    
def load_augmentation(dataset, args, edge=False) -> Dataset:
    aug_type = args.aug_type
    if aug_type is None:
        return dataset
    elif aug_type == "saliencymix":
        return SaliencyMix(dataset, args)
    elif aug_type == "cutmix":
        return CutMix(dataset, args)
    elif aug_type == "original":
        return Original(dataset, args)
    elif aug_type == "mixup":
        return MixUp(dataset, args)
    else:
        raise ValueError("augmentation type is not supported")