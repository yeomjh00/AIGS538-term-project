from .instahide import InstaHide
from .cutmix import CutMix
from .original import Original
from typing import Callable

def load_augmentation(args, aug_type: str) -> Callable:
    if aug_type is None:
        return None
    elif aug_type == "instahide":
        return InstaHide(args)
    elif aug_type == "cutmix":
        return CutMix(args, args.batch_size, args.batch_size + args.additional_augment)
    elif aug_type == "original":
        return Original(args)
    else:
        raise ValueError("augmentation type is not supported")