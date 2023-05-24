import argparse

def return_args():
    args = argparse.ArgumentParser(description='')
    
    # Basic
    args.add_argument('--seed', type=int, default=1, help='random seed')
    args.add_argument('--save_path', type=str, default='./', help='save path')
    args.add_argument('--load', type=bool, default=False, help='load model')
    args.add_argument('--output_path', type=str, default='./', help='output path')
    
    # Dataset
    args.add_argument('--dataset', type=str, default='cifar10', help='cifar10, cifar100, svhn')
    args.add_argument('--batch_size', type=int, default=4, help='batch size')
    args.add_argument('--additional_augment', type=int, help="number of produced images from augmentation", default=2)
    args.add_argument('--epochs', type=int, default=200, help='number of epochs')
    
    # Optimizer
    args.add_argument('--lr', type=float, default=0.1, help='learning rate')
    args.add_argument('--momentum', type=float, default=0.9, help='momentum')
    args.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    
    # Federated Learning
    args.add_argument('--functioin', type=str, default=None, help='train/test/attack/None(=all)')

    # Augmentation
    args.add_argument('--aug_type', type=str, default=None, \
                      help="augmentation type: cutmix, cutout, instahide, original, none(do not apply any augmentation)")

    # Augmentation - CutMix
    args.add_argument('--beta', type=float, default=1.0, help='beta for mixup')
    args.add_argument('--cutmix_prob', type=float, default=0.5, help='cutmix probability')
    args.add_argument('--mix', type=int, default=2, help='how many images to mix')

    return args