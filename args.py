import argparse

def return_args():
    args = argparse.ArgumentParser(description='')
    
    # Basic
    args.add_argument('--seed', type=int, default=1, help='random seed')
    args.add_argument('--save_path', type=str, default='./', help='save path')
    
    # Dataset
    args.add_argument('--dataset', type=str, default='cifar10', help='cifar10, cifar100, svhn')
    args.add_argument('--batch_size', type=int, default=128, help='batch size')
    args.add_argument('--epochs', type=int, default=200, help='number of epochs')
    
    # Optimizer
    args.add_argument('--lr', type=float, default=0.1, help='learning rate')
    args.add_argument('--momentum', type=float, default=0.9, help='momentum')
    args.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    
    # Federated Learning
    args.add_argument('--num_clients', type=int, default=10, help='number of clients')

    # Augmentation
    args.add_argument('--aug', type=str, default="none", help="augmentation type: cutmix, cutout, instahide, original, none")

    # Augmentation - CutMix
    args.add_argument('--beta', type=float, default=1.0, help='beta for mixup')
    args.add_argument('--cutmix_prob', type=float, default=0.5, help='cutmix probability')
    args.add_argument('--mix', type=int, default=2, help='how many images to mix')

    return args