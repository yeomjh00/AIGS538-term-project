import argparse

def return_args():
    arg_parse = argparse.ArgumentParser(description='')
    
    # Basic
    arg_parse.add_argument('--seed', type=int, default=42, help='random seed')
    arg_parse.add_argument('--save_path', type=str, default='./models', help='save path')
    arg_parse.add_argument('--output_path', type=str, default='./outputs', help='output path')
    arg_parse.add_argument('--device', type=str, default='cuda', help='cuda/cpu')
    
    # Dataset
    arg_parse.add_argument('--dataset', type=str, default='cifar10', help='cifar10, cifar100, svhn')
    arg_parse.add_argument('--num_class', type=int, default=10, help='number of classes in dataset')
    arg_parse.add_argument('--batch_size', type=int, default=128, help='batch size')
    arg_parse.add_argument('--edge_batch', type=int, default=4, help='batch size for edge device')
    arg_parse.add_argument('--epochs', type=int, default=200, help='number of epochs')
    
    # Optimizer
    arg_parse.add_argument('--lr', type=float, default=0.1, help='learning rate')
    
    # Federated Learning
    arg_parse.add_argument('--function', type=str, default=None, help='train/test/attack/None(=all)')

    # Augmentation
    arg_parse.add_argument('--aug_type', type=str, default=None, \
                      help="augmentation type: cutmix, cutout, instahide, original, none(do not apply any augmentation)")

    # Augmentation - CutMix
    arg_parse.add_argument('--cutmix_mix_num', type=int, default=2, help='number of images to mix')
    arg_parse.add_argument('--cutmix_beta', type=float, default=1.0, help='beta for mixup')
    arg_parse.add_argument('--cutmix_prob', type=float, default=0.5, help='cutmix probability')

    args = arg_parse.parse_args()

    return args