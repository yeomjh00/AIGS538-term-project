import argparse

def return_args():
    arg_parse = argparse.ArgumentParser(description='')
    
    # Basic
    arg_parse.add_argument('--seed', type=int, default=42, help='random seed')
    arg_parse.add_argument('--save_path', type=str, default='./models', help='save path')
    arg_parse.add_argument('--load', type=bool, default=False, help='load model')
    arg_parse.add_argument('--output_path', type=str, default='./', help='output path')
    arg_parse.add_argument('--device', type=str, default='cuda', help='cuda/cpu')
    
    # Dataset
    arg_parse.add_argument('--dataset', type=str, default='cifar10', help='cifar10, cifar100, svhn')
    arg_parse.add_argument('--batch_size', type=int, default=4, help='batch size')
    arg_parse.add_argument('--additional_augment', type=int, help="number of produced images from augmentation", default=2)
    arg_parse.add_argument('--epochs', type=int, default=200, help='number of epochs')
    
    # Optimizer
    arg_parse.add_argument('--lr', type=float, default=0.1, help='learning rate')
    
    # Federated Learning
    arg_parse.add_argument('--function', type=str, default=None, help='train/test/attack/None(=all)')

    # Augmentation
    arg_parse.add_argument('--aug_type', type=str, default=None, \
                      help="augmentation type: cutmix, cutout, instahide, original, none(do not apply any augmentation)")

    # Augmentation - CutMix
    arg_parse.add_argument('--beta', type=float, default=1.0, help='beta for mixup')
    arg_parse.add_argument('--cutmix_prob', type=float, default=0.5, help='cutmix probability')
    arg_parse.add_argument('--mix', type=int, default=2, help='how many images to mix')

    # Attack
    arg_parse.add_argument('--target_id', type=int, default=None, help='Cifar validation image used for reconstruction.')

    # Results
    arg_parse.add_argument('--name', default='iv', type=str, help='Name tag for the result table and model.')
    arg_parse.add_argument('--save_image', action='store_true', help='Save the output to a file.')
    arg_parse.add_argument('--image_path', default='images/', type=str)
    arg_parse.add_argument('--table_path', default='tables/', type=str)

    args = arg_parse.parse_args()

    return args