import classifier
import time
import args
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from utils import *
import pickle
from augmentations import load_augmentation

def main(args):
    set_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)
    device = args.device
    cuda = args.device == "cuda"

    load_batch = args.batch_size
    input_batch = args.batch_size

    additional_augment = args.additional_augment if args.aug_type else 0
    input_batch += additional_augment
    augmentation = load_augmentation(args, args.aug_type)

    _transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    # labels of CIFAR 10
    string_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    labels_dict = {l:idx for idx, l in enumerate(string_labels)}
    victim = classifier.ResNet().to(device)

    # returned data size: x = (batch_size, 3, 32, 32), y : Tensor of long = (batch_size)
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,\
                                             transform=_transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, \
                                            transform=_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=load_batch, shuffle=True, num_workers=3)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=load_batch, shuffle=False, num_workers=3)
    
    
    optimizer = torch.optim.Adam(victim.parameters(), lr=args.lr)

    if args.load:
        victim.load_state_dict(f"{args.save_path}/{args.aug_type}.pkl")
        # TODO: if need, add loading epoch, accuracy, etc for continuing training.

    if args.function == "train" or args.function is None:
        for epoch in range(args.epochs):
            start = time.time()
            print("Start epoch: %d/%d" % (epoch+1, args.epochs))
            best_loss = float("inf")
            train_loss, train_acc, victim = train(args, victim, train_loader, optimizer, augmentation, cuda=cuda)
            if best_loss > train_loss['mean']:
                best_loss = train_loss['mean']
                best_state = victim.state_dict()
            print("time: [%s], loss: %.4f, accuracy: %.4f" % (time_since(start), train_loss, train_acc))
        torch.save(best_state, f"{args.save_path}/{args.aug_type}.pkl")
    
    if args.function == "test" or args.function is None:
        test_loss, test_acc = test(args, victim, test_loader, optimizer, augmentation, cuda=cuda)
        # write accuracy

    if args.function == "attack" or args.function is None:
        pass
        # store trained images
        # w, grad = victim.leak_info()
        
        # attacker = attacker(w, grad)
        # img = attacker.gradientinversion()

        # leakage_score = leakage_metric(img, target)
        # accuracy = victim(img_batch)

    # store accuracy, learkage score, image

    # visualization tools

if __name__ == "__main__":
    args = args.return_args()
    main(args)