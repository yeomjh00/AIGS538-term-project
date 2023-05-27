import classifier
import time
import args
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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

    train_batch = args.batch_size
    edge_batch = args.edge_batch

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
    
    train_set = load_augmentation(train_set, args, edge=False)
    test_set = load_augmentation(test_set, args, edge=False)
    edge_set = load_augmentation(Subset(test_set, range(400)), args, edge=True)

    train_loader = DataLoader(train_set, batch_size=train_batch, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=train_batch, shuffle=False, num_workers=1)
    edge_loader = DataLoader(edge_set, batch_size=edge_batch, shuffle=False, num_workers=1)
    
    
    optimizer = torch.optim.Adam(victim.parameters(), lr=args.lr)

    if args.function == "test":
        victim.load_state_dict(torch.load(f"{args.save_path}/{str(args.aug_type)}.pkl"))

    if args.function == "train" or args.function is None:
        for epoch in range(args.epochs):
            start = time.time()
            print("Start epoch: %d/%d" % (epoch+1, args.epochs))
            train_loss, train_acc, victim = train(args, victim, train_loader, optimizer, cuda=cuda)
            print("time: [%s], loss: %.4f, accuracy: %.4f" % (time_since(start), train_loss["mean"], train_acc["mean"]))
    
    if args.function == "test" or args.function is None:
        test_loss, test_acc = test(args, victim, test_loader, cuda=cuda)
        torch.save(victim.state_dict(), f"{args.save_path}/{str(args.aug_type)}.pkl")
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
