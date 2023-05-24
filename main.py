import classifier
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

    load_batch = args.batch
    input_batch = args.batch
    if args.aug_type != "none":
        input_batch += args.additional_augment
        augmentation = load_augmentation(args, args.aug_type)

    _transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])



    # labels of CIFAR 10
    string_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    labels_dict = {l:idx for idx, l in enumerate(string_labels)}
    def label_transform(label):
        return labels_dict[label]

    victim = classifier.ResNet()

    # returned data size: x = (batch_size, 3, 32, 32), y : Tensor of long = (batch_size)
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,\
                                             transform=_transform, target_transform=label_transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, \
                                            transform=_transform, target_transform=label_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=load_batch, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=load_batch, shuffle=False, num_workers=2)
    
    
    optimizer = torch.optim.Adam(victim.parameters(), lr=args.lr)

    if args.load:
        victim.load_state_dict(f"{args.save_path}/{args.aug_type}.pkl")
        # TODO: if need, add loading epoch, accuracy, etc for continuing training.

    if args.function == "train" or args.function is None:
        for epoch in range(args.epochs):
            best_loss = float("inf")
            train_loss, train_acc, victim = train(args, victim, train_loader, optimizer, augmentation)
            if best_loss > train_loss:
                best_loss = train_loss
                best_state = victim.state_dict()
        torch.save(best_state, f"{args.save_path}/{args.aug_type}.pkl")
    
    if args.function == "test" or args.function is None:
        test_loss, test_acc = test(args, victim, test_loader, optimizer, augmentation)
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