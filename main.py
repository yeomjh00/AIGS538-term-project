import classifier
import time
import args
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
from utils import *
import pickle
from augmentations import load_augmentation

def main(args):
    set_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)
    tensorboard_path = f"{args.output_path}/{str(args.aug_type)}"
    if not os.path.exists(tensorboard_path):
        os.mkdir(tensorboard_path)
    writer = SummaryWriter(log_dir=tensorboard_path)
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
    
    train_set = load_augmentation(train_set, args, edge=True)
    test_set = load_augmentation(test_set, args, edge=True)
    edge_set = load_augmentation(Subset(test_set, range(400)), args, edge=True)

    cur = 0
    for idx, (img, label) in enumerate(train_set):
        with open("./data.txt", "w+") as f:
            f.write(str(img.numpy()))
        plt.imshow(np.transpose(img.numpy(), (1,2,0)))
        plt.savefig(f"train_{idx}.png")
        cur += 1
        if cur > 3:
            exit(1)
            break

    train_loader = DataLoader(train_set, batch_size=train_batch, shuffle=True, num_workers=1) # , pin_memory=True
    test_loader = DataLoader(test_set, batch_size=train_batch, shuffle=False, num_workers=1) # , pin_memory=True
    edge_loader = DataLoader(edge_set, batch_size=edge_batch, shuffle=False, num_workers=1) # , pin_memory=True
    
    
    optimizer = torch.optim.Adam(victim.parameters(), lr=args.lr)

    if args.function == "test" or args.function == "attack":
        victim.load_state_dict(torch.load(f"{args.save_path}/{str(args.aug_type)}.pkl"))

    if args.function == "train" or args.function is None:
        val_loss = float("inf")
        for epoch in range(args.epochs):
            start = time.time()
            print("Start epoch: %d/%d" % (epoch+1, args.epochs))
            train_loss, train_acc, victim = train(args, victim, train_loader, optimizer, cuda=cuda)
            print("TRAIN: time: [%s], loss: %.4f, accuracy: %.4f" % (time_since(start), train_loss["mean"], train_acc["mean"]))
            writer.add_scalar("train/loss", train_loss["mean"], epoch)
            writer.add_scalar("train/loss_std", train_loss["std"], epoch)

            if args.function == "test" or args.function is None:
                with open(f"{args.output_path}/{str(args.aug_type)}.txt", "a+") as f:
                    f.write(f"epoch: {epoch+1}/{args.epochs}\n")
                test_loss, test_acc = test(args, victim, test_loader, cuda=cuda)
                if val_loss > test_loss["mean"]:
                    best_state = victim.state_dict()
                    val_loss = test_loss["mean"]
                
                writer.add_scalar("test/loss", test_loss["mean"], epoch)
                writer.add_scalar("test/loss_std", test_loss["std"], epoch)

        torch.save(best_state, f"{args.save_path}/{str(args.aug_type)}.pkl")
    
    if (args.function == "test" or args.function is None) and not args.function == "train":
        test_loss, test_acc = test(args, victim, test_loader, cuda=cuda)
        torch.save(victim.state_dict(), f"{args.save_path}/{str(args.aug_type)}.pkl")
        # write accuracy

    if args.function == "attack" or args.function is None:
        pass
        # 1. store trained images
        # 2. train images
        # 3. attack
        # 4. store attack result & metric
        # 5. visualize by TensorBoard
    writer.close()


if __name__ == "__main__":
    args = args.return_args()
    main(args)
