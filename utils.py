import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from typing import Callable
import time
import math

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def set_seed(seed, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def compute_accuracy(logits, label):
    pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
    accuracy = torch.mean((pred == label).float()) * 100
    return accuracy

def train(args, model: torch.nn.Module , train_set, optimizer, cuda=True):
    loss_epoch = []
    accuracy_epoch = []
    device = "cuda" if cuda else "cpu"

    model.train()

    for batch_idx, (image, label) in enumerate(train_set):

        image, label = image.to(device), label.to(device)

        logits = model(image)
        loss = F.cross_entropy(logits, label)

        accuracy = 0
        if args.aug_type is None:
            accuracy = compute_accuracy(logits, label).to("cpu").numpy()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch.append(loss.item())
        accuracy_epoch.append(accuracy)

    loss_mean_epoch     = np.mean(loss_epoch)
    loss_std_epoch      = np.std(loss_epoch)

    accuracy_mean_epoch = np.mean(accuracy_epoch)
    accuracy_std_epoch  = np.std(accuracy_epoch)

    loss = {'mean' : loss_mean_epoch, 'std' : loss_std_epoch}
    accuracy = {'mean' : accuracy_mean_epoch, 'std' : accuracy_std_epoch}

    return (loss, accuracy, model)

def test(args, model: torch.nn.Module , test_set, cuda=True):
    loss_epoch      = []
    accuracy_epoch  = []
    device = "cuda" if cuda else "cpu"

    with torch.no_grad():

        for batch_idx, (image, label) in enumerate(test_set):
            image, label = image.to(device), label.to(device)

            logits = model(image)
            loss= F.cross_entropy(logits, label)
            accuracy = 0 
            if args.aug_type is None:
                accuracy = compute_accuracy(logits, label).to("cpu").numpy()

            loss_epoch.append(loss.item())
            accuracy_epoch.append(accuracy)
        
            with open(f"{args.output_path}/{str(args.aug_type)}.txt", "a+") as f:
                f.write(f"{batch_idx}th batch: loss/acc: {loss.item()}/{accuracy}")
                f.write("\n")

        loss_mean_epoch = np.mean(loss_epoch)
        loss_std_epoch = np.std(loss_epoch)

        accuracy_mean_epoch = np.mean(accuracy_epoch)
        accuracy_std_epoch = np.std(accuracy_epoch)

        with open(f"{args.output_path}/{str(args.aug_type)}.txt", "a+") as f:
            f.write(f"Total mean/std of loss: {loss_mean_epoch}/{loss_std_epoch}")
            f.write("\n")
            f.write(f"Total mean/std of acc: {accuracy_mean_epoch}/{accuracy_std_epoch}")
            f.write("\n")

        loss = {'mean' : loss_mean_epoch, 'std' : loss_std_epoch}
        accuracy = {'mean' : accuracy_mean_epoch, 'std' : accuracy_std_epoch}

    return (loss, accuracy)  
