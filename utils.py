import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from typing import Callable
import time
import math
import os
import csv

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

def pixel_0_to_1(image: torch.Tensor) -> torch.Tensor:
    """
    [-1, 1] values to [0, 1] values
    """
    img = image.clone()
    return (img + 1) / 2

def pixel_0_to_255(image: torch.Tensor) -> np.ndarray:
    """
    [-1, 1]-value tensor to [0, 255]-value tensor
    """
    img = image.clone()
    img = (((img + 1) / 2) * 255).numpy()
    img = img.astype(np.uint8)
    return img

def save_to_table(out_dir, name, dryrun, **kwargs):
    """Save keys to .csv files. Function adapted from Micah."""
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, f'table_{name}.csv')
    fieldnames = list(kwargs.keys())

    # Read or write header
    try:
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = [line for line in reader][0]
    except Exception as e:
        print('Creating a new .csv table...')
        with open(fname, 'w') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writeheader()
    if not dryrun:
        # Add row for this experiment
        with open(fname, 'a') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writerow(kwargs)
        print('\nResults saved to ' + fname + '.')
    else:
        print(f'Would save results to {fname}.')
        print(f'Would save these keys: {fieldnames}.')