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

def train(args, model: torch.nn.Module , train_set, optimizer, augmentation: Callable, cuda=True):
    loss_epoch = []
    accuracy_epoch = []
    device = "cuda" if cuda else "cpu"

    model.train()
    total_images = 50000

    batch = args.additional_augment + args.batch_size if args.aug_type else args.batch_size
    cur = 0

    for index_batch, (image, label) in enumerate(train_set):
        cur += batch
        if cur % 10000 == 0:
            print(f"{cur/50000 * 100}% of single epoch is done")
        image, label = image.to(device), label.to(device)
        onehot_label = F.one_hot(label, num_classes=10).float().to(device)
        if augmentation is not None:
            image, onehot_label = augmentation(args, image, onehot_label)
        
        logits = model(image)
        loss = F.cross_entropy(logits, onehot_label)
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

    loss        = {'mean' : loss_mean_epoch, 'std' : loss_std_epoch}
    accuracy    = {'mean' : accuracy_mean_epoch, 'std' : accuracy_std_epoch}

    return (loss, accuracy, model)   

def test(args, model: torch.nn.Module , test_set, augmentation: Callable, cuda=True):
    loss_epoch      = []
    accuracy_epoch  = []
    device = "cuda" if cuda else "cpu"

    with torch.no_grad():

        for index_batch, (image, label) in enumerate(test_set):
            image, label = image.to(device), label.to(device)
            onehot_label = F.one_hot(label, num_classes=10).float().to(device)

            logits = model(image)
            loss= F.cross_entropy(logits, onehot_label)
            accuracy = compute_accuracy(logits, label).to("cpu").numpy()

            loss_epoch.append(loss.item())
            accuracy_epoch.append(accuracy)

        loss_mean_epoch     = np.mean(loss_epoch)
        loss_std_epoch      = np.std(loss_epoch)

        accuracy_mean_epoch = np.mean(accuracy_epoch)
        accuracy_std_epoch  = np.std(accuracy_epoch)

        loss        = {'mean' : loss_mean_epoch, 'std' : loss_std_epoch}
        accuracy    = {'mean' : accuracy_mean_epoch, 'std' : accuracy_std_epoch}

    return (loss, accuracy)  
