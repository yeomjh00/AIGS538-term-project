import classifier
import time
import args
import attack.attack as attack
import attack.metrics as metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import cv2
from utils import *
import pickle
from augmentations import load_augmentation

def main(args):
    set_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)
    tensorboard_path = f"{args.output_path}/{str(args.aug_type)}{str(args.name)}"
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
    
    # train_set = load_augmentation(train_set, args, edge=False)
    # test_set = load_augmentation(test_set, args, edge=False)
    # edge_set = load_augmentation(Subset(test_set, range(20)), args, edge=True)

    # train_loader = DataLoader(train_set, batch_size=train_batch, shuffle=True, num_workers=1) # , pin_memory=True
    # test_loader = DataLoader(test_set, batch_size=train_batch, shuffle=False, num_workers=1) # , pin_memory=True
    # edge_loader = DataLoader(edge_set, batch_size=edge_batch, shuffle=False, num_workers=1) # , pin_memory=True
    
    optimizer = torch.optim.SGD(victim.parameters(), \
                                lr=args.lr, \
                                momentum=args.momentum, \
                                weight_decay=args.weight_decay, \
                                nesterov=True)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1)

    if args.function == "test" or args.function == "attack":
        victim.load_state_dict(torch.load(f"{args.save_path}/{str(args.aug_type)}{str(args.name)}.pkl", map_location=args.device))

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
                with open(f"{args.output_path}/{str(args.aug_type)}{str(args.name)}.txt", "a+") as f:
                    f.write(f"epoch: {epoch+1}/{args.epochs}\n")
                test_loss, test_acc = test(args, victim, test_loader, cuda=cuda)
                if val_loss > test_loss["mean"]:
                    best_state = victim.state_dict()
                    val_loss = test_loss["mean"]
                
                writer.add_scalar("test/loss", test_loss["mean"], epoch)
                writer.add_scalar("test/loss_std", test_loss["std"], epoch)
            
            scheduler.step()

        torch.save(best_state, f"{args.save_path}/{str(args.aug_type)}{str(args.name)}.pkl")
    
    if (args.function == "test" or args.function is None) and not args.function == "train":
        test_loss, test_acc = test(args, victim, test_loader, cuda=cuda)
        torch.save(victim.state_dict(), f"{args.save_path}/{str(args.aug_type)}{str(args.name)}.pkl")
        # write accuracy

    if args.function == "attack" or args.function is None:
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, \
                                            transform=_transform)

        # 1. prerequisites
        if not os.path.exists(args.attack_path):
            os.mkdir(args.attack_path)
        if not os.path.exists(f"{args.attack_path}/{str(args.aug_type)}{str(args.name)}"):
            os.mkdir(f"{args.attack_path}/{str(args.aug_type)}{str(args.name)}")

        cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
        cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]

        dm = torch.as_tensor(cifar10_mean, dtype=torch.float).to(device)[:, None, None]
        ds = torch.as_tensor(cifar10_std, dtype=torch.float).to(device)[:, None, None]


        sample_list = []
        for i, (ground_truth, labels) in enumerate(test_set):
            if i > 1: break
            sample_list.append((ground_truth, labels))
        model = str(args.aug_type) + args.name

        if not os.path.exists(args.attack_path):
            os.mkdir(args.attack_path)
        if not os.path.exists(f"{args.attack_path}/{str(model)}"):
            os.mkdir(f"{args.attack_path}/{str(model)}")
        _sample_list = sample_list.copy()
        victim.load_state_dict(torch.load(f"{args.save_path}/{str(model)}.pkl", map_location=args.device))
        victim.to(device)
        victim.eval()
        for i, (ground_truth, labels) in enumerate(_sample_list):

            # if args.aug_type is None:
            ground_truth, labels = (
                ground_truth.unsqueeze(0).to(device),
                torch.as_tensor((labels,), device=args.device),
                )
            # else:
            #     ground_truth, labels = (
            #         ground_truth.unsqueeze(0).to(device),
            #         labels.unsqueeze(0).to(device),
            #     )
            img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])

            # 2. train images
            victim.zero_grad()
            target_loss = F.cross_entropy(victim(ground_truth), labels)
            input_gradient = torch.autograd.grad(target_loss, victim.parameters())
            input_gradient = [grad.detach() for grad in input_gradient]
            
            # 3. attack
            # have to modify values according to several options(?)
            config = dict(
                signed=True,
                boxed=True,
                cost_fn='sim',
                indices='def',
                weights='equal',
                lr=0.1,
                optim='adam',
                restarts=1,
                max_iterations=4000,
                total_variation=1e-6,
                init='randn',
                filter='none',
                lr_decay=True,
                scoring_choice='loss',
            )

            rec_machine = attack.GradientReconstructor(victim, (dm, ds), config)
            output, stats, mid_output = rec_machine.reconstruct(input_gradient, labels, img_shape=img_shape)

            test_mse = (output.detach() - ground_truth).pow(2).mean()
            feat_mse = (victim(output.detach())- victim(ground_truth)).pow(2).mean()  
            test_psnr = metrics.psnr(output, ground_truth, factor=1/ds)

            # 4. store attack result & metric
            output_denormalized = torch.clamp(output * ds + dm, 0, 1)
            attck_filename = f"attck_{args.name}_idx{i}.png"
            torchvision.utils.save_image(output_denormalized, f"{args.attack_path}/{str(model)}/"+attck_filename)
            gt_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)
            victim_filename = f"victim_{args.name}_idx{i}.png"
            torchvision.utils.save_image(gt_denormalized, f"{args.attack_path}/{str(model)}/"+victim_filename)

            print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |")
            save_to_table(
                f"{args.attack_path}/{str(model)}/",
                name=f"exp_{model}",
                dryrun=False,
                rec_loss=stats["opt"],
                psnr=test_psnr,
                test_mse=test_mse,
                feat_mse=f"{feat_mse:2.4e}",
                index=i,
                attck_img=attck_filename,
                victim_img=victim_filename,
            )

            data = metrics.activation_errors(victim, output, ground_truth)

            fig, axes = plt.subplots(2, 3, sharey=False, figsize=(14,8))
            axes[0, 0].semilogy(list(data['se'].values())[:-3])
            axes[0, 0].set_title('SE')
            axes[0, 1].semilogy(list(data['mse'].values())[:-3])
            axes[0, 1].set_title('MSE')
            axes[0, 2].plot(list(data['sim'].values())[:-3])
            axes[0, 2].set_title('Similarity')

            convs = [val for key, val in data['mse'].items() if 'conv' in key]
            axes[1, 0].semilogy(convs)
            axes[1, 0].set_title('MSE - conv layers')
            convs = [val for key, val in data['mse'].items() if 'conv1' in key]
            axes[1, 1].semilogy(convs)
            convs = [val for key, val in data['mse'].items() if 'conv2' in key]
            axes[1, 1].semilogy(convs)
            axes[1, 1].set_title('MSE - conv1 vs conv2 layers')
            bns = [val for key, val in data['mse'].items() if 'bn' in key]
            axes[1, 2].plot(bns)
            axes[1, 2].set_title('MSE - bn layers')
            fig.suptitle('Error between layers')
            plt.savefig(f"{args.attack_path}/{str(model)}/metrics_idx{i}.png")

            # 5. visualize by TensorBoard
            if config["restarts"] == 1:
                for i in range(len(mid_output)):
                    for j in range(len(mid_output[i])):
                        writer.add_image("image/iteration", pixel_0_to_255(mid_output[i][j].squeeze(0).cpu()), j)

        writer.close()


if __name__ == "__main__":
    args = args.return_args()
    main(args)
