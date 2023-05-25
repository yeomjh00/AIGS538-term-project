import classifier
import time
import args
import attack
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from utils import *
import pickle
from augmentations import load_augmentation
import matplotlib.pyplot as plt

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
    # victim = classifier.ResNet().to(device)
    victim = torchvision.models.resnet18(pretrained=True)

    # returned data size: x = (batch_size, 3, 32, 32), y : Tensor of long = (batch_size)
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,\
                                             transform=_transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, \
                                            transform=_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=load_batch, shuffle=True, num_workers=3)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=load_batch, shuffle=False, num_workers=3)
    
    
    optimizer = torch.optim.Adam(victim.parameters(), lr=args.lr)

    if args.load:
        victim.load_state_dict(torch.load(f"{args.save_path}/{args.aug_type}.pkl"))
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
            print("time: [%s], loss: %.4f, accuracy: %.4f" % (time_since(start), train_loss["mean"], train_acc["mean"]))
        torch.save(best_state, f"{args.save_path}/{args.aug_type}.pkl")
    
    if args.function == "test" or args.function is None:
        test_loss, test_acc = test(args, victim, test_loader, optimizer, augmentation, cuda=cuda)
        # write accuracy

    if args.function == "attack" or args.function is None:
        
        # store trained images
        # if not args.load:
        #     print("You have to load your trained model first...")
        #     return
        
        cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
        cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]

        dm = torch.as_tensor(cifar10_mean, dtype=torch.float)[:, None, None]
        ds = torch.as_tensor(cifar10_std, dtype=torch.float)[:, None, None]
        def plot(tensor, isResult):
            tensor = tensor.clone().detach()
            tensor.mul_(ds).add_(dm).clamp_(0, 1)
            if tensor.shape[0] == 1:
                plt.imshow(tensor[0].permute(1, 2, 0).cpu())
                if isResult:
                    plt.savefig("rec_result.png")
                else:
                    plt.savefig("ground_truth.png")

            else:
                fig, axes = plt.subplots(1, tensor.shape[0], figsize=(12, tensor.shape[0]*12))
                for i, im in enumerate(tensor):
                    axes[i].imshow(im.permute(1, 2, 0).cpu())

        victim.to(device)
        victim.eval()

        target_id = np.random.randint(len(test_loader.dataset))
        ground_truth, labels = test_loader.dataset[target_id]
        ground_truth, labels = (
            ground_truth.unsqueeze(0).to(device),
            torch.as_tensor((labels,), device=device),
        )
            
        img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])

        plot(ground_truth, False)
        print([test_loader.dataset.classes[l] for l in labels])

        victim.zero_grad()
        target_loss = F.cross_entropy(victim(ground_truth), labels)
        input_gradient = torch.autograd.grad(target_loss, victim.parameters())
        input_gradient = [grad.detach() for grad in input_gradient]
        
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
        output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=img_shape)

        test_mse = (output.detach() - ground_truth).pow(2).mean()
        feat_mse = (victim(output.detach())- victim(ground_truth)).pow(2).mean()  
        # test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1/ds)

        # visualization tools
        plot(output, True)
        plt.title(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
                # f"| PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |")
                f"| FMSE: {feat_mse:2.4e} |")
        
        # leakage_score = leakage_metric(img, target)
        # accuracy = victim(img_batch)

        # store accuracy, learkage score, image

if __name__ == "__main__":
    args = args.return_args()
    main(args)