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
        # 0. prerequisites
        if not os.path.exists(args.attack_path):
            os.mkdir(args.attack_path)
        if not os.path.exists(f"{args.attack_path}/{str(args.aug_type)}"):
            os.mkdir(f"{args.attack_path}/{str(args.aug_type)}")

        cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
        cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]

        dm = torch.as_tensor(cifar10_mean, dtype=torch.float)[:, None, None]
        ds = torch.as_tensor(cifar10_std, dtype=torch.float)[:, None, None]

        victim.to(device)
        victim.eval()

        # 1. store trained images
        # if args.target_id is None:
        #     # target_id = np.random.randint(len(test_loader.dataset))
        #     target_id = np.random.randint(len(edge_loader.dataset))
        #     # edge set에서 데이터 잘 만들어지는지 확인
        # else:
        #     target_id = args.target_id
        

        for batch_idx, (image, label) in enumerate(edge_set):
            for i in range(image.shape[0]):
                ground_truth, labels = image[i], label[i]
                ground_truth, labels = (
                    ground_truth.unsqueeze(0).to(device),
                    torch.as_tensor((labels,), device=device),
                )
                img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])

                print([test_loader.dataset.classes[l] for l in labels])

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
                output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=img_shape)

                test_mse = (output.detach() - ground_truth).pow(2).mean()
                feat_mse = (victim(output.detach())- victim(ground_truth)).pow(2).mean()  
                test_psnr = metrics.psnr(output, ground_truth, factor=1/ds)

                # visualization tools
                # plt.title(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
                #         # f"| PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |")
                #         f"| FMSE: {feat_mse:2.4e} |")

                # 4. store attack result & metric
                


                # Save the resulting image
                # 
                # for i in range(0):
                #     victim_img = pixel_0_to_255(victim_img)
                #     attak_result = pixel_0_to_255(attak_result)
                #     cv2.imwrite(f"{args.attack_path}/{str(args.aug_type)}/victim_{args.name}_{i}.png", victim_img)
                #     cv2.imwrite(f"{args.attack_path}/{str(args.aug_type)}/attck_{args.name}_{i}.png", attak_result)


                # Save to a table:
                print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |")

                save_to_table(
                    args.table_path,
                    name=f"exp_{args.name}",
                    dryrun=False,
                    rec_loss=stats["opt"],
                    psnr=test_psnr,
                    test_mse=test_mse,
                    feat_mse=feat_mse,
                    target_id=target_id,
                    timing_attack=str(datetime.timedelta(seconds=time.time() - start_time)),
                    rec_img=rec_filename,
                    gt_img=gt_filename,
                )

                # Print final timestamp
                print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
                print("---------------------------------------------------")
                print(f"Finished computations with time (only for attack): {str(datetime.timedelta(seconds=time.time() - start_time))}")
                print("-------------Job finished.-------------------------")



                # 5. visualize by TensorBoard
            writer.add_image("image/iteration", pixel_0_to_255(img), epoch)
                
            writer.close()


if __name__ == "__main__":
    args = args.return_args()
    main(args)
