import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import argparse
import itertools

from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from plotting import VisdomLinePlotter
from model import Model, VGG16FeatureExtractor
from loss import Loss
import utils
import os
import json

plotter = VisdomLinePlotter()

parser = argparse.ArgumentParser(description='An implementation of the 2018 Image Inpainting for Irregular Holes Using Partial Convolutions paper in PyTorch')

parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for Adam Optimizer')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--resume', action='store_true', default=False, help='resume from existing save file')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to run')
parser.add_argument('--single_gpu', default=False, action='store_false', help='use only a single GPU')
parser.add_argument('--reset', default=False, action='store_true', help='delete saved parameters and reset all configurations')
parser.add_argument('--savedir', type=str, default='saved', help='directory with saved configurations and parameters')
parser.add_argument('--valid_freq', type=int, default=500, help='frequency of evaluation on valid dataset')

args = parser.parse_args()

mean = torch.Tensor([0.485, 0.456, 0.406])
std = torch.Tensor([0.229, 0.224, 0.225])

def unnormalize(image):
    return (image.transpose(1, 3) * std + mean).transpose(1, 3)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
masks = ImageFolder("masks", transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.Grayscale(), transforms.ToTensor()]))
train = ImageFolder("images/train", transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)])) # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
valid = ImageFolder("images/val", transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize(mean = mean, std = mean)])) # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

std = std.to(device)
mean = mean.to(device)

train_loader = DataLoader(train, num_workers=4, batch_size=args.batch_size, pin_memory=True, drop_last=True, shuffle=True)
valid_loader = DataLoader(valid, num_workers=0, batch_size=args.batch_size, pin_memory=True, drop_last=True, shuffle=True)

mask_loader = iter(DataLoader(masks, num_workers=0, batch_size=args.batch_size, pin_memory=True, drop_last=True))

print("Training partial convolution inpainting model with parameters {}".format(args))
print("Train dataset is {}, valid dataset is {}".format(train, valid))

model = Model()
criterion = VGG16FeatureExtractor()
loss_func = Loss(criterion)

if torch.cuda.device_count() > 1 and not args.single_gpu:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(device)
criterion.to(device)
loss_func.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr) # betas=(0.5, 0.999)

def try_remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

saved_params = os.path.join(args.savedir, "parameters.pt")
saved_stats = os.path.join(args.savedir, "stats.json")

if args.reset:
    try_remove(saved_params)
    try_remove(saved_stats)

if os.path.exists(saved_params) and args.resume:
    try:
        model.load_state_dict(torch.load(saved_params))
        print("Loaded saved parameters dict")
    except:
        print("Could not load parameters dict")

if os.path.exists(saved_stats):
    with open(saved_stats, "r") as f:
        stats = json.load(f)

    lowest = stats["lowest"]
else:
    lowest = 1E50

def network(image):
    global mask_loader
    try:
        mask = next(mask_loader)[0].to(device)
    except Exception as e:
        print("finished mask loader with exception {}".format(e))
        mask_loader = iter(DataLoader(masks, num_workers=0, batch_size=args.batch_size, pin_memory=True))
        mask = next(mask_loader)[0].to(device)

    model.train()
    image, mask = image[0].to(device), mask.to(device)
    mask[mask != 1] = 0
    
    masked = image * mask

    out = model(masked, mask)
    loss = loss_func(masked, mask, out, image)

    return loss, out, image, mask

for epoch in range(args.epochs):
    model.train()
    print("[EPOCH {}]".format(epoch))
    for i, image in enumerate(train_loader):
        loss, out, image, mask = network(image)
                
        print("[ITER {}] Loss {}".format(i, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            plotter.plot("In-Painting", "Loss", "Loss for In-Painting NN", epoch * len(train_loader) + i, float(loss.detach().cpu()), xlabel='iterations')
    
        if i % args.valid_freq == 0:
            model.eval()

            total_loss = 0
            for j, image in enumerate(itertools.islice(valid_loader, 50)):
                loss, out, image, mask = network(image)
                total_loss += float(loss.detach().cpu())

            total_loss /= j

            plotter.plot("In-Painting", "Valid Loss", "Loss for In-Painting NN", epoch * len(train_loader) + i, total_loss, xlabel='iterations')

            plotter.imshow("masks", mask)
            plotter.imshow("masked", mask * unnormalize(image) + (1 - mask) * unnormalize(out).clamp(0., 1.))
            plotter.imshow("ground-truth", unnormalize(image))
    
    plotter.imshow("masks", mask)
    plotter.imshow("masked", mask * unnormalize(image) + (1 - mask) * unnormalize(out).clamp(0., 1.))
    plotter.imshow("ground-truth", unnormalize(image))

    if total_loss < lowest:
        print("[EPOCH {}] Lowest loss {} found".format(epoch, total_loss))
        torch.save(model.state_dict(), saved_params)
        lowest = total_loss

        with open(saved_stats, "w") as f:
            json.dump({"lowest", lowest}, f)