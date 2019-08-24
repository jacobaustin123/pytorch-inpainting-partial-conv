import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import argparse
import itertools
import PIL

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

parser.add_argument('--datadir', type=str, default="images/train", help='directory for training data')
parser.add_argument('--validdir', type=str, default="images/valid", help='directory for validation data')
parser.add_argument('--maskdir', type=str, default="masks/masks", help='directory for mask data')

parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for Adam Optimizer')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--resume', action='store_true', default=False, help='resume from existing save file')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to run')
parser.add_argument('--single_gpu', default=False, action='store_false', help='use only a single GPU')
parser.add_argument('--reset', default=False, action='store_true', help='delete saved parameters and reset all configurations')
parser.add_argument('--savedir', type=str, default='saved', help='directory with saved configurations and parameters')
parser.add_argument('--valid_freq', type=int, default=500, help='frequency of evaluation on valid dataset')
parser.add_argument('--freeze_bn', default=False, action='store_true', help='freeze the batchnorm layer for encoder layers')

args = parser.parse_args()

mean = torch.Tensor([0.485, 0.456, 0.406])
std = torch.Tensor([0.229, 0.224, 0.225])

def unnormalize(image):
    return (image.transpose(1, 3) * std + mean).transpose(1, 3)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
masks = ImageFolder(args.maskdir, transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.Grayscale(), transforms.Lambda(lambda img : PIL.ImageOps.invert(img)), transforms.ToTensor()]))
train = ImageFolder(args.datadir, transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)])) # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
valid = ImageFolder(args.validdir, transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize(mean = mean, std = mean)])) # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

std = std.to(device)
mean = mean.to(device)

train_loader = DataLoader(train, num_workers=4, batch_size=args.batch_size, pin_memory=True, drop_last=True, shuffle=True)
valid_loader = DataLoader(valid, num_workers=0, batch_size=args.batch_size, pin_memory=True, drop_last=True, shuffle=True)

mask_loader = iter(DataLoader(masks, num_workers=0, batch_size=args.batch_size, pin_memory=True, drop_last=True, shuffle=True))

print("Training partial convolution inpainting model with parameters {}".format(args))
print("Train dataset has size {}, valid dataset has size {}, mask dataset has size {}".format(len(train), len(valid), len(masks)))
print("Train directory is {}, valid directory is {}, mask directory is {}".format(args.datadir, args.validdir, args.maskdir))

model = Model(freeze_bn=args.freeze_bn)
criterion = VGG16FeatureExtractor()
loss_func = Loss(criterion)

if torch.cuda.device_count() > 1 and not args.single_gpu:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(device)
criterion.to(device)
loss_func.to(device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr) # betas=(0.5, 0.999)

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
        print("Loaded saved parameters dict {}".format(saved_params))
    except:
        print("Could not load parameters dict {}".format(saved_params))

if os.path.exists(saved_stats) and args.resume:
    with open(saved_stats, "r") as f:
        stats = json.load(f)

    lowest = stats["lowest"]
else:
    lowest = 1E50

print("CURRENT LOWEST LOSS IS {}".format(lowest))

def network(image, verbose=False):
    global mask_loader
    try:
        mask = next(mask_loader)[0].to(device)
    except Exception as e:
        print("finished mask loader with exception {}".format(e))
        mask_loader = iter(DataLoader(masks, num_workers=0, batch_size=args.batch_size, pin_memory=True, shuffle=True))
        mask = next(mask_loader)[0].to(device)

    image, mask = image[0].to(device), mask.to(device)

    mask[mask != 1] = 0

    masked = image * mask

    out = model(masked, mask)
    loss = loss_func(masked, mask, out, image, verbose=verbose)

    return loss, out, image, mask

def validate(n, valid=True):
    model.eval()

    print("Validating on {} samples from {} dataset".format(n, "valid" if valid else "train"))

    total_loss = 0
    first = True
    for j, image in enumerate(itertools.islice(valid_loader if valid else train_loader, n)):
        loss, out, image, mask = network(image, verbose=first)
        first=False
        total_loss += float(loss.detach().cpu())

    total_loss /= j

    if valid:
        plotter.plot("In-Painting", "Valid Loss", "Loss for In-Painting NN", epoch * len(train_loader) + i, total_loss, xlabel='iterations')
    else:
        plotter.plot("In-Painting", "Longer Train Loss", "Loss for In-Painting NN", epoch * len(train_loader) + i, total_loss, xlabel='iterations')

    if valid:
        plotter.imshow("trainmasks", mask)
        plotter.imshow("masked", mask * unnormalize(image) + (1 - mask) * unnormalize(out).clamp(0., 1.))
        plotter.imshow("output", unnormalize(out).clamp(0., 1.))
        plotter.imshow("maskedalt", mask * unnormalize(image))
        plotter.imshow("ground-truth", unnormalize(image))

    model.train()

    return total_loss

for epoch in range(args.epochs):
    model.train()

    print("[EPOCH {}]".format(epoch))
    for i, image in enumerate(train_loader):
        loss, out, image, mask = network(image)
                
        # print("[ITER {}] Loss {}".format(i, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            plotter.plot("In-Painting", "Loss", "Loss for In-Painting NN", epoch * len(train_loader) + i, float(loss.detach().cpu()), xlabel='iterations')
    
        if i % args.valid_freq == 0:
            validate(50)
            validate(50, valid=False)

    valid_loss = validate(1000)

    if valid_loss < lowest:
        print("[EPOCH {}] Lowest loss {} found".format(epoch, valid_loss))
        torch.save(model.state_dict(), saved_params)
        lowest = valid_loss

        with open(saved_stats, "w") as f:
            json.dump({"lowest" : lowest}, f)
    else:
        print("[EPOCH {}] loss is {}".format(epoch, valid_loss))