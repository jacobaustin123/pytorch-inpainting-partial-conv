import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import argparse

from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from plotting import VisdomLinePlotter
from model import Model, VGG16FeatureExtractor
from loss import Loss
import utils
import os
import collections
import imageio

loaded = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model = Model()
criterion = VGG16FeatureExtractor()
loss_func = Loss(criterion)

model.to(device)
criterion.to(device)
loss_func.to(device)

optimizer = optim.Adam(model.parameters(), lr=3e-4)

if os.path.exists("/home/jacob/Documents/inpainting/parameters.pt"):
    try:
        state_dict = torch.load("/home/jacob/Documents/inpainting/parameters.pt")
        new_dict = collections.OrderedDict()

        for key, value in state_dict.items():
            new_dict[key.replace("module.", "")] = value

        print(model.load_state_dict(new_dict))
        print("Loaded saved parameters dict")
        loaded = True
    except Exception as e:
        breakpoint()
        print(e)
        print("Could not load parameters dict")
        raise FileNotFoundError("parameter not found")
        loaded = False
        exit(0)

def network(image, mask):
    model.eval()

    image, mask = image.to(device), mask.to(device)
    masked = image * (1 - mask)
    
    return model(masked, (1 - mask))

def format(image):
    image = image.detach().cpu().squeeze()
    if len(image.shape) != 3:
        image = image.unsqueeze(0)

    return image.permute(1, 2, 0).numpy()

if __name__ == "__main__":
    masks = ImageFolder("masks", transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.Grayscale(), transforms.ToTensor()]))
    train = ImageFolder("server/static/images", transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])) # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    index = torch.randint(0, 10, (1,))
    image = train[0][0].view(1, 3, 256, 256).cuda()
    mask = masks[index][0].view(1, 1, 256, 256).cuda()
    mask[mask != 0] = 1

    output = format(network(image, 1 - mask))
    print(output.shape)

    imageio.imwrite("result.png", output)
    imageio.imwrite("mask.png", format(mask))