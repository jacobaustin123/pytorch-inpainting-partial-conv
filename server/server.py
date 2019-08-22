from flask import Flask, render_template, flash, redirect, session, url_for, request, g, Markup, jsonify, send_from_directory
import os
import imageio
import io
import numpy as np
import cv2
import base64
import datetime
import sys
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

sys.path.append('/home/jacob/Documents/inpainting')
import inpainting

app = Flask(__name__)

@app.route("/")
def canvas():
    return send_from_directory(".", "canvas.html")

images = ImageFolder("static/images", transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()])) # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

@app.route("/upload", methods=["POST"])
def upload():    
    mask = request.form["mask"]
    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")

    img = images[0][0].unsqueeze(0)
    mask_str = base64.decodestring(mask.split(',')[1].encode())
    mask = torch.Tensor(imageio.imread(mask_str)).permute(2, 0, 1).mean(0).unsqueeze(0).unsqueeze(0)
    mask[mask != 0] = 1.0

    imageio.imwrite("masktest.png", mask.cpu().squeeze().numpy())
    print(img.shape, mask.shape)

    print(mask)

    # with open(os.path.join("static/images", "example.jpeg"), "r") as f:
    #     f.write(base64.decodestring(img.split(',')[1].encode()))

    with open(os.path.join("static/masks/masks", timestamp + ".png"), "wb") as f:
        f.write(mask_str)

    #with open(os.path.join("static/results", timestamp + ".png"), "wb") as f:
    result = inpainting.network(img, mask).detach().cpu().squeeze().permute(1, 2, 0).numpy()
    print(result.shape, mask.shape, img.shape)
    imageio.imwrite(os.path.join("static/results", timestamp + ".png"), result)

    return timestamp, 200

# @app.route("/getimage", methods=["GET"])
# def getimage():
#     timestamp = request.args["timestamp"]
#     return send_from_directory("static/images", timestamp + ".png"), 200

@app.route("/getmask", methods=["GET"])
def getmask():
    timestamp = request.args["timestamp"]
    return send_from_directory("static/masks/masks", timestamp + ".png"), 200

@app.route("/getresult", methods=["GET"])
def getresult():
    timestamp = request.args["timestamp"]
    return send_from_directory("static/results", timestamp + ".png"), 200