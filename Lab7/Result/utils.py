import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from os import listdir
from tqdm import tqdm
import json
import numpy as np
from dataset import iclevr_dataset

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

# def read_images(args, path):
#     transforms = torchvision.transforms.Compose([
#         torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
#         torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#     file_name = [f for f in listdir(path) if f.endswith('.png')]
#     images = []
#     for f in tqdm(file_name):
#         image = Image.open(path+f).convert('RGB')
#         images.append(transforms(image))
#     print('images number = ',len(images))
#     return images, file_name

# def read_labels(args, file_name):
#     object_dict = json.load(open(args.dataset_path+'/objects.json'))
#     train_dict = json.load(open(args.dataset_path+'/train.json'))
#     print('object_dict = ',object_dict)
#     labels = []
#     for f in file_name:
#         labels = train_dict[f]
#         tmp = np.zeros(24)
#         for l in labels:
#             tmp[object_dict[l]] = 1
#         labels.append(tmp)
#     print('labels number = ',len(labels))
#     return labels

def get_data(args):
    dataset = iclevr_dataset(args)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return data_loader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def label_to_onehot(label, label_dict):
    onehot = np.zeros(24, dtype=np.float32)
    for l in label:
        onehot[label_dict[l]] = 1
    return onehot