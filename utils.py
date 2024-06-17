import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import pickle
import numpy as np
from PIL import Image
class Dataset:
    def __init__(self, dict, transforms):
        self.data = dict[b'data']
        self.label = dict[b'labels']
        self.transforms = transforms
    def __len__(self):
        return len(self.label)
    def __getitem__(self, i):
        img = self.data[i]
        tensor_img = torch.tensor(np.resize(img, (3,32,32)))
        return self.transforms(tensor_img), self.label[i]

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs, nrow=10)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data(args):
    transforms = torchvision.transforms.Compose([
        # torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    path = os.path.join(args.dataset_path, "data_batch_1")
    data_dict = unpickle(path)
    for i in range(2,6):
        path = os.path.join(args.dataset_path, f"data_batch_{i}")
        data_temp = unpickle(path)
        data_dict[b'data'] = np.concatenate((data_dict[b'data'], data_temp[b'data']), axis = 0)
        data_dict[b'labels'] = np.concatenate((data_dict[b'labels'], data_temp[b'labels']), axis = 0)
    dataset = Dataset(data_dict, transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

