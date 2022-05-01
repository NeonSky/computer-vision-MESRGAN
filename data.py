import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
import random
from PIL import Image
# IDK what this does but it might help... https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images 
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    """ Dataset class used for training super-resolution 
    Args:
        root_dirs (list or str): List of strings or single string of image folder/s. 
        transform (transform): PyTorch transforms class
        file_ext (str): Expected file extension of images, currently only supports one file extension.    
    """
    def __init__(self, root_dirs, transform=None, file_ext="png"):

        self.image_paths = []
        if not isinstance(root_dirs, list):
            root_dirs = [root_dirs]
         
        for root_dir in root_dirs:
            root_dir = Path(root_dir)
            if not (root_dir.exists() and root_dir.is_dir()):
                raise ValueError(f"Invalid root dir: {root_dir}")
            
            
            root_image_paths = list(glob(str(root_dir / ("**/*." + file_ext)), recursive=True))

            self.image_paths.extend(root_image_paths)
            
        self.image_paths.sort()
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.image_paths)


def plot_image(axis, image_tensor):
    image_data = image_tensor.permute(1, 2, 0).numpy()
    axis.imshow(image_data)
    axis.grid(None)


def plot_images(image_tensors):
    fig, axes = plt.subplots(1, len(image_tensors), dpi=150)
    fig.set_size_inches(15.0, 5.0)
    for (axis, image_tensor) in zip(axes, image_tensors):
        plot_image(axis, image_tensor)


def calc_dataset_mean_and_var(dataset_folder):
    dataset = ImageDataset(dataset_folder, transforms.ToTensor())
    image_channels = dataset[0].shape[0]

    mean = torch.zeros(image_channels)
    for i, img in enumerate(dataset):
        img = torch.flatten(img, 1)
        mean += img.mean(1)
    mean /= len(dataset)

    var = torch.zeros(image_channels)
    n = 0
    for i, img in enumerate(dataset):
        img = torch.flatten(img, 1)
        n += img.shape[1]
        var += ((img - mean.unsqueeze(1))**2).sum(1)

    std = torch.sqrt(var / n)
    return mean, std

class RandomRotationsTransform:
    def __init__(self, set_of_degrees):
        self.set_of_degrees = set_of_degrees

    def __call__(self, x):
        angle = random.choice(self.set_of_degrees)
        return TF.rotate(x, angle)
   