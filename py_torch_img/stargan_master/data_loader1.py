from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

import torch
import os
import random
from PIL import Image


class Mytest(data.Dataset):

    def __init__(self, image_dir, imagename,  transform, image_size ):

        self.image_dir = image_dir
        self.transform = transform
        self.testdata = []
        self.imagename=imagename
        self.testdata.append([self.imagename])
        self.num_images = len(self.testdata)
        self.image_size=image_size
        
    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""

        image = Image.open(os.path.join(self.image_dir, self.imagename))
        image= image.convert('RGB')
        return self.transform(image)


    def __len__(self):
        """Return the number of images."""
        return self.num_images




def get_loader1(image_dir, imagename, dataset, image_size, num_workers=1):
    """Build and return a data loader."""
    transform = []
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = Mytest(image_dir, imagename, transform,image_size)
    data_loader = data.DataLoader(dataset=dataset, batch_size=1,  num_workers=num_workers)
    return data_loader