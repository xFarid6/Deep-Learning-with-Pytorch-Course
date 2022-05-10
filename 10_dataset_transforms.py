'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet

complete list of built-in trans forms:
https://pytorch.org/docs/stable/torchvision/transforms.html

On Images
-----------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

On Tensors
------------
LinearTransformation, Normalize, RandomErasing, RandomFlip, RandomGrotation, RandomSizedCrop

Conversion
----------
ToTensor, ToPILImage

Generic
-------
ToDict, ToList, ToNumpy, ToRecord, ToRecordList, ToRecordTensor, ToTensorList
Use lambda functions to create custom transforms


Custom
-------
Write own class to create custom transforms

Compose multiple Transforms
----------------------------
Compose multiple transforms into one
composed = transforms.Compose([Rescale(256), 
                                RandomCrop(224)])

# torchvision.transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
torchvision.transorms.ReScale(256)
torchvision.transforms.ToTensor()
'''

import torch
import torchvision
from torch.utils.data import DataSet
import numpy as np


class MyDataSet(DataSet):
    def __init__(self, data, transform=None):
        xy = np.loadtxt(fname=data, delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        self.x = xy[:, 1:]
        self.y = xy[:, [0]]

        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = self.x[idx], self.y[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor:
    def __call__(self, sample):
        x, y = sample
        return torch.from_numpy(x), torch.from_numpy(y)
