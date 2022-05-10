# so far we load the data
# then put it in the training loop
# and optimize the model based on the whole dataset, 
# very time consimuing if we use gradient calculation over the entire dataset

# from now we'll divide the data into baches and loop over the batches
# we then get x_batch and y_batch samples and optimize based on those batches
# pytorch automatically divides data into batches for us and we don't need to worry about that
# it'll also automatically shuffle the data for us before we get the next batch
# and perform the necessary calculations

'''
epoch: 1 forward and backward pass of ALL training samples

batch size: number of training samples in one forward and backward pass

number of iterations: number of forward and backward passes, each passing using [batch size] number of samples

e.g. 100 samples, batch size = 20 --> 100/20 = 5 iterations (5 forward and backward passes) for each epoch
'''

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math


class WineDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # n_samples, 1
        self.n_samples = xy.shape[0]

    def __len__(self):
        return self.n_samples # len(self.data)

    def __getitem__(self, idx):
        # return self.data[idx], self.labels[idx]
        return self.x[idx], self.y[idx]

dataset = WineDataset()
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)
dataLoader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

'''
dataiter = iter(dataLoader)
data = dataiter.next()
features, labels = data
print(features, labels)
'''

# training loop
# we'll use the same model and optimizer as before
# we'll also use the same criterion and learning rate
# we'll also use the same number of epochs
# we'll also use the same batch size
# we'll also use the same shuffle
# we'll also use the same number of workers

num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)
print(f'Total samples: {total_samples}', f'Number of iterations: {n_iterations}')

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataLoader):
        # forward, backward, update
        if (i+1) % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, iteration {i+1}/{n_iterations}, inputs {inputs.shape}, labels {labels.shape}')

torchvision.dataset.MNIST()
# fashion mnist, cifrar, coco