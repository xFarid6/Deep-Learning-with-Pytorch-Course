# made of neurons with learnable weights and biases
# mainly work on image data
# use convolutional filters to process the image
# use pooling to reduce the size of the output
# use fully connected layers to process the output
# use activation functions to process the output
# use softmax to compute the probabilities
# use cross entropy loss to compute the loss

# use filters on images (like we did on the cv course in the past, but with pytorch)
# getting the correct size is SUPER IMPORTANT

# max pooling 
# max pooling is a technique that takes the maximum value from a pool of values
# max pooling is used to reduce the size of the output
# used to downsize an image, applies a maximum filter to subregions of the image
# used to reduce the number of parameters in the model
# reduce the computational cost by reducing the size, so the number of parameters and reduce overfitting 
# by appliyng abstracted images to the model


# convolutional neural network

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# hyper parameters (learning rate, batch size, epochs)
num_epochs = 4
batch_size = 4
learning_rate = 0.001


# dataset has PILImage images of range [0, 1]
# we transform the images to Tensors of normalized range [-1, 1]
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform) 


# data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                            shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                            shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 
            'frog', 'horse', 'ship', 'truck')


# helper function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# implement the convolutional neural network
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        '''self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(512, num_classes)'''

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 100) # fully connected layer, input is the output of the convolutional layer
        self.fc2 = nn.Linear(120, 84) # fully connected layer
        self.fc3 = nn.Linear(84, 10) # fully connected layer, 10 is fixed, because we have 10 classes and also the output of the convolutional layer is a tensor of size 16 * 5 * 5

    def forward(self, x):
        '''out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        # no softmax here
        return out'''
        x = self.pool(F.relu(self.conv1(x))) # convolutional layer
        x = self.pool(F.relu(self.conv2(x))) # output is a tensor of size 16 * 5 * 5
        x = x.view(-1, 16 * 5 * 5)  # flatten the output of the convolutional layer
        x = F.relu(self.fc1(x)) # fully connected layer
        x = F.relu(self.fc2(x)) # fully connected layer
        x = self.fc3(x) # fully connected layer
        return x    # no softmax here


# create model
model = ConvNet().to(device)


# define loss and optimizer
criterion = nn.CrossEntropyLoss() # because is a multiclass classification problem 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# TODO: is a good idea to wrap all training in a function ? since it is always the same...
# def train(model, device, train_loader, optimizer, criterion):
# train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input layer: 3 input channels, 6 output channels, 5x5 kernel
        # move tensors to GPU
        images, labels = images.to(device), labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished training')


# test the model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        # move tensors to GPU
        images, labels = images.to(device), labels.to(device)
        # forward pass
        outputs = model(images)
        # get the max predicted labels
        _, predicted = torch.max(outputs, 1)
        # compare them to the ground truth
            # correct = predicted.eq(labels).sum() # gc
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy: {acc}%')
    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy for class {i}: {acc}%')


# save the model
torch.save(model.state_dict(), './models/cnn_test_model.ckpt')
print('Saved model')
