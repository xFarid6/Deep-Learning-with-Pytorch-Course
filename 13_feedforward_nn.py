# MNIST
# Dataloader, Transorms
# Multilayer Neural Network, activation functions
# Loss function, optimizer
# Train, Test, Evaluate
# Visualize the results
# Save the model
# GPU support if available

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters (learning rate, batch size, epochs)
input_size = 784    # our images will be 28x28 pixels
hidden_size = 200   # hidden layer size
num_classes = 10    # 10 classes, because we have 10 classes from 0-9
num_epochs = 10      # number of epochs
batch_size = 500    # batch size
learning_rate = 0.001 # learning rate, 0.001 is default for SGD optimizer in Pytorch (Stochastic Gradient Descent) 
                    # (0.01 is default for Adam optimizer in Pytorch) 
                    # (0.0001 is default for RMSProp optimizer in Pytorch) 
                    # (0.00001 is default for AdaGrad optimizer in Pytorch) 
                    # (0.000001 is default for AdamW optimizer in Pytorch) 

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                        transform=transforms.ToTensor(), 
                                        download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                        transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                            shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                            shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape) # [batch size, one channel (no colors), height, width] and for each class label we have one value (100)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0], cmap='gray')
    plt.title('Class: {}'.format(labels[i]))
# plt.show()

# we want to setup a fully connected nn with a hidden layer to recognize handwriting

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes): # num_classes is basically the output size
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) # input_size is the number of features
        self.relu = nn.ReLU() # as activation function
        self.linear2 = nn.Linear(hidden_size, num_classes) # output layer

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax at the end because we want to use cross entropy loss function instead
        return out


model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
model.to(device)


# loss function, applies softmax and cross entropy and optimizer Adam 
criterion = nn.CrossEntropyLoss() # loss function, applies softmax and cross entropy
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # optimizer


# train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs): # for each epoch
    for i, (images, labels) in enumerate(train_loader): # for each batch
        # from 100, 1, 28, 28 as seen before
        # to 100, 784 as seen in the neural net
        images = images.reshape(-1, 28*28).to(device) # flatten the images
        labels = labels.to(device) # labels are already in the right format

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels) # loss function

        # backward pass
        optimizer.zero_grad() # reset the gradients
        loss.backward() # backpropagate the loss
        optimizer.step() # update the weights

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}') # print the loss


# test the model on the test data and evaluation
with torch.no_grad(): # no need to track the gradients
    n_correct = 0
    n_samples = 0
    n_total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # get the index of the max log-probability
        _, predicted = torch.max(outputs, 1)   # returns value and index, we intrested in the index
        n_samples += labels.shape[0]
        n_correct += (predicted == labels).sum().item() # sum the tensor, returns a scalar

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy: {acc}%')


# save the model
torch.save(model.state_dict(), './models/hand_writing_model.ckpt')
print('Model saved!')