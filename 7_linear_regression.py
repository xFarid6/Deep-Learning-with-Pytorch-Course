# training pipeline
# model -> loss -> optimizer

# 1. design the model (input size, output size, forward pass and layers)
# 2. design the loss function and the optimizer
# 3. training loop
#  - forward pass: compute prediction
#  - backward pass: compute the gradient
#  - update the weights
# 4. evaluate the model


import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# 0. Prepare data
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

x = torch.from_numpy(x_numpy.astype(np.float32))    # convert numpy array to tensor and dtype float from double
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)  # reshape the tensor to have 1 column

n_samples, n_features = x.shape


# 1. design the model (input size, output size, forward pass and layers)
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)


# 2. loss function and the optimizer
criterion = nn.MSELoss() # built-in in pytorch, in case of linear regression is median squared error

# stacasting gradient descent, needs the parameters to optimize (allo those of the model) and a learnirng rate
optimizer =  torch.optim.SGD(model.parameters(), lr=0.01)


# 3. training loop
#  - forward pass: compute prediction
#  - backward pass: compute the gradient
#  - update the weights
n_epochs = 100
for epoch in range(n_epochs):
    # prediction = forward pass and loss
    y_predicted = model(x)
    loss = criterion(y_predicted, y)

    # gradient = backward pass
    loss.backward()

    # update the weights
    optimizer.step()

    # zero gradients, never forget!
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch + 1}: w = {model.weight.item():.3f}, loss = {loss.item():.4f}')


# plot the results
predicted = model(x).detach().numpy()   # generates a new tensor with no gradients and then convert to numpy array
plt.plot(x_numpy, y_numpy, 'ro', label='Original data')
plt.plot(x_numpy, predicted, label='Fitted line')
plt.legend()
plt.show()
