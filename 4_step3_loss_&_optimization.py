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


X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32) # times 2

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return x * w


print(f'Prediction before training: forward(5) = {forward(5):.3f}')

# training
learning_rate = 0.01
n_iters = 100 # 20 # 10

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate) # stacasting (?) gradient descent

for epoch in range(n_iters):
    # prediction = forward pass
    y_predicted = forward(X)

    # loss = MSE
    l = loss(Y, y_predicted)

    # gradient = backward pass
    l.backward() 

    # update the weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'Epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')


print(f'Prediction after training: forward(5) = {forward(5):.3f}')