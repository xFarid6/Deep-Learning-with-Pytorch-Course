# 1.
# Prediction: Manually
# Gradients Computation: Manually
# Loss Computation: Manually
# Parameter updates: Manually

# 2.
# Prediction: Manually
# Gradients Computation: Autograd
# Loss Computation: Manually
# Parameter updates: Manually

# 3.
# Prediction: Manually
# Gradients Computation: Autograd
# Loss Computation: Pytorch loss
# Parameter updates: Pytorch optimizer

# 4.
# Prediction: Pytorch model
# Gradients Computation: Autograd
# Loss Computation: Pytorch loss
# Parameter updates: Pytorch optimizer

# now 1 and 2, next 3 and 4


import numpy as np

# linear regression
# y = w*x + b
# f = 2 * x <- our formula

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32) # times 2

w = 0.0

# model prediction
def forward(x):
    return x * w


# loss function = MSE (mean squared error)
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()


# gradient of the loss function
# MSE = 1/N * (w*x - y)**2
# dL/dw = 1/N * 2 * x * (w*x - y)
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted - y).mean()


print(f'Prediction before training: forward(5) = {forward(5):.3f}')

# training
learning_rate = 0.01
n_iters = 20 # 10

for epoch in range(n_iters):
    # prediction = forward pass
    y_predicted = forward(X)

    # loss = MSE
    l = loss(Y, y_predicted)

    # gradient = dL/dw
    dw = gradient(X, Y, y_predicted)

    # update the weights
    w -= learning_rate * dw

    print(f'Epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')


print(f'Prediction after training: forward(5) = {forward(5):.3f}')