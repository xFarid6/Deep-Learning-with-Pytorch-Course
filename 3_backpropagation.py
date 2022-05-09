# The chain rule:
#   dC/dW = dC/dA * dA/dW
#   dC/db = dC/dA * dA/db
#   dC/dA = dC/dZ * dZ/dA

# first we compute the derivative of the loss with respect to the output
# we use the chain rule to compute the derivative of the loss with respect to the weights

# computational graph
# we calculate local gradients 

# Chain rule: dLoss / dx = dLoss / dC * dC / dx

# The whole concept is three steps:
# 1. compute the derivative of the loss with respect to the output
# 2. compute the derivative of the loss with respect to the weights
# 3. compute the derivative of the loss with respect to the input

# or

# Forward pass: compute loss
# Compute local gradients
# Backward pass: compute dLoss / dWeights using the chain rule

# loss is predicted y - actual y -> then we minimizee the loss; 
# for example with linear regression we minimize the mean squared error


import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# forward pass and compute the loss
y_hat = w * x
loss = (y_hat - y)**2
print(loss)

# backward pass
loss.backward() # this is the whole gradient computation
print(w.grad)

### update the weights
### next forward and backward pass, for a couple of iterations
