import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2

# with gradient we can backpropagate
# it means that we can compute the gradient of the loss with respect to the tensor
# we can use the gradient to update the tensor
# we add backward a grandient, grad_fm, to the tensor

y = x + 2
print(y)

z = y*y*2
print(z)
z = z.mean()
print(z)

# when calculating the gradints
z.backward() # will calculate the gradients of z with respect to x, dz/dx
print(x.grad)

# creates a vector jacobian matrix of the tensor
# jacobian matrix is a matrix of the derivatives of the tensor with respect to the tensor
# it is a matrix of the derivatives of the tensor with respect to the tensor
# in the backgroud this is a vector jackobian product

v = torch.tensor([0.1, 0.2, 1.0, 0.001], dtype=torch.float32)
# z.backward(v) # dz/dx = v
print(x.grad)


# to avoid tracking gradients use
# with torch.no_grad()
# torch.requires_grad(False)
# torch.detach()

x = torch.randn(3, requires_grad=True)
print(x)

x.requires_grad_(False) # modified in place
print(x)

y = x.detach() # detach the tensor from the graph, new tensor no gradients required
print(y)

with torch.no_grad():
    y = x + 2
    print(y)



# example training
weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
    model_output = (weights*3).sum()

    # calculate the gradients
    model_output.backward()

    # update the weights
    # weights.data += 0.01*weights.grad.data

    # clear the gradients
    # weights.grad.data.zero_()

    print(weights.grad)

    weights.grad.zero_()

    # optimize before next iteration
    optimizer = torch.optim.SGD([weights], lr=0.01)
    optimizer.step()
    optimizer.zero_grad()
