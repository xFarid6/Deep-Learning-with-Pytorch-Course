# replacing the gradient
import torch

# linear regression
# y = w*x + b
# f = 2 * x <- our formula

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32) # times 2

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return x * w


# loss function = MSE (mean squared error)
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()


print(f'Prediction before training: forward(5) = {forward(5):.3f}')

# training
learning_rate = 0.01
n_iters = 100 # 20 # 10

for epoch in range(n_iters):
    # prediction = forward pass
    y_predicted = forward(X)

    # loss = MSE
    l = loss(Y, y_predicted)

    # gradient = backward pass
    l.backward() # will calculate the gradients of l with respect to w, dl/dw
    # the backpropagation algorithm is not as exact as the numerical gradient computation

    # update the weights, this should not be part of our computational graph
    with torch.no_grad():
        w -= learning_rate * w.grad

    # zero gradients, so that we can use them again
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'Epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')


print(f'Prediction after training: forward(5) = {forward(5):.3f}')