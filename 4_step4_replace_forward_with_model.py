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

# these are 2d arrays now, the # of rows is the # of samples
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32) # times 2

X_test = torch.tensor([[5]], dtype=torch.float32)

n_samples, n_features = X.shape
print("n_samples:", n_samples, "n_features:", n_features)
input_size = n_features
output_size = n_features

# model = nn.Linear(in_features=input_size, out_features=output_size)

# if we were to want a custom model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        # define layers
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


model = LinearRegression(input_size, output_size)

print(f'Prediction before training: forward(5) = {model(X_test).item():.3f}')

# training
learning_rate = 0.01
n_iters = 100 # 20 # 10

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # stacasting (?) gradient descent

for epoch in range(n_iters):
    # prediction = forward pass
    y_predicted = model(X)

    # loss = MSE
    l = loss(Y, y_predicted)

    # gradient = backward pass
    l.backward() 

    # update the weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'Epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')


print(f'Prediction after training: forward(5) = {model(X_test).item():.3f}')