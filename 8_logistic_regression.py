# same old training pipeline
# model -> loss -> optimizer
#   1. design the model (input size, output size, forward pass and layers)
#   2. design the loss function and the optimizer
#   3. training loop
#    - forward pass: compute prediction
#    - backward pass: compute the gradient
#    - update the weights
#   4. evaluate the model
#


import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets # load a binary classification dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 0. prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
print(f'Number of samples: {n_samples}, Number of features: {n_features}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale features to have zero mean and unit variance, 
# to avoid numerical instability in the gradient descent algorithm 
# and to make the model more robust to the data 
# (it is not a good idea to scale the data to have zero mean and unit variance)
# recommended when dealing with logistic regression
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# 1. model
# f = wx + b (linear regression), w = weight, b = bias; then we applay a sigmoid function at the end
class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted


model = LogisticRegression(n_features)


# 2. loss function and the optimizer
learning_rate = 0.01
criterion = nn.BCELoss() # built-in in pytorch, in case of linear regression is median squared error
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 3. training loop
n_epochs = 100
for epoch in range(n_epochs):
    # prediction = forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # gradient = backward pass
    loss.backward()

    # update the weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}: loss = {loss.item():.4f}')


# 4. evaluate the model
with torch.no_grad():   # no need to compute the gradient during evaluation, should not be part of our computational graph
    y_predicted = model(X_test)
    y_predicted_class = y_predicted.round() # (y_predicted > 0.5).float()
    accuracy = y_predicted_class.eq(y_test).sum().item() / float(y_test.shape[0]) # len(y_test)
    print(f'Accuracy: {accuracy*100:.4f}%')