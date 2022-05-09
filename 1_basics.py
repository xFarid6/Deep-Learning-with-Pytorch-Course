import torch
import numpy as np


x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.ones(2, 2, dtype=torch.int) # double, float16, int, long
print(x)
print(x.dtype)
print(x.size())

x = torch.tensor([5, 3, 2.5, 0.1])
print(x)

# operations
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print("Sum: ", x + y)
print("Subtraction: ", x - y)
print("Multiplication: ", x * y)
print("Division: ", x / y)
print("Exponent: ", x ** y)
print("Floor Division: ", x // y)
print("Modulo: ", x % y)
# print("Product: ", x.dot(y)) # 1D tensor only
print("Min: ", x.min())
print("Max: ", x.max())
print("Mean: ", x.mean())
print("Median: ", x.median())
print("Standard Deviation: ", x.std())
print("Variance: ", x.var())
print("Argmin: ", x.argmin())
print("Argmax: ", x.argmax())
print("Argmax: ", x.argmax(dim=0))
print("Argmax: ", x.argmax(dim=1))
print("Argmax: ", x.argmax(dim=1, keepdim=True))
print("Argmax: ", x.argmax(dim=1, keepdim=False))

# z = torch.add_(x, y) # _ is for inplace
# print(z)

x = torch.rand(5, 3)
print("First row, all elements: ", x[0, :])
print("First row, first element: ", x[0, 0])
print("First row, first element: ", x[0, 0].item()) # item() to get the value
print("First row, first element: ", x[0, 0].type())
print("First row, first element: ", x[0, 0].dtype)
print("First row, first element: ", x[0, 0].shape)
print("First row, first element: ", x[0, 0].size())
print("First row, first element: ", x[0, 0].dim())
print("First row, first element: ", x[0, 0].numel())

# reshaping
x = torch.rand(5, 5)
print(x)
print(x.reshape(25))
print(x.reshape(5, 5))
print(x.reshape(5, 5).shape)
y = x.view(-1, 5)
print(y)
print(y.size())


# from numpy to tensor and vice-versa
x = np.random.rand(5, 3)
print(x)
print(torch.from_numpy(x))
print(torch.from_numpy(x).dtype)
print(torch.from_numpy(x).shape)
print(torch.from_numpy(x).size())
print(torch.from_numpy(x).dim())


a = torch.ones(5)
print(a)
b = a.numpy()

a.add_(1)
print(a)
print(b)

a = np.ones(5)
print(a)
b = torch.from_numpy(a, dtype=torch.int)
print(b)

a += 1
print(a)
print(b)


# see if cuda is available
print(torch.cuda.is_available())

# requires grad is for autograd
x = torch.ones(5, requires_grad=True)
print(x)