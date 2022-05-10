'''
Softmax = S(yi) = e^yi / sum(e^yi)
exponential to all the elements
squashes the output to be between 0 and 1

Linear -> scores / logits, Probabilities = exp(scores) / sum(exp(scores)), y predictions
'''

import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))  # prevent overflow
    return e_x / np.sum(e_x)
    # return np.exp(x) / np.sum(np.exp(x), axis=0) # sum along axis 0


x = np.array([-1, -0.5, 0, 1, 2])
output = softmax(x)
print('softmax(x) =', output)

x = torch.tensor([-1, -0.5, 0, 1, 2])
output = torch.softmax(x, dim=0)
print('torch.softmax(x, dim=0) =', output)

'''
Cross-Entrpy = -sum(yi * log(S(yi)))

    yi = 1 if yi = yi_hat
    yi = 0 if yi != yi_hat

    yi_hat = argmax(S(yi))

    yi_hat = 1 if yi = 1
    yi_hat = 0 if yi != 1

    log(S(yi)) = log(S(yi_hat))

    log(S(yi_hat)) = 0 if yi_hat = 1
    log(S(yi_hat)) = -inf if yi_hat = 0
    
cross entropy is a measure of the difference between two distributions
    - the distribution of the true labels
    - the distribution of the predicted labels

you usually calculate in on the softmax output
'''

def cross_entropy(actual, predicted):
    return -np.sum(actual * np.log(predicted))


# y must be one hot encoded
# if class 0: [1, 0, 0]
# if class 1: [0, 1, 0]
# if class 2: [0, 0, 1]
Y = np.array([1, 0, 0])

# y_pred has probabilities for each class
y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.2, 0.7])
l1 = cross_entropy(Y, y_pred_good)
l2 = cross_entropy(Y, y_pred_bad)
print('Loss1 numpy =', l1)
print('Loss2 numpy =', l2)


'''
now do it with pytorch
nn.CrossEntropyLoss

Careful!
nn.CrossEntropyLoss() expects the input to be a tensor of size NxC, where N is the batch size and C is the number of classes. 
If you have a minibatch of size 64 and 2 classes, your input should be a tensor of size 64x2. 
If you have a minibatch of size 64 and 10 classes, your input should be a tensor of size 64x10. 

nn.CrossEntropyLoss() applies 
    - the softmax function to the input
    - the cross entropy function to the output
    - the mean function to the loss

nn.CrossEntropyLoss applies
nn.LogSoftmax + nn.NLLLoss (log softmax + negative log likelihood)

_ No Softmax in the last layer! _

Y has class labels, not One.Hot!
Y_pred has raw scores (logits), no Softmax!
'''

loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])
# size = # samples * # classes = 1x3
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])   # logits, the class 0 has the highest value -> its a good prediction
Y_pred_bad = torch.tensor([[0.1, 1.0, 2.0]])   # logits, the class 2 has the highest value -> its a bad prediction

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print('Loss1 torch =', l1)
print('Loss2 torch =', l2)

print(l1.item())
print(l2.item())

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print('Predictions1 =', predictions1)
print('Predictions2 =', predictions2)   # we choose the class with the highest score

# example with 3 samples and 3 classes
Y = torch.tensor([2, 0, 1])

# n samples * n classes = 3x3
Y_pred_good = torch.tensor([[2.0, 1.0, 2.1], [2.0, 1.0, 0.1], [2.0, 3.0, 0.1]])
Y_pred_bad = torch.tensor([[0.1, 1.0, 2.1], [0.1, 1.0, 0.1], [0.1, 3.0, 0.1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(l1.item())
print(l2.item())

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print('Predictions1 =', predictions1)
print('Predictions2 =', predictions2)

# in pytorch use nn.CrossEntropyLoss()
# No Softmax at the end!