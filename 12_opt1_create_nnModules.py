import torch
import torch.nn as nn
import torch.nn.functional as F


# option 1 (create nn modules)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        # nn.Sigmoid() is already included in Pytorch
        # nn.Softmax() is already included in Pytorch
        # nn.LogSoftmax() is already included in Pytorch
        # nn.Tanh() is already included in Pytorch
        # nn.leaky_relu() is already included in Pytorch
        self.linear2 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sigmoid at the end
        y_pred = self.sigmoid(out)
        return y_pred


# option 2 (use activation functions directly in forward pass)
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # can also do:
        # F.leaky_relu()
        # F.relu()
        # F.sigmoid() 
        # F.softmax() etc..
        out = torch.relu(input=self.linear1(x))
        out = torch.sigmoid(input=self.linear2(out))
        # torch.softmax
        # torch.log_softmax
        return out
