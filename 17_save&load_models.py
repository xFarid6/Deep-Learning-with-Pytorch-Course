import torch
import torch.nn as nn


# can use tensors models or any dictionary as parameter for saving, can use any dict with it
# makes use of python pickle module to save the model and serialize it
# complete model
torch.save(arg, PATH)


# model class must be defined somewhere
model = torch.load(PATH)
model.eval() # lazy option, the serialized data is bound to the specific structure used when saving the model

# State dict, save it and use it later for _inference_
torch.save(model.state_dict(), PATH)

# model must be created again with parameters
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval() 


#### in practice ####

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.linear1 = nn.Linear(n_input_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear1(x))


model = Model(n_input_features=6)
print(model.state_dict())
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(optimizer.state_dict())
# train the model...

# 1st way
FILE = 'model.pth'
torch.save(model, FILE)  

# and
model = torch.load(FILE)
model.eval()
for param in model.parameters():
    print(param)


# 2nd way, preferred
FILE = 'model.pth'
torch.save(model.state_dict(), FILE)

# and
loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()
for param in loaded_model.parameters():
    print(param)


# 3rd way, stop half-way and save a checkpoint
checkpoint = {
    "epoch": 90,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}

torch.save(checkpoint, "checkpoint.pth")

# and
# loaded_model.load_state_dict(loaded_checkpoint["model_state_dict"])
loaded_checkpoint = torch.load("checkpoint.pth")
epoch = loaded_checkpoint["epoch"]

model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)

model.load_state_dict(loaded_checkpoint["model_state_dict"])
optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
# from here can continue training

print(optimizer.state_dict())   # as you can see the learning rate is the same as the saved one
