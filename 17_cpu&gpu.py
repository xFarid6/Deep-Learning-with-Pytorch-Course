import torch
import torch.nn as nn

# Save on GPU load on CPU
device = torch.device('cuda')
model.to(device)
torch.save(model.state_dict(), PATH)

# and
device = torch.device('cpu')
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))



# Save on GPU and load on GPU
device = torch.device('cuda')
model.to(device)
torch.save(model.state_dict(), PATH)

# and
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)



# Save on CPU and load on GPU
torch.save(model.state_dict(), PATH)

# and
device = torch.device('cuda')
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0")) # choose whatever GPu you want to load the model on
model.to(device)