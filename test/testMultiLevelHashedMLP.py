from model.HashedMLP import HashedMLP
import torch
import torch.nn as nn

n_input = 3
n_output = 2
n_hidden = 64
n_layers = 2
n_entries = 2**16
n_feature = 2
base_grids = torch.tensor([16,16,16]).cuda()
n_level = 3
n_factor = 2
n_auxin = 0
act = nn.ReLU()
model = HashedMLP(n_input, n_output, n_hidden, n_layers, n_entries, n_feature, base_grids, n_level, n_factor, n_auxin, act)
position = torch.rand(100,3).cuda()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

print(model(position))