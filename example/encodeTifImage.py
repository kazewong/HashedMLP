from model.HashedMLP import HashedMLP
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import copy
Image.MAX_IMAGE_PIXELS = 933120000

n_input = 2
n_output = 3
n_hidden = 64
n_layers = 2
n_entries = 2**16
n_feature = 2
base_grids = torch.tensor([16,16]).cuda()
n_level = 11
n_factor = 2
n_auxin = 0
act = nn.ReLU()
model = HashedMLP(n_input, n_output, n_hidden, n_layers, n_entries, n_feature, base_grids, n_level, n_factor, n_auxin, act)
position = torch.rand(100,3).cuda()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.99), eps=1e-15)
batch_size = 1000000
n_epoch = 100
loss = nn.MSELoss()

data = torch.tensor(np.array(Image.open('/home/kaze/Work/HashedMLP/data/eso1625a.tif'))).float()
coord = torch.stack(torch.meshgrid(torch.arange(data.shape[0]),torch.arange(data.shape[1])),-1).float()
coord[...,0] = coord[...,0]/data.shape[0]
coord[...,1] = coord[...,1]/data.shape[1]
coord = torch.flatten(coord,end_dim=-2)
data = torch.flatten(data,end_dim=-2)

weight = copy.deepcopy(model.interpolator[0].hash_table.detach().cpu().numpy())
print(model.interpolator[0].hash_table.is_leaf)
for epoch in range(n_epoch):
    for i in range(0,data.shape[0],batch_size):
        optimizer.zero_grad()
        output = model(coord[i:i+batch_size].to(device))
        loss_value = loss(output,data[i:i+batch_size].to(device))
        loss_value.backward()
        optimizer.step()
    print(loss_value)