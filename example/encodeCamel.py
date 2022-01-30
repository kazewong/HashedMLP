from matplotlib import scale
from model.HashedMLP import HashedMLP
import torch
import torch.nn as nn
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler

n_input = 3
n_output = 1
n_hidden = 64
n_layers = 2
n_entries = 2**19
n_feature = 2
base_grids = torch.tensor([16,16,16]).cuda()
n_level = 9
n_factor = 1.5
n_auxin = 0
act = nn.ReLU()
model = HashedMLP(n_input, n_output, n_hidden, n_layers, n_entries, n_feature, base_grids, n_level, n_factor, n_auxin, act)
position = torch.rand(100,3).cuda()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model = nn.DataParallel(model)

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.99), eps=1e-8,weight_decay=1e-4)
batch_size = 1000000
n_epoch = 1000
n_check = 50
loss = nn.MSELoss()

stride = 1
data = np.load('./data/camel_compression_512.npz')['mass'][::stride,::stride,::stride][...,None]
#data = torch.log10(1+torch.tensor(data)).float()
data = torch.tensor(data).float()

original_shape = data.shape
coord = torch.stack(torch.meshgrid(torch.arange(data.shape[0]),torch.arange(data.shape[1]),torch.arange(data.shape[2])),-1).float()
coord[...,0] = coord[...,0]/data.shape[0]
coord[...,1] = coord[...,1]/data.shape[1]
coord[...,2] = coord[...,2]/data.shape[2]
coord = torch.flatten(coord,end_dim=-2)
data = torch.flatten(data,end_dim=-2)

data = torch.log10(1+data)
scaler = StandardScaler()
scaler.fit(data.detach().cpu().numpy())
data = torch.tensor(scaler.transform(data.detach().cpu().numpy())).float()*255
best_loss = 1e10

#weight = copy.deepcopy(model.interpolator[0].hash_table.detach().cpu().numpy())
#print(model.interpolator[0].hash_table.is_leaf)
for epoch in range(n_epoch):
    running_loss = 0.0
    for i in range(0,data.shape[0],batch_size):
        optimizer.zero_grad()
        output = model(coord[i:i+batch_size])
        loss_value = loss(output,data[i:i+batch_size].to(device))
        loss_value.backward()
        optimizer.step()
        running_loss += loss_value.item()
        if i == int(data.shape[0]/batch_size)*batch_size:
            print('loss: %.3f' % (running_loss*batch_size/data.shape[0]))
        if running_loss < best_loss:
            best_loss = running_loss
            torch.save(model.module.state_dict(), './checkpoint/camel_compression_512_best.pth')


pred = []
for i in range(0,data.shape[0],batch_size):
    with torch.no_grad():
        output = model(coord[i:i+batch_size].to(device))
        pred.append(output.cpu().numpy())
pred = np.concatenate(pred,axis=0).reshape(original_shape)
