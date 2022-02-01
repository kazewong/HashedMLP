from matplotlib import scale
from model.HashedMLP import HashedMLP
from dataset.GridDataset import GridDataset
import torch
import torch.nn as nn
import numpy as np
import h5py
import copy
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader,random_split

n_input = 3
n_output = 5
n_hidden = 64
n_layers = 2
n_entries = 2**19
n_feature = 2
base_grids = torch.tensor([16,16,16]).cuda()
n_level = 16
n_factor = 1.5
n_auxin = 0
act = nn.ReLU()
model = HashedMLP(n_input, n_output, n_hidden, n_layers, n_entries, n_feature, base_grids, n_level, n_factor, n_auxin, act)
position = torch.rand(100,3).cuda()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model = nn.DataParallel(model)

learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.99), eps=1e-8,weight_decay=1e-4)
batch_size = 1000000
n_epoch = 30000
verbose_epoch = 10
loss = nn.MSELoss()
writer = SummaryWriter("./runs/mixing_layer")

stride = 4
dataset = GridDataset("/mnt/ceph/users/wwong/Simulations/AstroSimChallenge/Athena/mixing_layer.hdf5",["rho","press","vel1","vel2","vel3"],stride=stride)
train_data, val_data = random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])

best_train_loss = 1e10
best_val_loss = 1e10


def train(epoch):
    model.train()
    running_loss = 0.0
    for i in range(0,train_data.__len__(),batch_size):
        optimizer.zero_grad()
        coord, data = train_data[i:i+batch_size]
        output = model(coord)
        loss_value = loss(output,data.to(device))
        loss_value.backward()
        optimizer.step()
        running_loss += loss_value.item()
        if i == int(train_data.__len__()/batch_size)*batch_size:
            if epoch % verbose_epoch == 0:
                print('training loss: %.3f' % (running_loss*batch_size/data.shape[0]))
            torch.save(model.module.state_dict(), './checkpoint/Athena_mixing_layer_all_best_train.pth')
            writer.add_scalar('loss/train', running_loss*batch_size/data.shape[0], i)
            running_loss = 0.0

def validate(epoch):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i in range(0,val_data.__len__(),batch_size):
            coord, data = val_data[i:i+batch_size]
            output = model(coord)
            loss_value = loss(output,data.to(device))
            running_loss += loss_value.item()
            if i == int(val_data.__len__()/batch_size)*batch_size:
                if epoch % verbose_epoch == 0:
                    print('validation loss: %.3f' % (running_loss*batch_size/data.shape[0]))
                torch.save(model.module.state_dict(), './checkpoint/Athena_mixing_layer_all_best_val.pth')
                writer.add_scalar('loss/val', running_loss*batch_size/data.shape[0], i)
                running_loss = 0.0

print("Start training")
for epoch in range(n_epoch):
    train(epoch)
    if epoch%2 == 0:
        validate(epoch)

# pred = []
# for i in range(0,data.shape[0],batch_size):
#     with torch.no_grad():
#         output = model(coord[i:i+batch_size].to(device))
#         pred.append(output.cpu().numpy())
# pred = np.concatenate(pred,axis=0).reshape(original_shape)
