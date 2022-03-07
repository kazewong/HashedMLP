from model.models import MLP, MultigridInterpolator
from dataset.GridDataset import GridDataset
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

class HashedMLP(nn.Module):

    def __init__(self, n_input: int, n_output: int, n_hidden: int, n_layers: int,
                    n_entries: int, n_feature: int, base_grids: torch.tensor, 
                    n_level: int, n_factor: float, n_auxin:int=0, act = nn.ReLU()):
        super().__init__()

        self.feature_table = MultigridInterpolator(n_level, n_entries, base_grids, finest_grids,features=n_feature)
        self.MLP = MLP(n_feature*n_level, n_output, n_hidden, n_layers, act)

    def forward(self,data):
        x = self.feature_table(data)
        x = self.MLP(x)
        return x


n_input = 3
n_output = 1
n_hidden = 64
n_layers = 2
n_entries = 2**19
n_feature = 2
base_grids = torch.tensor([16,16,16]).cuda()
finest_grids = torch.tensor([1024,1024,1024]).cuda()
n_level = 13
n_factor = 1.5
n_auxin = 0
act = nn.GELU()
model = HashedMLP(n_input, n_output, n_hidden, n_layers, n_entries, n_feature, base_grids, n_level, n_factor, n_auxin, act)
position = torch.rand(100,3).cuda()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model = nn.DataParallel(model)

learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.99), eps=1e-8,weight_decay=1e-4)
batch_size = 1000000
n_epoch = 30000
verbose_epoch = 5
loss = nn.MSELoss()
#loss = nn.DataParallel(loss)
tag = "turbulence_velocity_2048"
writer = SummaryWriter("./runs/"+tag)

print("Loading dataset...")

stride = 4
dataset = GridDataset("/mnt/ceph/users/wwong/Simulations/AstroSimChallenge/Athena/mixing_layer.hdf5",["rho"],stride=stride)
#dataset = GridDataset("/mnt/ceph/users/wwong/Simulations/AstroSimChallenge/Athena/plasmoid_static.hdf5",["vel1","vel2","vel3"],stride=stride)

print("Shuffling dataset...")

total_index = np.arange(dataset.__len__())
#np.random.shuffle(total_index)
train_index = total_index[:int(total_index.__len__()*0.8)]
val_index = total_index[int(total_index.__len__()*0.8):]

train_data = dataset[train_index]
val_data = dataset[val_index]

train_length = train_data[0].shape[0]
val_length = val_data[0].shape[0]

# train_data, val_data = random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])

best_train_loss = 1e10
best_val_loss = 1e10

print("Start training")
for epoch in range(n_epoch):
    print("Epoch: ",epoch)
    model.train()
    running_loss = 0.0
    for i in tqdm(range(0,train_length,batch_size)):
        optimizer.zero_grad()
        coord = train_data[0][i:i+batch_size]
        data = train_data[1][i:i+batch_size]
        output = model(coord)
        loss_value = loss(output,data.to(device))
        loss_value.backward()
        optimizer.step()
        running_loss += loss_value.item()
        if i == int(train_length/batch_size)*batch_size:
            if epoch % verbose_epoch == 0:
                print('training loss: %.3f' % (running_loss*batch_size/train_length))
            torch.save(model.module.state_dict(), './checkpoint/Athena_'+tag+'_e'+str(int(np.log2(n_entries)))+'_l'+str(n_level)+'_train.pth')
            writer.add_scalar('loss/train', running_loss/train_length, i)
            running_loss = 0.0
    if epoch%2 == 0:
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i in tqdm(range(0,val_length,batch_size)):
                coord = val_data[0][i:i+batch_size]
                data = val_data[1][i:i+batch_size]
                output = model(coord)
                loss_value = loss(output,data.to(device))
                running_loss += loss_value.item()
                if i == int(val_length/batch_size)*batch_size:
                    if epoch % verbose_epoch == 0:
                        print('validation loss: %.3f' % (running_loss*batch_size/val_length))
                    torch.save(model.module.state_dict(), './checkpoint/Athena_'+tag+'_e'+str(int(np.log2(n_entries)))+'_l'+str(n_level)+'_val.pth')
                    writer.add_scalar('loss/val', running_loss/val_length, i)
                    running_loss = 0.0

# pred = []
# for i in range(0,data.shape[0],batch_size):
#     with torch.no_grad():
#         output = model(coord[i:i+batch_size].to(device))
#         pred.append(output.cpu().numpy())
# pred = np.concatenate(pred,axis=0).reshape(original_shape)

