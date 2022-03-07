from model.models import MLP, MultigridInterpolator
from dataset.GridDataset import GridDataModule
from argparse import ArgumentParser
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl


class HashedMLP_lightning(pl.LightningModule):
    
    @staticmethod
    def specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--n_input', type=int)
        parser.add_argument('--n_output', type=int)
        parser.add_argument('--learning_rate', type=float)
        parser.add_argument('--weight_decay', type=float,default=1e-4)
        parser.add_argument('--step_size', type=float, default=1)
        parser.add_argument('--gamma', type=float, default=0.5)
        parser.add_argument('--n_hidden', type=int, default=64)
        parser.add_argument('--n_layers', type=int, default=2)
        parser.add_argument('--n_entries', type=int, default=2**20)
        parser.add_argument('--n_feature', type=int, default=2)
        parser.add_argument('--base_grids', type=torch.tensor, default= torch.tensor([16,16,16]))
        parser.add_argument('--finest_grids', type=torch.tensor, default= torch.tensor([512,512,512]))
        parser.add_argument('--n_level', type=int, default=16)
        parser.add_argument('--n_factor', type=float, default=1.5)
        parser.add_argument('--n_auxin', type=int, default=0)
        return parser

    def __init__(self,args):
        super().__init__()

        self.feature_table = MultigridInterpolator(args.n_level, args.n_entries, args.base_grids, args.finest_grids,features=args.n_feature)
        self.MLP = MLP(args.n_feature*args.n_level, args.n_output, args.n_hidden, args.n_layers, nn.GELU())
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.step_size = args.step_size
        self.gamma = args.gamma
        self.loss = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self,data):
        x = self.feature_table(data)
        x = self.MLP(x)
        return x

    def training_step(self,batch,batch_idx):
        tb = self.logger.experiment
        x,y = batch
        output = self(x)
        loss = self.loss(output,y)
        self.log('loss',loss)
        return loss

    def validation_step(self,batch,batch_idx):
        x,y = batch
        output = self(x)
        val_loss = self.loss(output,y)
        self.log('val_loss',val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizers = torch.optim.Adam(list(self.feature_table.parameters())+list(self.MLP.parameters()), lr=self.learning_rate, betas=(0.9,0.99), eps=1e-8,weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizers, step_size=self.step_size, gamma=self.gamma)
        return [optimizers],[scheduler]

parser = ArgumentParser()
parser = HashedMLP_lightning.specific_args(parser)
#parser.add_argument('--filename', type=str,help='Data file path',required=True)
args = parser.parse_args()

args.n_input = 3
args.n_output = 3
args.learning_rate = 1e-4
args.n_hidden = 64
args.n_layers = 2
args.n_entries = 2**24
args.n_feature = 2
args.base_grids = torch.tensor([16,16,16])
args.n_level = 16
args.n_factor = 1.5
args.n_auxin = 0


stride = 1
batch_size = int(2**20)

model = HashedMLP_lightning(args)
model.setup()
loader = GridDataModule("/mnt/ceph/users/wwong/Simulations/AstroSimChallenge/Athena/mixing_layer.hdf5",["rho"],stride=stride,batch_size=batch_size)
logger = TensorBoardLogger(save_dir='tb_logs/', name='mixing_layer')
checkpoint_callback = ModelCheckpoint(monitor='val_loss',save_top_k=3,mode='min')
trainer = pl.Trainer(gpus=4,accelerator='ddp',max_epochs=3000,gradient_clip_val=1.0,logger=logger,callbacks=[checkpoint_callback],check_val_every_n_epoch=5,progress_bar_refresh_rate=1)
trainer.fit(model,loader)

