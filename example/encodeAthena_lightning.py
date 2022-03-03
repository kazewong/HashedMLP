
from model.HashedMLP import HashedMLP_lightning
from dataset.GridDataset import GridDataModule
from argparse import ArgumentParser
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

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

model = HashedMLP_lightning(args)
model.setup()
loader = GridDataModule("/mnt/ceph/users/wwong/Simulations/AstroSimChallenge/Athena/mixing_layer.hdf5",["rho"],stride=stride,num_workers=8)
logger = TensorBoardLogger(save_dir='tb_logs/', name='mixing_layer')
checkpoint_callback = ModelCheckpoint(monitor='val_loss',save_top_k=3,mode='min')
# trainer = pl.Trainer(gpus=4,accelerator='ddp',max_epochs=3000,gradient_clip_val=1.0,logger=logger,callbacks=[checkpoint_callback],check_val_every_n_epoch=5,progress_bar_refresh_rate=1)
# trainer.fit(model,loader)

