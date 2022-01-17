import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import json
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from pathlib import Path as P
import os
from utils import *
from loader import *
import warnings
warnings.filterwarnings("ignore")

# Configuration and hyper-parameters
parser = argparse.ArgumentParser()  # _new modify the learning rate and Adam betas
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--z_dim', type=int, default=100)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--display_step', type=int, default=5)
parser.add_argument('--d_lr', type=float, default=2e-4)  # lr = 2e-4 / 1e-5 for ori with mnist
parser.add_argument('--g_lr', type=float, default=1e-4)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint_dcgan')
parser.add_argument('--checkpoint_path', type=str, default=None)
parser.add_argument('--model', type=str, default='dcgan')
parser.add_argument('--dataset', type=str, default='ANIME-face')
parser.add_argument('--seed', type=int, default=28)
parser.add_argument('--device', type=int, default=7)

args = parser.parse_args()

# Seed for reproduction
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size, dataset, z_dim, model_type = args.batch_size, args.dataset, args.z_dim, args.model

checkpoint_dir = P(args.checkpoint_dir)
checkpoint_dir.mkdir(exist_ok=True)
config_path = checkpoint_dir / 'config.json'
with config_path.open('w') as f:
    json.dump(vars(args), f, indent=4, sort_keys=True)
args.channel = 3
args.Size = 64
print(args)

# Dataloader for training and validation
loader, val_loader = get_dataset(args)

# Models
discriminator, generator = get_model(args, device)

# Optimizer
optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.d_lr, betas=(0.5, 0.999))
optim_gen = optim.Adam(generator.parameters(), lr=args.g_lr, betas=(0.5, 0.999))  # betas (0.0, 0.9) / (0.5, 0.999)

# Loss function
criterion = nn.BCEWithLogitsLoss()

if __name__ == '__main__':
    # Loading state_dicts
    global_epoch = 1  # Start from epoch one
    if args.checkpoint_path is not None:
        global_epoch = load_checkpoint(args.checkpoint_path, discriminator, generator, optim_disc, optim_gen, 0)

    # Training process
    start_trainer(model_type, loader, val_loader, discriminator, generator, optim_disc, optim_gen, criterion, args,
                  device, global_epoch)
