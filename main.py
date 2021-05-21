import argparse
import torch
from models import *
import numpy as np
import random
import json
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path as P
import os
from utils import *
import warnings
warnings.filterwarnings("ignore")

# Configuration and hyper-parameters
parser = argparse.ArgumentParser()  # _new modify the learning rate and Adam betas
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--z_dim', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--display_step', type=int, default=5)
parser.add_argument('--d_lr', type=float, default=2e-4)  # lr = 2e-4 / 1e-5 for ori with mnist
parser.add_argument('--g_lr', type=float, default=2e-4)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
parser.add_argument('--checkpoint_path', type=str, default=None)
parser.add_argument('--model', type=str, default='gan')
parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--seed', type=int, default=28)
parser.add_argument('--device', type=int, default=7)

args = parser.parse_args()
print(args)

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

args.channel = 1 if dataset == 'MNIST' else 3 if dataset == 'CIFAR10' else None

if model_type == 'gan':
    transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
else:
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*args.channel, (0.5,)*args.channel),
    ])

if dataset == 'MNIST':
    loader = DataLoader(
        datasets.MNIST('./dataset', download=False, transform=transform, train=True),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        datasets.MNIST('./dataset', download=False, transform=transform, train=False), batch_size=batch_size
    )
elif dataset == 'CIFAR10':
    loader = DataLoader(
        datasets.CIFAR10('./dataset', download=False, transform=transform, train=True),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        datasets.CIFAR10('./dataset', download=False, transform=transform, train=False), batch_size=batch_size
    )
else:
    raise ValueError(f'Wrong dataset type :{dataset}')

if model_type == 'gan':
    discriminator = gan.get_disc_model(args).to(device)
    generator = gan.get_gen_model(args).to(device)
elif model_type == 'dcgan':
    discriminator = dcgan.get_disc_model(args).to(device)
    generator = dcgan.get_gen_model(args).to(device)
elif model_type == 'dcgan_wl':
    discriminator = dcgan_wl.get_disc_model(args).to(device)
    generator = dcgan_wl.get_gen_model(args).to(device)
elif model_type == 'dcgan_sn':
    discriminator = dcgan_sn.get_disc_model(args).to(device)
    generator = dcgan_sn.get_gen_model(args).to(device)
elif model_type == 'dcgan_un':
    discriminator = dcgan_un.get_disc_model(args).to(device)
    generator = dcgan_un.get_gen_model(args).to(device)
elif model_type == 'dcgan_un_1':
    discriminator = dcgan_un_1.get_disc_model(args).to(device)
    generator = dcgan_un_1.get_gen_model(args).to(device)
elif model_type == 'dcgan_sa':
    discriminator = dcgan_sa.get_disc_model(args).to(device)
    generator = dcgan_sa.get_gen_model(args).to(device)
else:
    raise ValueError(f'Wrong models type :{model_type}')

# Optimizer
optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.d_lr, betas=(0.5, 0.999))
optim_gen = optim.Adam(generator.parameters(), lr=args.g_lr, betas=(0.5, 0.999))  # betas (0.0, 0.9) / (0.5, 0.999)

# Loss function
criterion = nn.BCEWithLogitsLoss()

if __name__ == '__main__':
    # Loading state_dicts
    global_epoch = 1
    if args.checkpoint_path is not None:
        global_epoch = load_checkpoint(args.checkpoint_path, discriminator, generator, optim_disc, optim_gen, 0)

    # Training process
    if model_type == 'gan':
        gan.train(loader, val_loader, discriminator, generator, optim_disc, optim_gen, criterion, args, device,
                  global_epoch)
    elif model_type == 'dcgan':
        dcgan.train(loader, val_loader, discriminator, generator, optim_disc, optim_gen, criterion, args, device,
                    global_epoch)
    elif model_type == 'dcgan_wl':
        args.c_lambda = 10
        args.crit_repeats = 5
        dcgan_wl.train(loader, val_loader, discriminator, generator, optim_disc, optim_gen, args, device, global_epoch)
    elif model_type == 'dcgan_sn':
        dcgan_sn.train(loader, val_loader, discriminator, generator, optim_disc, optim_gen, criterion, args, device,
                       global_epoch)
    elif model_type == 'dcgan_un':
        args.unrolled_steps = 10
        dcgan_un.train(loader, val_loader, discriminator, generator, optim_disc, optim_gen, criterion, args, device,
                       global_epoch)
    elif model_type == 'dcgan_un_1':
        args.unrolled_steps = 10
        dcgan_un_1.train(loader, val_loader, discriminator, generator, optim_disc, optim_gen, criterion, args, device,
                         global_epoch)
    elif model_type == 'dcgan_sa':
        dcgan_sa.train(loader, val_loader, discriminator, generator, optim_disc, optim_gen, criterion, args, device,
                       global_epoch)
    else:
        raise ValueError(f'Wrong models type :{model_type}')

    # Testing
