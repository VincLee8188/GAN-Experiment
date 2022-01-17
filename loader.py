import torch
from torch.utils.data import DataLoader
from models import *
from torchvision import datasets, transforms


def get_dataset(args):
    model_type, dataset, channel, batch_size = args.model, args.dataset, args.channel, args.batch_size
    if model_type == 'gan':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * channel, (0.5,) * channel),
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
    return loader, val_loader


def get_model(args, device):
    model_type = args.model
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
    elif model_type == 'dcgan_sa':
        discriminator = dcgan_sa.get_disc_model(args).to(device)
        generator = dcgan_sa.get_gen_model(args).to(device)
    else:
        raise ValueError(f'Wrong models type :{model_type}')
    return discriminator, generator


def start_trainer(model_type, loader, val_loader, discriminator, generator, optim_disc, optim_gen, criterion, args,
                  device, global_epoch):
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
    elif model_type == 'dcgan_sa':
        dcgan_sa.train(loader, val_loader, discriminator, generator, optim_disc, optim_gen, criterion, args, device,
                       global_epoch)
    else:
        raise ValueError(f'Wrong models type :{model_type}')
