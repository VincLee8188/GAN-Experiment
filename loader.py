import torch
from torch.utils.data import DataLoader, Dataset
from models import *
from torchvision import datasets, transforms
from PIL import Image
import os

base_dir = '/data2/huqw/cvhsg/'
face_train_path = base_dir + 'ANIME-face/train/'
face_validation_path = base_dir + 'ANIME-face/validation/'

def eachFile(filepath):                 
    pathDir =  os.listdir(filepath)
    out = []
    for allDir in pathDir:
        out.append(allDir)
    return out

class MyDataset(Dataset):
    def __init__(self, path, transform = None):
        pic_set = eachFile(path)  
        X = []
        for pic_name in pic_set:
            img = Image.open(os.path.join(path,pic_name))
            X.append(img.copy())
            img.close()
        self.imgs = X                        
        self.transform = transform
 
    def __getitem__(self, index):
        img = self.imgs[index]   
        if self.transform is not None:
            img = self.transform(img) 
        return img              
 
    def __len__(self):
        return len(self.imgs)

def get_dataset(args):
    model_type, dataset, channel, batch_size = args.model, args.dataset, args.channel, args.batch_size
    transform = transforms.Compose([
        transforms.Resize(args.Size),
        transforms.CenterCrop(args.Size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * channel, (0.5,) * channel),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if dataset == 'ANIME-face':
        loader = DataLoader(
            MyDataset(face_train_path, transform=transform),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            MyDataset(face_validation_path, transform=transform), batch_size=batch_size
        )
    else:
        raise ValueError(f'Wrong dataset type :{dataset}')
    return loader, val_loader


def get_model(args, device):
    model_type = args.model
    if model_type == 'dcgan':
        discriminator = dcgan.get_disc_model(args).to(device)
        generator = dcgan.get_gen_model(args).to(device)
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
    if model_type == 'dcgan':
        dcgan.train(loader, val_loader, discriminator, generator, optim_disc, optim_gen, criterion, args, device,
                    global_epoch)
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
