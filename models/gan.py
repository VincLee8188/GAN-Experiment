import torch
from torch import nn
from tqdm.auto import tqdm
from pathlib import Path as P
from utils import save_checkpoint
from evaluation import *
import numpy as np
import json
from torchvision.utils import make_grid, save_image


def save_tensor_images(image_tensor, checkpoint_dir, epoch, num_images=16, size=(3, 32, 32), prefix='train', label='real'):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    folder = P(checkpoint_dir) / 'samples_epoch{:05d}'.format(epoch)
    folder.mkdir(exist_ok=True)
    for i in range(len(image_unflat) // num_images):
        save_image(make_grid(image_unflat[i*num_images:(i+1)*num_images], nrow=4), folder / f'{prefix}_{i}_{label}.jpg')


def get_generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )


class Generator(nn.Module):
    def __init__(self, z_dim=64, im_dim=1024, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )

    def forward(self, noise):
        return self.gen(noise)


def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)


def get_discriminator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(negative_slope=0.2)
    )


class Discriminator(nn.Module):
    def __init__(self, im_dim=1024, hidden_dim=128, out_dim=1):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, image):
        return self.disc(image)


def get_disc_model(args):
    return Discriminator(im_dim=1024*args.channel, hidden_dim=args.hidden_dim, out_dim=args.channel)


def get_gen_model(args):
    return Generator(z_dim=args.z_dim, im_dim=1024*args.channel, hidden_dim=args.hidden_dim)


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    d_real = disc(real)
    real_truth = torch.ones_like(d_real)
    loss_real = criterion(d_real, real_truth)

    z = get_noise(num_images, z_dim, device)
    g_z = gen(z)
    d_g_z = disc(g_z.detach())
    fake_truth = torch.zeros_like(d_g_z)
    loss_fake = criterion(d_g_z, fake_truth)

    disc_loss = (loss_fake + loss_real) / 2
    return disc_loss


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    z = get_noise(num_images, z_dim, device)
    g_z = gen(z)
    d_g_z = disc(g_z)
    ground_truth = torch.ones_like(d_g_z)
    gen_loss = criterion(d_g_z, ground_truth)
    return gen_loss


def train(dataloader, val_loader, disc, gen, disc_opt, gen_opt, criterion, args, device, ori_epoch):
    # Initialize variables
    n_epochs, z_dim, display_step = args.n_epochs, args.z_dim, args.display_step
    global_epoch = ori_epoch
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0

    # Start iterations
    for cur_epoch in range(args.n_epochs):
        epoch = cur_epoch + global_epoch
        print('\nStarting Epoch: {}'.format(epoch))
        num_batch = len(dataloader)
        for real, _ in tqdm(dataloader):
            # Move data to target machine
            cur_batch_size = len(real)
            real = real.view(cur_batch_size, -1).to(device)

            # Calculate results, conduct gradient backpropagation, and update model parameters
            disc_opt.zero_grad()
            disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
            disc_loss.backward()
            disc_opt.step()

            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
            gen_loss.backward()
            gen_opt.step()

            mean_discriminator_loss += disc_loss.item() / num_batch
            mean_generator_loss += gen_loss.item() / num_batch
            cur_step += 1

        # Print logs
        print(f'Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, '
              f'Discriminator loss:  {mean_discriminator_loss}')

        mean_discriminator_loss = 0
        mean_generator_loss = 0

        # Evaluate model and save checkpoint
        if epoch % display_step == 0:
            with torch.no_grad():
                evaluate(val_loader, epoch, args, gen, disc, criterion, device)
            save_checkpoint(args.checkpoint_dir, disc, gen, disc_opt, gen_opt, epoch)


def evaluate(val_loader, global_epoch, args, gen, disc, criterion, device):
    print('\nStart evaluating')
    z_dim, checkpoint_dir = args.z_dim, args.checkpoint_dir
    running_disc_loss = []
    running_gen_loss = []
    g_feats, gt_feats = [], []

    for real, _ in val_loader:
        cur_batch_size = len(real)
        real = real.view(cur_batch_size, -1).to(device)
        fake_noise = get_noise(cur_batch_size, z_dim, device)
        fake = gen(fake_noise)

        inception = FID.get_eval_model(device)
        img_size = (args.channel, 32, 32)
        g_feat = inception(fake.view(-1, *img_size))[0] if args.channel == 3 else inception(fake.view(-1, *img_size).repeat(1, 3, 1, 1))[0]
        gt_feat = inception(real.view(-1, *img_size))[0] if args.channel == 3 else inception(real.view(-1, *img_size).repeat(1, 3, 1, 1))[0]
        g_feats.append(g_feat.cpu().numpy().reshape(g_feat.size(0), -1))
        gt_feats.append(gt_feat.cpu().numpy().reshape(gt_feat.size(0), -1))

        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
        running_disc_loss.append(disc_loss.item())
        running_gen_loss.append(gen_loss.item())

    g_feats = np.concatenate(g_feats, axis=0)
    gt_feats = np.concatenate(gt_feats, axis=0)
    fid_score = FID.calculate_fid_score(g_feats, gt_feats)

    avg_disc_loss = sum(running_disc_loss) / len(running_disc_loss)
    avg_gen_loss = sum(running_gen_loss) / len(running_gen_loss)

    print(f'Epoch {global_epoch}: Generator loss: {avg_gen_loss}, '
          f'Discriminator loss:  {avg_disc_loss}, FID: {fid_score}')

    save_tensor_images(fake, checkpoint_dir, global_epoch, label='fake', size=(args.channel, 32, 32))
    save_tensor_images(real, checkpoint_dir, global_epoch, size=(args.channel, 32, 32))

    logs_file = P(checkpoint_dir) / 'logs.json'
    if logs_file.is_file():
        with logs_file.open('r') as f:
            logs = json.load(f)
    else:
        logs = {}

    logs[str(global_epoch)] = {
        'fid_score': '{:.4g}'.format(fid_score),
        'disc_loss': '{:.4g}'.format(avg_disc_loss),
        'gen_loss': '{:.4g}'.format(avg_gen_loss),
    }
    with logs_file.open('w') as f:
        json.dump(logs, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    net_g = Generator()
    net_d = Discriminator()
    print("Generator Parameters:", sum(p.numel() for p in net_g.parameters() if p.requires_grad))
    print("Discriminator Parameters:", sum(p.numel() for p in net_d.parameters() if p.requires_grad))
