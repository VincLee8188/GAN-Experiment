import torch
from torch import nn
from tqdm.auto import tqdm
from pathlib import Path as P
from utils import save_checkpoint, save_tensor_images
from evaluation import *
import numpy as np
import json
from .spectral_normalization import SpectralNorm


class SelfAttention(nn.Module):

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  
        energy = torch.bmm(proj_query, proj_key)  
        attention = self.softmax(energy)  
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class Generator(nn.Module):

    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 8, stride=1, padding=0),
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4),
            SelfAttention(hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2),
            SelfAttention(hidden_dim * 2),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            SelfAttention(hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, final_layer=True)
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1, final_layer=False):
        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                SpectralNorm(nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding)),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
            )
        else:  # Final Layer
            return nn.Sequential(
                (nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding)),
                nn.Tanh()
            )

    def unsqueeze_noise(self, noise):
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        x = self.unsqueeze_noise(noise)
        return self.gen(x)


def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)


class Discriminator(nn.Module):

    def __init__(self, im_chan=1, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            SelfAttention(hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, hidden_dim * 4),
            SelfAttention(hidden_dim * 4),
            self.make_disc_block(hidden_dim * 4, hidden_dim * 8),
            SelfAttention(hidden_dim * 8),
            self.make_disc_block(hidden_dim * 8, 1, stride=1, padding=0, final_layer=True)
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=1, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                SpectralNorm(nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2)
            )
        else:
            return nn.Sequential(
                SpectralNorm(nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding))
            )

    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)


def weight_inits(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0, 0.02)
        nn.init.constant_(m.bias, 0)


def get_disc_model(args):
    return Discriminator(im_chan=args.channel, hidden_dim=args.hidden_dim)


def get_gen_model(args):
    return Generator(z_dim=args.z_dim, im_chan=args.channel, hidden_dim=args.hidden_dim)


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
    n_epochs, z_dim, display_step = args.n_epochs, args.z_dim, args.display_step
    global_epoch = ori_epoch
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0

    disc.apply(weight_inits)
    gen.apply(weight_inits)

    for cur_epoch in range(args.n_epochs):
        epoch = cur_epoch + global_epoch
        print('\nStarting Epoch: {}'.format(epoch))
        num_batch = len(dataloader)
        for real in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.to(device)

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
        print(f'Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, '
              f'Discriminator loss:  {mean_discriminator_loss}')

        mean_discriminator_loss = 0
        mean_generator_loss = 0

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

    for real in val_loader:
        cur_batch_size = len(real)
        real = real.to(device)
        fake_noise = get_noise(cur_batch_size, z_dim, device)
        fake = gen(fake_noise)

        inception = FID.get_eval_model(device)
        img_size = (args.channel, args.Size, args.Size)
        g_feat = inception(fake)[0] 
        gt_feat = inception(real)[0] 
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

    save_tensor_images(fake, checkpoint_dir, global_epoch, label='fake', size=(args.channel, args.Size, args.Size))
    save_tensor_images(real, checkpoint_dir, global_epoch, size=(args.channel, args.Size, args.Size))

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
    gen = Generator(z_dim=100, im_chan=3, hidden_dim=64)
    print(gen)