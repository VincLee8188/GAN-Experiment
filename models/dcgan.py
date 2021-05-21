import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim, hidden_dim, im_chan):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(z_dim, hidden_dim * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(hidden_dim, im_chan, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (ngf) x 32 x 32
        )

    def forward(self, input):
        return self.main(input)


if __name__ == '__main__':
    net_g = Generator(64, 64, 3)
    # net_d = Discriminator()
    print("Generator Parameters:", sum(p.numel() for p in net_g.parameters() if p.requires_grad))
    # print("Discriminator Parameters:", sum(p.numel() for p in net_d.parameters() if p.requires_grad))
