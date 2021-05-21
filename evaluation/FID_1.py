import torch, torchvision
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy import linalg


class InceptionV3(nn.Module):
    def __init__(self, resize=True):
        super(InceptionV3, self).__init__()
        self.resize = resize
        self.blocks = nn.ModuleList()
        inception = torchvision.models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0).eval())

        # Block 1: maxpool1 to maxpool2
        block1 = [
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block1).eval())

        # Block 2: maxpool2 to aux classifier
        block2 = [
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
        ]
        self.blocks.append(nn.Sequential(*block2).eval())

        # Block 3: aux classifier to final avgpool
        block3 = [
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        ]
        self.blocks.append(nn.Sequential(*block3).eval())

        for bl in self.blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.transform = nn.functional.interpolate
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False)

    def __call__(self, inputs):
        if len(inputs.size()) > 4:
            inputs = torch.cat([inputs[:, :, i] for i in range(inputs.size(2))], dim=0)
        if inputs.shape[1] != 3:
            inputs = inputs.repeat(1, 3, 1, 1)
        if self.resize:
            inputs = self.transform(inputs, mode='bilinear', size=(299, 299), align_corners=False)
        x = (inputs - self.mean) / self.std
        for block in self.blocks:
            x = block(x)

        return x


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def get_fid_score(g_feats, gt_feats):
    def calcu_statis(feats):
        mu = np.mean(feats, axis=0)
        sigma = np.cov(feats, rowvar=False)
        return mu, sigma
    m1, s1 = calcu_statis(g_feats)
    m2, s2 = calcu_statis(gt_feats)
    fid_score = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_score


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InceptionV3().to(device)


    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)


    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    mnist1 = dset.CIFAR10(root='../dataset', train=True, download=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                          ])
                          )

    mnist2 = dset.CIFAR10(root='../dataset', train=False, download=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                          ])
                          )

    dataset1 = torch.stack(list(iter(IgnoreLabelDataset(mnist1)))[:1000], dim=0).to(device)
    dataset2 = torch.stack(list(iter(IgnoreLabelDataset(mnist2)))[:1000], dim=0).to(device)

    print("Calculating FID ...")
    g_feat = model(dataset1)
    gt_feat = model(dataset2)
    print(get_fid_score(g_feat.cpu().numpy().squeeze(), gt_feat.cpu().numpy().squeeze()))
