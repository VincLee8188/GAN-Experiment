import torch
from pathlib import Path as P
from torchvision.utils import make_grid, save_image


def write_config_to_file(config, save_path):
    with open(P(save_path) / 'config.txt', 'w') as file:
        for arg in vars(config):
            file.write(str(arg) + ': ' + str(getattr(config, arg)) + '\n')


def save_tensor_images(image_tensor, checkpoint_dir, epoch, num_images=16, size=(3, 32, 32), prefix='train', label='real'):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    folder = P(checkpoint_dir) / 'samples_epoch{:05d}'.format(epoch)
    folder.mkdir(exist_ok=True)
    for i in range(len(image_unflat) // num_images):
        save_image(make_grid(image_unflat[i*num_images:(i+1)*num_images], nrow=4), folder / f'{prefix}_{i}_{label}.jpg')


def save_checkpoint(checkpoint_dir, disc, gen, opt_d, opt_g, epoch, prefix=''):
    checkpoint_path = P(checkpoint_dir) / '{}checkpoint_epoch{:05d}.pth'.format(prefix, epoch)
    opt_d_state = opt_d.state_dict()
    opt_g_state = opt_g.state_dict()
    torch.save({
        'disc_state_dict': disc.state_dict(),
        'gen_state_dict': gen.state_dict(),
        'opt_d_state': opt_d_state,
        'opt_g_state': opt_g_state,
        'global_epoch': epoch,
    }, checkpoint_path)
    print('Saved checkpoint:', checkpoint_path)


def load_checkpoint(path, disc, gen, opt_d, opt_g, rank):
    def _load(checkpoint_path):
        if torch.cuda.is_available():
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
        else:
            # Load all tensors onto a CPU, using a function.
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        return checkpoint

    def _model_load(model, ori_dict):
        new_s = {}
        for k, v in ori_dict.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

    checkpoint = _load(path)
    disc_dict = checkpoint['disc_state_dict']
    gen_dict = checkpoint['gen_state_dict']

    _model_load(disc, disc_dict)
    _model_load(gen, gen_dict)

    opt_d.load_state_dict(checkpoint['opt_d_state'])
    opt_g.load_state_dict(checkpoint['opt_g_state'])
    global_epoch = checkpoint['global_epoch'] + 1

    return global_epoch
