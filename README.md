# GAN-Experiment

## News
A [anime branch](https://github.com/VincLee8188/GAN-Experiment/tree/anime) for anime generation is now AVAILABLE!!! It was developed by my friend [hsj](https://github.com/hsj576).

## Usage
` python main.py --checkpoint_dir checkpoint --model dcgan --dataset MNIST --device 0`

## TO DO
- Add support for CelebA and LSUN dataset.
- Tensorboard to visualize the training process and generated images
- Multi-GPU training

[comment]: <> (### Install datasets &#40;CelebA or LSUN&#41;)

[comment]: <> (```bash)

[comment]: <> ($ bash download.sh CelebA)

[comment]: <> (or )

[comment]: <> ($ bash download.sh LSUN)

[comment]: <> (```)

## Reference
- A good [tutorial](https://www.kaggle.com/ibtesama/gan-in-pytorch-with-fid/notebook#Fretchet-Inception-Distance) for GAN beginners.
- The calculation of Inception Score is borrowed from this [repository](https://github.com/sbarratt/inception-score-pytorch).
- The calculation of FID is borrowed from this [repository](https://github.com/mseitzer/pytorch-fid).
- The implementation of self-attention is borrowed from this [repository](https://github.com/heykeetae/Self-Attention-GAN), 
  and the download script for CelebA and LSUN `download.sh` is also borrowed from it. 
