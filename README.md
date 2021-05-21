# GAN-Experiment

## TO DO
- BEGAN
- SAGAN
- UGAN
- Tensorboard to visualize the training process and generated images
- Multi-GPU training

### Install datasets (CelebA or LSUN)
```bash
$ bash download.sh CelebA
or 
$ bash download.sh LSUN
```

## Reference
- A good [tutorial](thttps://www.kaggle.com/ibtesama/gan-in-pytorch-with-fid/notebook#Fretchet-Inception-Distance) for GAN beginners.
- The calculation of Inception Score is borrowed from this [repository](https://github.com/sbarratt/inception-score-pytorch).
- The calculation of FID is borrowed from this [repository](https://github.com/mseitzer/pytorch-fid).
- The implementation of self-attention is borrowed from this [repository](https://github.com/heykeetae/Self-Attention-GAN).