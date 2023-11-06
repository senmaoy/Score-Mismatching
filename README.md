# SCORE MISMATCHING FOR GENERATIVE MODELING

Official Pytorch implementation for our paper [SCORE MISMATCHING FOR GENERATIVE MODELING](https://arxiv.org/abs/2309.11043) 


### Requirements
- python 3.7
- pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
- 2080 TI or 3090 TI (it will be very slow for 1080 TI)
Most of the code is based on stylegan2-ada-pytorch, the required packages can be found in the [link](https://github.com/NVlabs/stylegan2-ada-pytorch)
### Training

**Train SMM models:**
  - For cifar10 dataset: python train.py --outdir=./training-runs --data=./datasets/cifar10.zip --gpus=1 --cfg=cifar --snap=50 --aug=noaug
  
TODO: soon

