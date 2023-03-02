# Class-Conditional Sharpness Aware Minimization for Deep Long-Tailed Learning
This is the source code for our paper based on Pytorch. 
This version is a demo of how to use CC-SAM.

## Installation

**Requirements**
* python 3.6
* PyTorch 1.4.0
* torchvision 0.5.0
* opencv 4.2
* numpy 1.19.5


## CIFAR-10-LT
Specify the data path ("data_root") in configs/Cifar10.json. Then running the following commend:
```bash
$ python train_cifar.py --config ./configs/Cifar10.json
```

## CIFAR-100-LT
Specify the data path ("data_root") in configs/Cifar100.json. Then running the following commend:
```bash
$ python train_cifar.py --config ./configs/Cifar100.json
```
