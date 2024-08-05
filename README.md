# SEMBG

## Introduction

This repository is the Pytorch code for the paper [Low-Cost Self-Ensembles Based on Multi-Branch Transformation and Grouped Convolution].

These codes are examples for CIFAR100 with Wide_ResNet28-10.

## Dependencies

* Python 3.6.13 (Anaconda)
* Pytorch 1.7.1
* CUDA 10.1

## Run

run for training a single model:

```
python train_single.py
```

run for training SEMBG (N=3):

```
python train_SEMBG.py
```

## Citation 

```latex
@article{SEMBG,
  title={Low-Cost Self-Ensembles Based on Multi-Branch Transformation and Grouped Convolution},
  author ={H. {Lee} and J. -S. {Lee}},
  journal = {arXiv preprint	arXiv:1912.02757}
  year={2024},
}
```
