# Example

## Overview
* **imbalanced-DL** (imported as imbalanceddl) is a package for Deep Imbalanced Learning.
* In this example, we can run a collection of benchmark for research purpose.

## How to use
### Environment
* Python version: 3.6.8
* GPU: GeForce GTX 1080

### Quick Start: Imbalanced CIFAR-10 Training
* To reproduce Empirical Risk Minimization (ERM) training with long-tailed imbalance type and imbalance ratio of 0.01
```
python main.py --gpu 0 --seed 1126 --config config/config_cifar10.yaml --strategy ERM
```
|  Parameter | Description|
|:----------:|:----------:|
| `--config` | Path to config file (specify by different dataset)|
|`--strategy`| `ERM`, `DRW`, `LDAM_DRW`, `Mixup_DRW`, `Remix_DRW`|
|`--seed`    | Recommend to use seed training|

* The best result will be around 71% validation accuracy.
* To change imbalance type and imbalance ratio, you can either modify the config file or change it in command line.
    * For example, `python main.py --gpu 0 --seed 1126 --c --config/config_cifar10.yaml --imb_type step --imb_factor 0.2 --strategy ERM`
* To evaluate the trained model, you can specify path to `--best_model`, for example:
```
python main.py --gpu 0 --seed 1126 --c config/config_cifar10.yaml --best_model ./checkpoint_cifar10/cifar10_exp_0.01_ERM_200_1126/ckpt.best.pth.tar

```
* We provide one trained example model in `checkpoint_cifar10/`
* You will get the per class accuracy as the following:
```
[0.964,0.984,0.847,0.790,0.807,0.603,0.669,0.607,0.434,0.487]
```

### Strategy Supported
* We support a couple of strategies for dealing with deep imbalanced classification.
* To specify, simply put the name of the strategy in the command line, such as, `--strategy Mixup_DRW`.

| Strategy | Description|
|:--------:|:----------:|
|  `ERM`   |Baseline Training|
|[`DRW`](https://arxiv.org/pdf/1906.07413.pdf)| Deferred Re-Weighting|
|[`LDAM_DRW`](https://arxiv.org/pdf/1906.07413.pdf)| Label Distribution Aware Margin Loss with DRW|
|[`Mixup_DRW`](https://github.com/facebookresearch/mixup-cifar10)| Mixup with DRW|
|[`Remix_DRW`](https://arxiv.org/pdf/2007.03943.pdf)| Remix with DRW|
|[`MAMix_DRW`]()| MAMix with DRW|

### Dataset supported
* We provide some common image datasets for research benchmark.
* To specify, use the corresponding config file, for example:
    * `python main.py --gpu 0 --seed 1126 --c config/config_tiny200.yaml` for using `Tiny-ImageNet`.
* We follow [This repo](https://github.com/YyzHarry/imbalanced-semi-self) to prepare imbalanced SVHN.
* We follow [LDAM-DRW](https://github.com/kaidic/LDAM-DRW) to prepare imbalanced CIFAR.

| Dataset | Description|
|:-------:|:----------:|
|`CIFAR10`|`torchvision.datasets.CIFAR10`|
|`CIFAR100`|`torchvision.datasets.CIFAR100`|
|`SVHN`|`torchvision.datasets.SVHN`|
|`CINIC10`|Go to [This repo](https://github.com/BayesWatch/cinic-10) to download the dataset (`CINIC-10.tar`)|
|`Tiny-ImageNet`|Go to [Standford CS231](http://cs231n.stanford.edu/tiny-imagenet-200.zip) to download `tiny-imagenet-200.zip`|

### Model supported
#### Backbone (feature extractor)
* Now we support two kinds of backbone for common research usage.
    * `resnet32` and `resnet18`
* Other backbones can be extended in `lib/net`.
#### Classifier
* We support two kinds of classifier.
    * `dot product classifier` and `cosine similarity classifier`
* Typically, cosine similarity classifier is used only when the strategy is related to `LDAM`.


## Benchmark Result
### CIFAR-10
* Train from scratch
```
python main.py --gpu 0 --seed 1126 --c config/config_cifar10.yaml --strategy [Strategy_Name]
```
* Eval with best model
```
python main.py --gpu 0 --seed 1126 --c config/config_cifar10.yaml --strategy [Strategy_Name] --best_model [Path to Best Model]
```
* Following [LDAM Work](https://arxiv.org/pdf/1906.07413.pdf), we use **ResNet32** as the backone.

| `imb_type` |`imb_factor`|   Model   | Strategy | Validation Top 1 |
|:----------:|:----------:|:---------:|:--------:|:----------------:|
|long-tailed | 100        | ResNet32  | ERM      | 71.23            |
|long-tailed | 100        | ResNet32  | DRW      | 75.08            |
|long-tailed | 100        | ResNet32  | LDAM-DRW | 77.75            |
|long-tailed | 100        | ResNet32  | Mixup-DRW| 82.11            |
|long-tailed | 100        | ResNet32  | Remix-DRW| 81.82            |
|long-tailed | 100        | ResNet32  | MAMix-DRW| 82.29            |


### CIFAR-100
* Train from scratch
```
python main.py --gpu 0 --seed 1126 --c config/config_cifar100.yaml --strategy [Strategy_Name]
```
* Eval with best model
```
python main.py --gpu 0 --seed 1126 --c config/config_cifar100.yaml --strategy [Strategy_Name] --best_model [Path to Best Model]
```
* Following [LDAM Work](https://arxiv.org/pdf/1906.07413.pdf), we use **ResNet32** as the backone.

| `imb_type` |`imb_factor`|   Model   | Strategy | Validation Top 1 |
|:----------:|:----------:|:---------:|:--------:|:----------------:|
|long-tailed | 100        | ResNet32  | ERM      | 38.46            |
|long-tailed | 100        | ResNet32  | DRW      | 40.40            |
|long-tailed | 100        | ResNet32  | LDAM-DRW | 41.28            |
|long-tailed | 100        | ResNet32  | Mixup-DRW| 46.91            |
|long-tailed | 100        | ResNet32  | Remix-DRW| 46.00            |
|long-tailed | 100        | ResNet32  | MAMix-DRW| 46.93            |


### CINIC-10
* Train from scratch
```
python main.py --gpu 0 --seed 1126 --c config/config_cinic10.yaml --strategy [Strategy_Name]
```
* Eval with best model
```
python main.py --gpu 0 --seed 1126 --c config/config_cinic10.yaml --strategy [Strategy_Name] --best_model [Path to Best Model]
```
* Following [Remix Work](https://arxiv.org/pdf/2007.03943.pdf), we use **ResNet18** as the backone.

| `imb_type` |`imb_factor`|   Model   | Strategy | Validation Top 1 |
|:----------:|:----------:|:---------:|:--------:|:----------------:|
|long-tailed | 100        | ResNet18  | ERM      | 61.08            |
|long-tailed | 100        | ResNet18  | DRW      | 63.75            |
|long-tailed | 100        | ResNet18  | LDAM-DRW | 68.15            |
|long-tailed | 100        | ResNet18  | Mixup-DRW| 71.40            |
|long-tailed | 100        | ResNet18  | Remix-DRW| 71.15            |
|long-tailed | 100        | ResNet18  | MAMix-DRW| 71.76            |


### SVHN
* Train from scratch
```
python main.py --gpu 0 --seed 1126 --c config/config_svhn10.yaml --strategy [Strategy_Name]
```
* Eval with best model
```
python main.py --gpu 0 --seed 1126 --c config/config_svhn10.yaml --strategy [Strategy_Name] --best_model [Path to Best Model]
```
* Following [imbalanced-semi-self Work](https://github.com/YyzHarry/imbalanced-semi-self), we use **ResNet32** as the backone.

| `imb_type` |`imb_factor`|   Model   | Strategy | Validation Top 1 |
|:----------:|:----------:|:---------:|:--------:|:----------------:|
|long-tailed | 100        | ResNet32  | ERM      | 79.91            |
|long-tailed | 100        | ResNet32  | DRW      | 80.68            |
|long-tailed | 100        | ResNet32  | LDAM-DRW | 83.48            |
|long-tailed | 100        | ResNet32  | Mixup-DRW| 85.19            |
|long-tailed | 100        | ResNet32  | Remix-DRW| 84.52            |
|long-tailed | 100        | ResNet32  | MAMix-DRW| 85.41            |


### Tiny-ImageNet
* Train from scratch
```
python main.py --gpu 0 --seed 1126 --c config/config_tiny200.yaml --strategy [Strategy_Name]
```
* Eval with best model
```
python main.py --gpu 0 --seed 1126 --c config/config_tiny200.yaml --strategy [Strategy_Name] --best_model [Path to Best Model]
```
* Following [LDAM Work](https://arxiv.org/pdf/1906.07413.pdf), we use **ResNet18** as the backone.

| `imb_type` |`imb_factor`|   Model   | Strategy | Validation Top 1 |
|:----------:|:----------:|:---------:|:--------:|:----------------:|
|long-tailed | 100        | ResNet18  | ERM      | 32.86            |
|long-tailed | 100        | ResNet18  | DRW      | 33.81            |
|long-tailed | 100        | ResNet18  | LDAM-DRW | 31.90            |
|long-tailed | 100        | ResNet18  | Mixup-DRW| 37.97            |
|long-tailed | 100        | ResNet18  | Remix-DRW| 36.89            |
|long-tailed | 100        | ResNet18  | MAMix-DRW| 37.73            |


## Acknowledgement
We thank the following repos for the code sharing.
* [LDAM-DRW official implementation](https://github.com/kaidic/LDAM-DRW)
* [Rethinking the Value of Labels for Improving Class-Imbalanced Learning](https://github.com/YyzHarry/imbalanced-semi-self)
* [Mixup](https://github.com/facebookresearch/mixup-cifar10)
* [Liu Resnet18](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py)
* [CINIC-10](https://github.com/BayesWatch/cinic-10)
* [Tiny-ImageNet Downloader](https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4)
* [Tiny-ImageNet Prepare Data](https://github.com/DennisHanyuanXu/Tiny-ImageNet)
* [Decoupling Representation and Classifier for Long-Tailed Recognition](https://github.com/facebookresearch/classifier-balancing)
