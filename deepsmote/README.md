# DeepSMOTE: Fusing Deep Learning and SMOTE for Imbalanced Data

DeepSMOTE paper: https://arxiv.org/pdf/2105.02340.pdf
## Overview

This is the implementation of DeepSMOTE method, based on the implementationof authors on MNIST dataset, we implemented DeepSMOTE method on CIFAR10, CIFAR100, SVHN10, CINIC10, TINY200. The file Deepsmote_Generate_Balanced.py, first train a DeepSMOTE model based on specific dataset and then generate synthetic samples on a trained model.

We heritage the structure of the code and dataset configuration setting from **Imbalanceddl** package. For more information about parameters in configuration, you can take a look into **Imbalanceddl** package.

## How to use
### Environment
* Python version: 3.8
* GPU: GeForce GTX 2080

### Quick Start: DeepSMOTE Generative synthetic data on CIFAR-10 dataset
* To generate DeepSMOTE synthetic data with long-tailed imbalance type and imbalance ratio of 0.01
```
python Deepsmote_Generate_Balanced.py --epochs 200 --dataset cifar10 --imb_factor 0.01 --imb_type 'exp'
```
|  Parameter | Description|
|:----------:|:----------:|
| `--epochs` | The number of epochs that you want to change for DeepSMOTE model|
|`--dataset`| Specific the dataset that you want to apply DeepSMOTE|
|`--imb_factor`| The ratio of imbalanced dataset such as: 100, 50, 10|
|`--imb_type`| The imbalance type fo dataset: exp (long-tailed), step|

After run this script, DeepSMOTE will generate the balanced dataset and save it into `deepsmote_models/cifar10` folder.

This balanced dataset will be used for training model into **Imbalanceddl** package.