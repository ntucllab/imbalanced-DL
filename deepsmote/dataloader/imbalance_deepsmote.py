import numpy as np
import os
from dataloader.dataset_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from dataloader.dataset_svhn10 import M2M_SVHN_LT
from dataloader.dataset_tiny200 import M2M_TINYIMAGENET200_LT
import torchvision.transforms as transforms
from dataloader.dataset_cinic10 import M2M_CINIC10_LT
from torch.utils.data import DataLoader


def cifar10_deepsmote(imb_type, imb_factor):
    val_transform = transforms.Compose(
            [transforms.ToTensor()])

    datasets = {}
    datasets['train'] = IMBALANCECIFAR10(root='./datasets', train=True, download=True, transform=val_transform, imb_type=imb_type, imb_factor=imb_factor)
    cls_num_list = datasets['train'].get_cls_num_list()
    train_data = datasets['train'].data
    # Normalization
    train_data = np.asarray(train_data, dtype=float)
    train_data /= 255.
    # train_data -= np.asarray((0.4914, 0.4822, 0.4465))
    # train_data /= np.asarray((0.2023, 0.1994, 0.2010))
    train_data = np.transpose(train_data, axes= (0, 3, 1, 2)) # 0, 3*32*32
    train_data = np.resize(train_data, new_shape=(train_data.shape[0], 3*32*32)) #
    label = datasets['train'].targets
    label = np.asarray(label, dtype=int)

    return train_data, label, cls_num_list

def cifar100_deepsmote(imb_type, imb_factor):
    val_transform = transforms.Compose(
            [transforms.ToTensor()])

    datasets = {}
    datasets['train'] = IMBALANCECIFAR100(root='./datasets', train=True, download=True, transform=val_transform, imb_type=imb_type, imb_factor=imb_factor)
    cls_num_list = datasets['train'].get_cls_num_list()
    train_data = datasets['train'].data

    # Normalization
    train_data = np.asarray(train_data, dtype=float)
    train_data /= 255.
    # train_data -= np.asarray((0.4914, 0.4822, 0.4465))
    # train_data /= np.asarray((0.2023, 0.1994, 0.2010))
    train_data = np.transpose(train_data, axes= (0, 3, 1, 2)) # 0, 3*32*32
    train_data = np.resize(train_data, new_shape=(train_data.shape[0], 3*32*32)) #
    label = datasets['train'].targets
    label = np.asarray(label, dtype=int)

    return train_data, label, cls_num_list

def svhn10_deepsmote(imb_type, imb_factor):
    val_transform = transforms.Compose(
            [transforms.ToTensor()])

    datasets = {}
    datasets['train'] = M2M_SVHN_LT(root='./datasets', split = 'train', download=True, is_imbalance_data = True, transform=val_transform, imb_type=imb_type, imb_factor=imb_factor)
    cls_num_list = datasets['train'].get_cls_num_list()
    train_data = datasets['train'].data

    # Normalization
    train_data = np.asarray(train_data, dtype=float)
    train_data /= 255.
    # train_data -= np.asarray((.5, .5, .5))
    # train_data /= np.asarray((.5, .5, .5))
    train_data = np.transpose(train_data, axes= (0, 3, 1, 2)) # 0, 3*32*32
    train_data = np.resize(train_data, new_shape=(train_data.shape[0], 3*32*32)) #
    label = datasets['train'].labels
    label = np.asarray(label, dtype=int)

    return train_data, label, cls_num_list

def cinic10_deepsmote(imb_type, imb_factor):
    val_transform = transforms.Compose(
            [transforms.ToTensor()])

    train_cinic10_lt = {}
    cinic_root = 'datasets/cinic/'
    traindir = os.path.join(cinic_root, 'train')

    train_cinic10_lt = M2M_CINIC10_LT(root=traindir, transform=val_transform, is_imbalance_data = True, imb_type=imb_type, imb_factor=imb_factor)
    cls_num_list = train_cinic10_lt.get_cls_num_list()

    batch_size = len(train_cinic10_lt.targets)
    train_in_loader = DataLoader(train_cinic10_lt, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    train_batch_X, train_batch_y = next(iter(train_in_loader))
    # Normalization
    train_data = np.asarray(train_batch_X, dtype=float)
    # if dont use dataloader --> will normalize by deviding 255
    # train_data /= 255.
    # train_data -= np.asarray((0.4914, 0.4822, 0.4465))
    # train_data /= np.asarray((0.2023, 0.1994, 0.2010))
    train_data = np.transpose(train_data, axes= (0, 3, 1, 2)) # 0, 3*32*32
    train_data = np.resize(train_data, new_shape=(train_data.shape[0], 3*32*32)) #
    label = train_batch_y
    label = np.asarray(label, dtype=int)

    return train_data, label, cls_num_list


def tiny200_deepsmote(imb_type, imb_factor):
    val_transform = transforms.Compose(
            [transforms.ToTensor()])
    
    train_tiny200_lt = {}
    tiny_root = 'datasets/tiny/tiny-imagenet-200/'
    traindir = os.path.join(tiny_root, 'train')

    train_tiny200_lt = M2M_TINYIMAGENET200_LT(root=traindir, transform=val_transform, is_imbalance_data = True, imb_type=imb_type, imb_factor=imb_factor)
    cls_num_list = train_tiny200_lt.get_cls_num_list()

    batch_size = len(train_tiny200_lt.targets)
    train_in_loader = DataLoader(train_tiny200_lt, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    train_batch_X, train_batch_y = next(iter(train_in_loader))
    # Normalization
    train_data = np.asarray(train_batch_X, dtype=float)
    # if dont use dataloader --> will normalize by deviding 255
    # train_data /= 255.
    # train_data -= np.asarray((0.4914, 0.4822, 0.4465))
    # train_data /= np.asarray((0.2023, 0.1994, 0.2010))
    train_data = np.transpose(train_data, axes= (0, 3, 1, 2)) # 0, 3*32*32
    train_data = np.resize(train_data, new_shape=(train_data.shape[0], 3*32*32)) #
    label = train_batch_y
    label = np.asarray(label, dtype=int)

    return train_data, label, cls_num_list