import os
import numpy as np
import collections
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CustomImageDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        _input = self.X[idx]
        target = self.Y[idx]
        if self.transform:
            _input = self.transform(_input)
        return _input, target

def get_balanced_deep_smote(dataset, batch_size, imb_type, imb_factor, num_workers=0):
    deepsmote_folder = 'deepsmote_models'
    train_data = 'train_data'
    train_label = 'train_label'
    if dataset == 'cifar10':
        dtrnimg = '../' + deepsmote_folder + '/' + dataset + '/' + dataset + '_' + imb_type + '_' + 'R' + str(int(1/imb_factor)) + '_' + train_data + '.txt'
        dtrnlab = '../' + deepsmote_folder + '/' + dataset + '/' + dataset + '_' + imb_type + '_' + 'R' + str(int(1/imb_factor)) + '_' + train_label + '.txt'
        val_transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif dataset == 'cifar100':
        dtrnimg = '../' + deepsmote_folder + '/' + dataset + '/' + dataset + '_' + imb_type + '_' + 'R' + str(int(1/imb_factor)) + '_' + train_data + '.txt'
        dtrnlab = '../' + deepsmote_folder + '/' + dataset + '/' + dataset + '_' + imb_type + '_' + 'R' + str(int(1/imb_factor)) + '_' + train_label + '.txt'
        val_transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    elif dataset == 'cinic10':
        dtrnimg = '../' + deepsmote_folder + '/' + dataset + '/' + dataset + '_' + imb_type + '_' + 'R' + str(int(1/imb_factor)) + '_' + train_data + '.txt'
        dtrnlab = '../' + deepsmote_folder + '/' + dataset + '/' + dataset + '_' + imb_type + '_' + 'R' + str(int(1/imb_factor)) + '_' + train_label + '.txt'
        val_transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835))])
    elif dataset == 'svhn10':
        dtrnimg = '../' + deepsmote_folder + '/' + dataset + '/' + dataset + '_' + imb_type + '_' + 'R' + str(int(1/imb_factor)) + '_' + train_data + '.txt'
        dtrnlab = '../' + deepsmote_folder + '/' + dataset + '/' + dataset + '_' + imb_type + '_' + 'R' + str(int(1/imb_factor)) + '_' + train_label + '.txt'
        val_transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
    elif dataset == 'tiny200':
        dtrnimg = '../' + deepsmote_folder + '/' + dataset + '/' + dataset + '_' + imb_type + '_' + 'R' + str(int(1/imb_factor)) + '_' + train_data + '.txt'
        dtrnlab = '../' + deepsmote_folder + '/' + dataset + '/' + dataset + '_' + imb_type + '_' + 'R' + str(int(1/imb_factor)) + '_' + train_label + '.txt'
        val_transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    print(dtrnimg)
    print(dtrnlab)
    dec_x = np.loadtxt(dtrnimg) 
    dec_y = np.loadtxt(dtrnlab)

    print('train imgs before reshape ',dec_x.shape) #(44993, 3072) 45500, 3072)
    print('train labels ',dec_y.shape) #(44993,) (45500,)

    print(collections.Counter(dec_y))
    dec_y = torch.tensor(dec_y,dtype=torch.long)

    if dataset == 'mnist':
        dec_x = dec_x.reshape(dec_x.shape[0],1,28,28) #(50000, 32, 32, 3)
        print('train imgs after reshape ',dec_x.shape)
    else:
        dec_x = dec_x.reshape(dec_x.shape[0],3,32,32) #(50000, 32, 32, 3)
        print('train imgs after reshape ',dec_x.shape) 

    dec_x = np.transpose(dec_x, axes= (0, 2, 3, 1)) # 0, 32, 32, 3
    print('train imgs after reshape ',dec_x.shape) 

    dec_x *=255.
    dec_x = np.clip(dec_x, 0, 255)
    dec_x = np.asarray(dec_x, dtype=np.uint8)

    balance_dataset = CustomImageDataset(dec_x, dec_y, val_transform)
    train_smote_loader = torch.utils.data.DataLoader(balance_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return train_smote_loader