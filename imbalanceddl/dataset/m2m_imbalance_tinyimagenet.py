import torch
import os
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import torch.nn as nn
import torch.backends.cudnn as cudnn
from .dataset_base import BaseDataset
from .m2m_dataset_base import M2mBaseDataset
from imbalanceddl.utils.config import get_args

cfg = get_args()
cudnn.benchmark = True
if torch.cuda.is_available():
    N_GPUS = torch.cuda.device_count()
else:
    N_GPUS = 0

class M2M_TINYIMAGENET200_LT(datasets.ImageFolder, BaseDataset, M2mBaseDataset):
    """Imbalance TINY200 Dataset

    Code for creating Imbalance TINY200 dataset
    """
    cls_num = 200

    def __init__(self,
                 root,
                 imb_type=cfg.imb_type,
                 imb_factor=cfg.imb_factor,
                 rand_number=0,
                 transform=None,
                 is_imbalance_data = False,
                 target_transform=None):
        super(M2M_TINYIMAGENET200_LT, self).__init__(root, transform,
                                               target_transform)
        np.random.seed(rand_number)
        self.data = np.array(self.samples)
        self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        if is_imbalance_data:
            print("=> Generating Imbalanced TINY200 with Type: {} | Ratio: {}".format(imb_type, imb_factor))
            self.gen_imbalanced_data(self.img_num_list)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of
            the target class.
        """
        path, target_n = self.data[index]
        target = self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        """
        Original Code is self.samples, we change to self.data
        Thus override this method
        """
        return len(self.data)

# In this implementation, we dont do normalization because M2m will normalize when adding noise into image.
def tinyimagenet_train_val_oversamples(tiny_root, batch_size=128):
    traindir = os.path.join(tiny_root, 'train')
    valdir = os.path.join(tiny_root, 'val/images')

    transform_train = transforms.Compose([
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
    transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])

    train_tiny200_lt = M2M_TINYIMAGENET200_LT(root=traindir, transform=transform_train, is_imbalance_data = True)
    val_tiny200_lt = datasets.ImageFolder(root=valdir, transform=transform_test)
    train_in_oversamples_idx = M2mBaseDataset.get_oversampled_data(train_tiny200_lt, train_tiny200_lt.img_num_list)

    train_in_loader = DataLoader(train_tiny200_lt, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_in_loader = DataLoader(val_tiny200_lt, batch_size=100, shuffle=False, num_workers=8, pin_memory=True)

    # Oversamples data is using for M2m method after epoch 250
    train_oversamples = DataLoader(train_tiny200_lt, batch_size=batch_size, sampler=WeightedRandomSampler(train_in_oversamples_idx, len(train_in_oversamples_idx)), num_workers=8)

    return train_in_loader, val_in_loader, train_oversamples