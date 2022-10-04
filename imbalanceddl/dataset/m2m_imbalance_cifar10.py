import torch
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
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

class M2M_CIFAR10_LT(datasets.CIFAR10, BaseDataset, M2mBaseDataset):
    """Imbalanced Cifar-10 Dataset

    References
    ----------
    https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
    """
    cls_num = 10

    def __init__(self,
                 root,
                 imb_type=cfg.imb_type,
                 imb_factor=cfg.imb_factor,
                 rand_number=0,
                 train=True,
                 transform=None,
                 target_transform=None,
                 is_imbalance_data=False,
                 download=False):
        super(M2M_CIFAR10_LT, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        if is_imbalance_data:
            self.gen_imbalanced_data(self.img_num_list)
            print("=> Generating Imbalanced CIFAR10 with Type: {} | Ratio: {}".format(imb_type, imb_factor))

# In this implementation, we dont do normalization because M2m will normalize when adding noise into image.
def cifar10_train_val_oversamples(cifar_root, batch_size=128):

    # No Normalization
    transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

    train_cifar10_lt = M2M_CIFAR10_LT(root=cifar_root, train=True, download=True, transform=transform_train, is_imbalance_data = True)
    val_cifar10_lt = datasets.CIFAR10(root=cifar_root, train=False, download=True, transform=transform_test)
    train_in_oversamples_idx = M2mBaseDataset.get_oversampled_data(train_cifar10_lt, train_cifar10_lt.img_num_list)

    train_in_loader = DataLoader(train_cifar10_lt, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_in_loader = DataLoader(val_cifar10_lt, batch_size=100, shuffle=False, num_workers=8, pin_memory=True)

    # Oversamples data is using for M2m method after epoch 160
    train_oversamples = DataLoader(train_cifar10_lt, batch_size=batch_size, sampler=WeightedRandomSampler(train_in_oversamples_idx, len(train_in_oversamples_idx)), num_workers=8)

    return train_in_loader, val_in_loader, train_oversamples