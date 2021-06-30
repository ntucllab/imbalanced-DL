import torchvision
import torchvision.transforms as transforms
import numpy as np

from .dataset_base import BaseDataset


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10, BaseDataset):
    """Imbalanced Cifar-10 Dataset

    References
    ----------
    https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
    """
    cls_num = 10

    def __init__(self,
                 root,
                 imb_type='exp',
                 imb_factor=0.01,
                 rand_number=0,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform,
                                               target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type,
                                                imb_factor)
        self.gen_imbalanced_data(img_num_list)


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = IMBALANCECIFAR10(root='./data',
                                train=True,
                                download=True,
                                transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    import pdb
    pdb.set_trace()
