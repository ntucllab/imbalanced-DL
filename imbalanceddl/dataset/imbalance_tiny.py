import torchvision
import numpy as np

from .dataset_base import BaseDataset


class IMBALANCETINY(torchvision.datasets.ImageFolder, BaseDataset):
    """Imbalanced Tiny-Imagenet Dataset

    Code for creating Imbalanced Tiny-Imagenet dataset
    """
    cls_num = 200

    def __init__(self,
                 root,
                 imb_type='exp',
                 imb_factor=0.01,
                 rand_number=0,
                 transform=None,
                 target_transform=None):
        super(IMBALANCETINY, self).__init__(root, transform, target_transform)
        np.random.seed(rand_number)
        self.data = np.array(self.samples)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type,
                                                imb_factor)
        self.gen_imbalanced_data(img_num_list)

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
