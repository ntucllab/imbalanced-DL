import unittest
from numpy.testing import assert_array_equal
import numpy as np

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from imbalanceddl.dataset.imbalance_cifar import IMBALANCECIFAR10
from imbalanceddl.dataset.imbalance_cifar import IMBALANCECIFAR100


class TestCIFAR10(unittest.TestCase):
    def test_cifar10_exp100(self):
        train_dataset = IMBALANCECIFAR10(root='./data',
                                         imb_type="exp",
                                         imb_factor=0.01,
                                         rand_number=0,
                                         train=True,
                                         download=True,
                                         transform=None)

        true_cls_num = np.array(
            [5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50])
        gen_cls_num = train_dataset.get_cls_num_list()
        assert_array_equal(gen_cls_num, true_cls_num)

    def test_cifar10_step100(self):
        train_dataset = IMBALANCECIFAR10(root='./data',
                                         imb_type="step",
                                         imb_factor=0.01,
                                         rand_number=0,
                                         train=True,
                                         download=True,
                                         transform=None)

        true_cls_num = np.array(
            [5000, 5000, 5000, 5000, 5000, 50, 50, 50, 50, 50])
        gen_cls_num = train_dataset.get_cls_num_list()
        assert_array_equal(gen_cls_num, true_cls_num)

    def test_cifar100_exp100(self):
        train_dataset = IMBALANCECIFAR100(root='./data',
                                          imb_type="exp",
                                          imb_factor=0.01,
                                          rand_number=0,
                                          train=True,
                                          download=True,
                                          transform=None)

        true_cls_num = np.array([
            500, 477, 455, 434, 415, 396, 378, 361, 344, 328, 314, 299, 286,
            273, 260, 248, 237, 226, 216, 206, 197, 188, 179, 171, 163, 156,
            149, 142, 135, 129, 123, 118, 112, 107, 102, 98, 93, 89, 85, 81,
            77, 74, 70, 67, 64, 61, 58, 56, 53, 51, 48, 46, 44, 42, 40, 38, 36,
            35, 33, 32, 30, 29, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
            15, 15, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 7, 6,
            6, 6, 6, 5, 5, 5, 5
        ])
        gen_cls_num = train_dataset.get_cls_num_list()
        assert_array_equal(gen_cls_num, true_cls_num)

    def test_cifar100_step100(self):
        train_dataset = IMBALANCECIFAR100(root='./data',
                                         imb_type="step",
                                         imb_factor=0.01,
                                         rand_number=0,
                                         train=True,
                                         download=True,
                                         transform=None)

        true_cls_num = np.array([
            500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
            500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
            500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
            500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 5, 5, 5, 5,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            5, 5
        ])
        gen_cls_num = train_dataset.get_cls_num_list()
        assert_array_equal(gen_cls_num, true_cls_num)


if __name__ == "__main__":
    unittest.main()
