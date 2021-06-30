import unittest
from numpy.testing import assert_array_equal
import numpy as np

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from imbalanceddl.dataset.imbalance_svhn import IMBALANCESVHN


class TestCIFAR10(unittest.TestCase):
    def test_svhn10_exp100(self):
        train_dataset = IMBALANCESVHN(root='./data',
                                      imb_type="exp",
                                      imb_factor=0.01,
                                      rand_number=0,
                                      split='train',
                                      download=True,
                                      transform=None)

        true_cls_num = np.array([10, 1000, 599, 359, 215, 129, 77, 46, 27, 16])
        gen_cls_num = train_dataset.get_cls_num_list()
        assert_array_equal(gen_cls_num, true_cls_num)

    def test_svhn10_step100(self):
        train_dataset = IMBALANCESVHN(root='./data',
                                      imb_type="step",
                                      imb_factor=0.01,
                                      rand_number=0,
                                      split='train',
                                      download=True,
                                      transform=None)

        true_cls_num = np.array(
            [10, 1000, 1000, 1000, 1000, 1000, 10, 10, 10, 10])
        gen_cls_num = train_dataset.get_cls_num_list()
        assert_array_equal(gen_cls_num, true_cls_num)


if __name__ == "__main__":
    unittest.main()
