import numpy as np

from torchvision import datasets
# from m2m_dataset_base import M2mBaseDataset

class M2M_SVHN_LT(datasets.SVHN):
    """Imbalance CINIC-10 Dataset

    Code for creating Imbalance CINIC-10 dataset
    """
    cls_num = 10

    def __init__(self,
                 root,
                 imb_type='exp',
                 imb_factor=0.01,
                 rand_number=0,
                 split='train',
                 transform=None,
                 target_transform=None,
                 is_imbalance_data = False,
                 download=False):
        super(M2M_SVHN_LT, self).__init__(root, split, transform,
                                            target_transform, download)
        np.random.seed(rand_number)
        self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        if is_imbalance_data:
            self.gen_imbalanced_data(self.img_num_list)
            print("=> Generating Imbalanced SVHN10 with Type: {} | Ratio: {}".format(imb_type, imb_factor))
        # else:
        #     self.change_order_class()
        #     print("=> Loading the orginal SVHN with Is_imbalance_data = False")
    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        # Follow the above repo
        img_max = 1000
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        # print(img_num_per_cls)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.labels, dtype=np.int64)
        classes = np.unique(targets_np)
        # shift label 0 to the last (as original SVHN labels)
        # since SVHN itself is long-tailed, label 10 (0 here) may not
        # contain enough images
        # classes = np.concatenate([classes[1:], classes[:1]], axis=0)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            # print(f"Class {the_class}:\t{len(idx)}")
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([
                the_class,
            ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.labels = new_targets
        assert new_data.shape[0] == len(new_targets), 'Length of data & \
            labels do not match!'

    def get_imbalanced_data(dataset, num_sample_per_class, shuffle=False, random_seed=0):
        """
        Return a list of imbalanced indices from a dataset.
        Input: A dataset (e.g., CIFAR-10), num_sample_per_class: list of integers
        Output: imbalanced_list
        """
        length = dataset.__len__()
        num_sample_per_class = list(num_sample_per_class)
        selected_list = []
        indices = list(range(0,length))

        for i in range(0, length):
            index = indices[i]
            _, label = dataset.__getitem__(index)
            if num_sample_per_class[label] > 0:
                selected_list.append(index)
                num_sample_per_class[label] -= 1

        return selected_list

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
