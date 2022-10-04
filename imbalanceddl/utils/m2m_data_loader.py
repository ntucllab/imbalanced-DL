import torch
import os
import numpy.random as nr
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from imbalanceddl.utils.m2m_utils import InputNormalize
import torch.backends.cudnn as cudnn
from imbalanceddl.utils.config import get_args

ARGS = get_args()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = ARGS.gpu
cudnn.benchmark = True
if torch.cuda.is_available():
    N_GPUS = torch.cuda.device_count()
else:
    N_GPUS = 0

num_test_samples_cifar10 = [1000] * 10
num_test_samples_cifar100 = [100] * 100
#---------############------------------

DATA_ROOT = os.path.expanduser('~/data')
DATASET = ARGS.dataset
BATCH_SIZE = ARGS.batch_size
MODEL = ARGS.model

def make_longtailed_imb(max_num, class_num, gamma):
    mu = np.power(1/gamma, 1/(class_num - 1))
    print(mu)
    class_num_list = []
    for i in range(class_num):
        class_num_list.append(int(max_num * np.power(mu, i)))

    return list(class_num_list)


def get_val_test_data(dataset, num_sample_per_class, shuffle=False, random_seed=0):
    """
    Return a list of indices for validation and test from a dataset.
    Input: A test dataset (e.g., CIFAR-10)
    Output: validation_list and test_list
    """
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    num_samples = num_sample_per_class[0] # Suppose that all classes have the same number of test samples

    val_list = []
    test_list = []
    indices = list(range(0, length))
    if shuffle:
        nr.shuffle(indices)
    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > (9 * num_samples / 10):
            val_list.append(index)
            num_sample_per_class[label] -= 1
        else:
            test_list.append(index)
            num_sample_per_class[label] -= 1

    return val_list, test_list


def get_oversampled_data(dataset, num_sample_per_class, random_seed=0):
    """
    Return a list of imbalanced indices from a dataset.
    Input: A dataset (e.g., CIFAR-10), num_sample_per_class: list of integers
    Output: oversampled_list ( weights are increased )
    """
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    num_samples = list(num_sample_per_class)

    selected_list = []
    indices = list(range(0,length))
    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > 0:
            selected_list.append(1 / num_samples[label])
            num_sample_per_class[label] -= 1
    # import pdb
    # pdb.set_trace()
    return selected_list


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


def get_oversampled(dataset, num_sample_per_class, batch_size, TF_train, TF_test):
    print("Building {} CV data loader with {} workers".format(dataset, 8))
    ds = []

    if dataset == 'cifar10':
        dataset_ = datasets.CIFAR10
        num_test_samples = num_test_samples_cifar10
    elif dataset == 'cifar100':
        dataset_ = datasets.CIFAR100
        num_test_samples = num_test_samples_cifar100
    else:
        raise NotImplementedError()

    train_cifar = dataset_(root=DATA_ROOT, train=True, download=False, transform=TF_train)
    targets = np.array(train_cifar.targets)
    classes, class_counts = np.unique(targets, return_counts=True)
    nb_classes = len(classes)

    imbal_class_counts = [int(i) for i in num_sample_per_class]
    class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]

    imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
    imbal_class_indices = np.hstack(imbal_class_indices)

    train_cifar.targets = targets[imbal_class_indices]
    train_cifar.data = train_cifar.data[imbal_class_indices]

    assert len(train_cifar.targets) == len(train_cifar.data)

    train_in_idx = get_oversampled_data(train_cifar, num_sample_per_class)
    train_in_loader = DataLoader(train_cifar, batch_size=batch_size,
                                 sampler=WeightedRandomSampler(train_in_idx, len(train_in_idx)), num_workers=8)
    ds.append(train_in_loader)

    test_cifar = dataset_(root=DATA_ROOT, train=False, download=False, transform=TF_test)
    val_idx, test_idx = get_val_test_data(test_cifar, num_test_samples)
    val_loader = DataLoader(test_cifar, batch_size=100,
                            sampler=SubsetRandomSampler(val_idx), num_workers=8)
    test_loader = DataLoader(test_cifar, batch_size=100,
                             sampler=SubsetRandomSampler(test_idx), num_workers=8)
    ds.append(val_loader)
    ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds

    return ds


def get_imbalanced(dataset, num_sample_per_class, batch_size, TF_train, TF_test):
    print("Building CV {} data loader with {} workers".format(dataset, 8))
    ds = []

    if dataset == 'cifar10':
        dataset_ = datasets.CIFAR10
        num_test_samples = num_test_samples_cifar10
    elif dataset == 'cifar100':
        dataset_ = datasets.CIFAR100
        num_test_samples = num_test_samples_cifar100
    else:
        raise NotImplementedError()

    train_cifar = dataset_(root=DATA_ROOT, train=True, download=False, transform=TF_train)
    train_in_idx = get_imbalanced_data(train_cifar, num_sample_per_class)
    train_in_loader = torch.utils.data.DataLoader(train_cifar, batch_size=batch_size,
                                                  sampler=SubsetRandomSampler(train_in_idx), num_workers=8)
    ds.append(train_in_loader)

    test_cifar = dataset_(root=DATA_ROOT, train=False, download=False, transform=TF_test)
    val_idx, test_idx= get_val_test_data(test_cifar, num_test_samples)
    val_loader = torch.utils.data.DataLoader(test_cifar, batch_size=100,
                                                  sampler=SubsetRandomSampler(val_idx), num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_cifar, batch_size=100,
                                                  sampler=SubsetRandomSampler(test_idx), num_workers=8)
    ds.append(val_loader)
    ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds

    return ds


def smote(data, targets, n_class, n_max):
    aug_data = []
    aug_label = []

    for k in range(1, n_class):
        indices = np.where(targets == k)[0]
        class_data = data[indices]
        class_len = len(indices)
        class_dist = np.zeros((class_len, class_len))

        # Augmentation with SMOTE ( k-nearest )
        if smote:
            for i in range(class_len):
                for j in range(class_len):
                    class_dist[i, j] = np.linalg.norm(class_data[i] - class_data[j])
            sorted_idx = np.argsort(class_dist)

            for i in range(n_max - class_len):
                lam = nr.uniform(0, 1)
                row_idx = i % class_len
                col_idx = int((i - row_idx) / class_len) % (class_len - 1)
                new_data = np.round(
                    lam * class_data[row_idx] + (1 - lam) * class_data[sorted_idx[row_idx, 1 + col_idx]])

                aug_data.append(new_data.astype('uint8'))
                aug_label.append(k)

    return np.array(aug_data), np.array(aug_label)


def get_smote(dataset,  num_sample_per_class, batch_size, TF_train, TF_test):
    print("Building CV {} data loader with {} workers".format(dataset, 8))
    ds = []

    if dataset == 'cifar10':
        dataset_ = datasets.CIFAR10
        num_test_samples = num_test_samples_cifar10
    elif dataset == 'cifar100':
        dataset_ = datasets.CIFAR100
        num_test_samples = num_test_samples_cifar100
    else:
        raise NotImplementedError()

    train_cifar = dataset_(root=DATA_ROOT, train=True, download=False, transform=TF_train)
    targets = np.array(train_cifar.targets)
    classes, class_counts = np.unique(targets, return_counts=True)
    nb_classes = len(classes)

    imbal_class_counts = [int(i) for i in num_sample_per_class]
    class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]

    imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
    imbal_class_indices = np.hstack(imbal_class_indices)

    train_cifar.targets = targets[imbal_class_indices]
    train_cifar.data = train_cifar.data[imbal_class_indices]

    assert len(train_cifar.targets) == len(train_cifar.data)

    class_max = max(num_sample_per_class)
    aug_data, aug_label = smote(train_cifar.data, train_cifar.targets, nb_classes, class_max)

    train_cifar.targets = np.concatenate((train_cifar.targets, aug_label), axis=0)
    train_cifar.data = np.concatenate((train_cifar.data, aug_data), axis=0)

    print("Augmented data num = {}".format(len(aug_label)))
    print(train_cifar.data.shape)

    train_in_loader = torch.utils.data.DataLoader(train_cifar, batch_size=batch_size, shuffle=True, num_workers=8)
    ds.append(train_in_loader)

    test_cifar = dataset_(root=DATA_ROOT, train=False, download=False, transform=TF_test)
    val_idx, test_idx = get_val_test_data(test_cifar, num_test_samples)
    val_loader = torch.utils.data.DataLoader(test_cifar, batch_size=100,
                                             sampler=SubsetRandomSampler(val_idx), num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_cifar, batch_size=100,
                                              sampler=SubsetRandomSampler(test_idx), num_workers=8)
    ds.append(val_loader)
    ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds

    return ds

# Data
print('==> Preparing data: %s' % DATASET)
if DATASET == 'cifar100':
    N_CLASSES = 100
    N_SAMPLES = 500
    mean = torch.tensor([0.5071, 0.4867, 0.4408])
    std = torch.tensor([0.2675, 0.2565, 0.2761])
elif DATASET == 'cifar10':
    N_CLASSES = 10
    N_SAMPLES = 5000
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
else:
    raise NotImplementedError()

normalizer = InputNormalize(mean, std).to(device)

if 'cifar' in DATASET:
    if ARGS.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
else:
    raise NotImplementedError()

# Data Loader ##
N_CLASSES = ARGS.num_classes
N_SAMPLES_PER_CLASS_BASE = [int(N_SAMPLES)] * N_CLASSES
if ARGS.imb_type == 'exp':
    N_SAMPLES_PER_CLASS_BASE = make_longtailed_imb(N_SAMPLES, N_CLASSES, ARGS.ratio)
elif ARGS.imb_type == 'step':
    for i in range(ARGS.imb_start, N_CLASSES):
        N_SAMPLES_PER_CLASS_BASE[i] = int(N_SAMPLES * (1 / ARGS.ratio))

N_SAMPLES_PER_CLASS_BASE = tuple(N_SAMPLES_PER_CLASS_BASE)
print(N_SAMPLES_PER_CLASS_BASE)

train_loader2, _, _ = get_oversampled(DATASET, N_SAMPLES_PER_CLASS_BASE, BATCH_SIZE, transform_train, transform_test)
train_loader, val_loader, test_loader = get_imbalanced(DATASET, N_SAMPLES_PER_CLASS_BASE, BATCH_SIZE,
                                                    transform_train, transform_test)

## To apply effective number for over-sampling or cost-sensitive ##
if ARGS.over and ARGS.effect_over:
    _beta = ARGS.eff_beta
    effective_num = 1.0 - np.power(_beta, N_SAMPLES_PER_CLASS_BASE)
    N_SAMPLES_PER_CLASS = tuple(np.array(effective_num) / (1 - _beta))
    print(N_SAMPLES_PER_CLASS)
else:
    N_SAMPLES_PER_CLASS = N_SAMPLES_PER_CLASS_BASE
N_SAMPLES_PER_CLASS_T = torch.Tensor(N_SAMPLES_PER_CLASS).to(device)