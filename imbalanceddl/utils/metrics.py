import numpy as np
import torch


def accuracy(output, target, topk=(1, )):
    """
    Function to compute topk accuracy for evaluation
    Common Usage is to use top 1 and top 5

    """

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # due to different PyTorch version, correct[:k].view(-1) may have
            # bug
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def shot_acc(args,
             preds,
             labels,
             train_data,
             many_shot_thr=100,
             low_shot_thr=20,
             acc_per_cls=False):
    """
    Function to compute many shot, median_shot, and low shot accuracy
    Typically used when the class number is huge, ex. CIFAR-100

    """

    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        if args.dataset == 'svhn':
            training_labels = np.array(train_data.labels).astype(int)
        else:
            training_labels = np.array(train_data.targets).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))

    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [
            c / cnt for c, cnt in zip(class_correct, test_class_count)
        ]
        return np.mean(many_shot), np.mean(median_shot), np.mean(
            low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)
