import random
import numpy as np
import torch
import shutil
import os
import torch

def fix_all_seed(seed):
    """
    Usage: Fix seed for training

    Args: seed: int

    Return: None

    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        print('=> You have chosen to seed {} for training. '
              'This will turn on the CUDNN deterministic setting, '
              'which can slow down your training considerably! '
              'You may see unexpected behavior when restarting '
              'from checkpoints.'.format(seed))
    else:
        print("=> Not use Seed Training !")


def prepare_folders(args):
    """
    Usage: Prepare folders for training log

    Args: args for setting

    Return: None

    """
    folders_util = [
        args.root_log, args.root_model,
        os.path.join(args.root_log, args.store_name),
        os.path.join(args.root_model, args.store_name)
    ]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


def prepare_store_name(args):
    """
    Usage: Prepare store name for each experiment

    Args: args for setting

    Return: None

    """
    if args.strategy == 'Mixup_DRW':
        args.store_name = '_'.join([
            args.dataset, args.imb_type,
            str(args.imb_factor), args.strategy,
            str(args.epochs),
            str(args.seed)
        ])
    elif args.strategy == 'Remix_DRW':
        args.store_name = '_'.join([
            args.dataset, args.imb_type,
            str(args.imb_factor), args.strategy,
            str(args.k_majority),
            str(args.tau),
            str(args.epochs),
            str(args.seed)
        ])
    elif args.strategy == 'MAMix_DRW':
        args.store_name = '_'.join([
            args.dataset, args.imb_type,
            str(args.imb_factor), args.strategy,
            str(args.mamix_ratio),
            str(args.epochs),
            str(args.seed)
        ])
    elif args.strategy == 'M2m_DRW':
        args.store_name = '_'.join([
            args.dataset, args.imb_type,
            str(args.imb_factor), args.strategy,
            str(args.loss_type),
            str(args.epochs),
            str(args.seed)
        ])
    else:
        args.store_name = '_'.join([
            args.dataset, args.imb_type,
            str(args.imb_factor), args.strategy,
            str(args.epochs),
            str(args.seed)
        ])
    return


def save_checkpoint(args, state, is_best, epoch):
    """
    Save modle checkpoint

    """
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def collect_result(args, output_best):
    """
    Collect quick results for performance check

    """
    fname = "_seed_result.txt"
    store_name_list = args.store_name.split("_")
    exclude_seed_store_name = "_".join(store_name_list[:-1])
    fname = exclude_seed_store_name + fname
    print("[Recording] {}".format(fname))
    with open(fname, "a") as f:
        f.write(
            "Seed = {} | Dataset = {} | Strategy = {} | Init Lr = {} | Epoch = {} | ImbType = {} | ImbFactor = {} | MAMix Ratio = {} | Best acc = {} \n"
            .format(
                args.seed, args.dataset, args.strategy, args.lr, args.epochs,
                args.imb_type, str(args.imb_factor),
                str(args.mamix_ratio) if args.strategy == 'MAMix_DRW' else 'None',
                output_best))

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
