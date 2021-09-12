import numpy as np
import torch
import torch.nn as nn
from .trainer import Trainer

from libdeep.utils.utils import AverageMeter
from libdeep.utils.metrics import accuracy


def get_k(n1, n2, f):
    k1 = pow(n1, f)
    k2 = pow(n2, f)
    return k1, k2


def get_lambda(x, k1, k2):
    lambda_lower = 0.0
    t_lower = 1.0
    lambda_upper = 1.0
    t_upper = 0.0
    lambda_middle = k1 / (k1 + k2)
    t_middle = 0.5
    if x < lambda_middle:
        lambda_target = ((-t_middle) *
                         (x - lambda_lower) / lambda_middle) + t_lower
    elif x > lambda_middle:
        lambda_target = ((x - lambda_upper) * (t_middle - t_upper) /
                         (lambda_middle - lambda_upper))
    else:
        raise ValueError("[-] Check Boundary Case !")
    return lambda_target


def mamix_data(x, y, args, alpha=1.0):
    if alpha > 0:
        lam_x = np.random.beta(alpha, alpha)
    else:
        lam_x = 1

    cls_num_list = args.cls_num_list
    cls_num_list = torch.tensor(cls_num_list)

    batch_size = x.size()[0]
    # get the index from random permutation for mix x
    index = torch.randperm(batch_size)

    # check will store the pair chosen for mixup with each other [batch, 2]
    check = []
    for i, j in enumerate(index):
        check.append([cls_num_list[y[i]].item(), cls_num_list[y[j]].item()])
    check = torch.tensor(check)

    # Now, we are going to compute lam_y for every pair
    lam_y = list()
    for i in range(check.size()[0]):
        # temp1 = n_i; temp2 = n_j
        temp1 = check[i][0].item()
        temp2 = check[i][1].item()

        f = args.mamix_ratio
        k1, k2 = get_k(temp1, temp2, f)
        lam_t = get_lambda(lam_x, k1, k2)

        lam_y.append(lam_t)

    lam_y = torch.tensor(lam_y).cuda(args.gpu)

    mixed_x = (1 - lam_x) * x + lam_x * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam_x, lam_y


def mamix_criterion(criterion, pred, y_a, y_b, lam_y, args):
    loss = torch.mul(criterion(pred, y_a), lam_y) + torch.mul(
        criterion(pred, y_b), (1 - lam_y))

    return loss.mean()


class MAMixTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_criterion(self):
        if self.strategy == 'MAMix_DRW':
            if self.cfg.epochs == 300:
                idx = self.epoch // 250
            else:
                idx = self.epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], self.cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(
                self.cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(
                self.cfg.gpu)
            print("=> Per Class Weight = {}".format(per_cls_weights))
            self.criterion = nn.CrossEntropyLoss(weight=per_cls_weights,
                                                 reduction='none').cuda(
                                                     self.cfg.gpu)
        else:
            raise ValueError("[Warning] Strategy is not supported !")

    def train_one_epoch(self):
        # Record
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        # for confusion matrix
        all_preds = list()
        all_targets = list()

        # switch to train mode
        self.model.train()

        for i, (_input, target) in enumerate(self.train_loader):

            if self.cfg.gpu is not None:
                _input = _input.cuda(self.cfg.gpu, non_blocking=True)
                target = target.cuda(self.cfg.gpu, non_blocking=True)

            # MAMix Data
            _input_mix, target_a, target_b, lam_x, lam_y = mamix_data(
                _input, target, self.cfg)
            # Two kinds of output
            output_prec, _ = self.model(_input)
            output_mix, _ = self.model(_input_mix)
            # For Loss, we use mixed output
            loss = mamix_criterion(self.criterion, output_mix, target_a,
                                   target_b, lam_y, self.cfg)
            acc1, acc5 = accuracy(output_prec, target, topk=(1, 5))
            _, pred = torch.max(output_prec, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            # measure accuracy and record loss
            losses.update(loss.item(), _input.size(0))
            top1.update(acc1[0], _input.size(0))
            top5.update(acc5[0], _input.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.cfg.print_freq == 0:
                output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                              self.epoch,
                              i,
                              len(self.train_loader),
                              loss=losses,
                              top1=top1,
                              top5=top5,
                              lr=self.optimizer.param_groups[-1]['lr'] * 0.1))
                print(output)
                self.log_training.write(output + '\n')
                self.log_training.flush()

        self.compute_metrics_and_record(all_preds,
                                        all_targets,
                                        losses,
                                        top1,
                                        top5,
                                        flag='Training')
