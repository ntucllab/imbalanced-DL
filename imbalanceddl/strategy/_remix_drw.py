import numpy as np
import torch
import torch.nn as nn
from .trainer import Trainer

from imbalanceddl.utils.utils import AverageMeter
from imbalanceddl.utils.metrics import accuracy


def remix_data(x, y, args, alpha=1.0):
    '''
    Returns mixed inputs, pairs of targets, and lambda_x, lambda_y
    *Args*
    k: hyper parameter of k-majority
    tau: hyper parameter
    where in original paper they suggested to use k = 3, and tau = 0.5
    Here, lambda_y is defined in the original paper of remix, where there
    are three cases of lambda_y as the following:
    (a). lambda_y = 0
    (b). lambda_y = 1
    (c). lambda_y = lambda_x
    '''
    if alpha > 0:
        lam_x = np.random.beta(alpha, alpha)
    else:
        lam_x = 1

    # two hyper parameters as Remix suggested, k = 3; \tau = 0.5
    K = args.k_majority
    tau = args.tau

    cls_num_list = args.cls_num_list
    cls_num_list = torch.tensor(cls_num_list)

    batch_size = x.size()[0]

    # get the index from random permutation for mix x
    index = torch.randperm(batch_size)

    # check list stored pairs of image index where one mixup with the other
    check = []
    for i, j in enumerate(index):
        check.append([cls_num_list[y[i]].item(), cls_num_list[y[j]].item()])
    check = torch.tensor(check)
    lam_y = []
    for i in range(check.size()[0]):
        # temp1 = n_i; temp2 = n_j
        temp1 = check[i][0]
        temp2 = check[i][1]

        if (temp1 / temp2) >= K and lam_x < tau:
            lam_y.append(0)
        elif (temp1 / temp2) <= (1 / K) and (1 - lam_x) < tau:
            lam_y.append(1)
        else:
            lam_y.append(lam_x)

    lam_y = torch.tensor(lam_y).cuda(args.gpu)
    mixed_x = lam_x * x + (1 - lam_x) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam_x, lam_y


def remix_criterion(criterion, pred, y_a, y_b, lam_y, args):
    """
    In Remix, the lambda for mixing label is different from original mixup.
    """
    # for each y, we calculated its loss individually with their respective
    # lambda_y
    loss = torch.mul(criterion(pred, y_a), lam_y) + torch.mul(
        criterion(pred, y_b), (1 - lam_y))
    return loss.mean()


class RemixTrainer(Trainer):
    """Remix-DRW Trainer

    Reference
    ----------
    Remix: Rebalanced Mixup
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_criterion(self):
        if self.strategy == 'Remix_DRW':
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

            # Remix Data
            _input_mix, target_a, target_b, lam_x, lam_y = remix_data(
                _input, target, self.cfg)
            # Two kinds of output
            output_prec, _ = self.model(_input)
            output_mix, _ = self.model(_input_mix)
            # For Loss, we use mixed output
            loss = remix_criterion(self.criterion, output_mix, target_a,
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
