import numpy as np
import torch
import torch.nn as nn
from .trainer import Trainer
from imbalanceddl.utils.utils import AverageMeter
from imbalanceddl.utils.metrics import accuracy
from imbalanceddl.utils.deep_smote_data_loader import get_balanced_deep_smote
from torchmetrics import F1Score
from torchmetrics.functional import precision_recall


class DeepSMOTETrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_criterion(self):
        if self.strategy == 'Deep_SMOTE':
            per_cls_weights = None
            print("=> CE Loss with Per Class Weight = {}".format(
                per_cls_weights))
            self.criterion = nn.CrossEntropyLoss(weight=per_cls_weights,
                                                 reduction='none').cuda(
                                                     self.cfg.gpu)
        else:
            raise ValueError("[Warning] Strategy is not supported !")

    def train_epoch(self):
        # Record
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        f1 = F1Score(num_classes=self.cfg.num_classes).to(self.cfg.gpu)

        # for confusion matrix
        all_preds = list()
        all_targets = list()
        all_f1_scores = []

        all_precisions = []
        all_recalls = []

        # switch to train mode
        self.model.train()

        for i, (_input, target) in enumerate(self.train_loader):

            if self.cfg.gpu is not None:
                _input = _input.cuda(self.cfg.gpu, non_blocking=True)
                target = target.cuda(self.cfg.gpu, non_blocking=True)

            # print("=> ERM training")
            out, _ = self.model(_input)
            # out = self.model(_input)
            loss = self.criterion(out, target).mean()
            acc1, acc5 = accuracy(out, target, topk=(1, 5))

            _, pred = torch.max(out, 1)
            F1_value = f1(pred, target)
            precision_value, recall_value = precision_recall(pred, target, average='macro', num_classes=self.cfg.num_classes)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_f1_scores.append(F1_value.cpu().numpy())
            all_precisions.append(precision_value.cpu().numpy())
            all_recalls.append(recall_value.cpu().numpy())

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
        
        all_f1_scores = np.array(all_f1_scores)
        f1=np.mean(all_f1_scores)
        print("-----------F1_Score of training dataset: {:.4f}% ------------".format(f1*100))

        all_precisions = np.array(all_precisions)
        all_recalls = np.array(all_recalls)
        precision =np.mean(all_precisions)
        recall =np.mean(all_recalls)
        f1_precision_recall = 2*precision*recall / (precision + recall)
        print("----------F1_Mixed_by_Ha of training dataset: {:.4f}% ----------".format(f1_precision_recall*100))

    def train_one_epoch(self):
        # import pdb
        # pdb.set_trace()
        if self.epoch == self.cfg.warm:
            print("----------Applying over sampling with Deep SMOTE-------------")
            train_loader2 = get_balanced_deep_smote(self.cfg.dataset, self.cfg.batch_size, self.cfg.imb_type, self.cfg.imb_factor, num_workers=8)
            self.train_loader = train_loader2

        ## Training ( ARGS.warm is used for deferred re-balancing ) ##
        if self.epoch >= self.cfg.warm:
            self.train_epoch()
        else:   
            self.train_epoch()