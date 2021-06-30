import abc
import os
import torch
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from imbalanceddl.utils.metrics import shot_acc
import numpy as np


class BaseTrainer(metaclass=abc.ABCMeta):
    """Base trainer for Deep Imbalanced Learning

    A trainer that will be learning with imbalanced data based on
    user-selected strategy.
    """
    def __init__(self, cfg, dataset, **kwargs):
        self.cfg = cfg
        self._dataset = dataset
        self._parse_train_val(dataset)
        self._prepare_logger()

    @property
    def dataset(self):
        """The Dataset object that is used for training"""
        return self._dataset

    @abc.abstractmethod
    def get_criterion(self):
        """Get criterion (loss function) when training

        Sub classes need to implement this method
        """
        return NotImplemented

    @abc.abstractmethod
    def train_one_epoch(self):
        """Main training strategy

        Sub classes need to implement this method
        """
        return NotImplemented

    def _parse_train_val(self, dataset):
        """Parse training and validation dataset

        Prepare trainining dataset, training dataloader, validation dataset,
        and validation dataloader.

        Note that we are training in imbalanced dataset, and evaluating in
        balanced dataset.
        """
        self.train_dataset, self.val_dataset = dataset.train_val_sets
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.workers,
            pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=100,
            shuffle=False,
            num_workers=self.cfg.workers,
            pin_memory=True)

    def _prepare_logger(self):
        """Logger for records

        Prepare logger for recording training and testing results
        and a tensorboard writer for visualization.
        """
        print("=> Preparing logger and tensorboard writer !")
        self.log_training = open(
            os.path.join(self.cfg.root_log, self.cfg.store_name,
                         'log_train.csv'), 'w')
        self.log_testing = open(
            os.path.join(self.cfg.root_log, self.cfg.store_name,
                         'log_test.csv'), 'w')
        self.tf_writer = SummaryWriter(
            log_dir=os.path.join(self.cfg.root_log, self.cfg.store_name))

        with open(
                os.path.join(self.cfg.root_log, self.cfg.store_name,
                             'args.txt'), 'w') as f:
            f.write(str(self.cfg))

    def compute_metrics_and_record(self,
                                   all_preds,
                                   all_targets,
                                   losses,
                                   top1,
                                   top5,
                                   flag='Training'):
        """Responsible for computing metrics and prepare string for logger"""
        if flag == 'Training':
            log = self.log_training
        else:
            log = self.log_testing

        if self.cfg.dataset == 'cifar100' or self.cfg.dataset == 'tiny200':
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            many_acc, median_acc, low_acc = shot_acc(self.cfg,
                                                     all_preds,
                                                     all_targets,
                                                     self.train_dataset,
                                                     acc_per_cls=False)
            group_acc = np.array([many_acc, median_acc, low_acc])
            # Print Format
            group_acc_string = '%s Group Acc: %s' % (flag, (np.array2string(
                group_acc,
                separator=',',
                formatter={'float_kind': lambda x: "%.3f" % x})))
            print(group_acc_string)
        else:
            group_acc = None
            group_acc_string = None

        # metrics (recall)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        # overall epoch output
        epoch_output = (
            '{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} \
            Loss {loss.avg:.5f}'.format(flag=flag,
                                        top1=top1,
                                        top5=top5,
                                        loss=losses))
        # per class output
        cls_acc_string = '%s Class Recall: %s' % (flag, (np.array2string(
            cls_acc,
            separator=',',
            formatter={'float_kind': lambda x: "%.3f" % x})))
        print(epoch_output)
        print(cls_acc_string)

        # if eval with best model, just return
        if self.cfg.best_model is not None:
            return cls_acc_string

        self.log_and_tf(epoch_output,
                        cls_acc,
                        cls_acc_string,
                        losses,
                        top1,
                        top5,
                        log,
                        group_acc=group_acc,
                        group_acc_string=group_acc_string,
                        flag=flag)

    def log_and_tf(self,
                   epoch_output,
                   cls_acc,
                   cls_acc_string,
                   losses,
                   top1,
                   top5,
                   log,
                   group_acc=None,
                   group_acc_string=None,
                   flag=None):
        """Responsible for recording logger and tensorboardX"""
        log.write(epoch_output + '\n')
        log.write(cls_acc_string + '\n')

        if group_acc_string is not None:
            log.write(group_acc_string + '\n')
        log.write('\n')
        log.flush()

        # TF
        if group_acc_string is not None:
            if flag == 'Training':
                self.tf_writer.add_scalars(
                    'acc/train_' + 'group_acc',
                    {str(i): x
                     for i, x in enumerate(group_acc)}, self.epoch)
            else:
                self.tf_writer.add_scalars(
                    'acc/test_' + 'group_acc',
                    {str(i): x
                     for i, x in enumerate(group_acc)}, self.epoch)

        else:
            if flag == 'Training':
                self.tf_writer.add_scalars(
                    'acc/train_' + 'cls_recall',
                    {str(i): x
                     for i, x in enumerate(cls_acc)}, self.epoch)
            else:
                self.tf_writer.add_scalars(
                    'acc/test_' + 'cls_recall',
                    {str(i): x
                     for i, x in enumerate(cls_acc)}, self.epoch)
        if flag == 'Trainig':
            self.tf_writer.add_scalar('loss/train', losses.avg, self.epoch)
            self.tf_writer.add_scalar('acc/train_top1', top1.avg, self.epoch)
            self.tf_writer.add_scalar('acc/train_top5', top5.avg, self.epoch)
            self.tf_writer.add_scalar('lr',
                                      self.optimizer.param_groups[-1]['lr'],
                                      self.epoch)
        else:
            self.tf_writer.add_scalar('loss/test_' + flag, losses.avg,
                                      self.epoch)
            self.tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg,
                                      self.epoch)
            self.tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg,
                                      self.epoch)
