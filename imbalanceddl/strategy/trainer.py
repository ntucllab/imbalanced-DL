import os
import torch
import torch.optim as optim

from imbalanceddl.utils.utils import AverageMeter, save_checkpoint, collect_result
from imbalanceddl.utils.metrics import accuracy

from .base import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyward-only argument: 'model' !")
        else:
            print("=> Model = {}".format(self.model))
        self.strategy = kwargs.pop('strategy', None)
        if self.strategy is None:
            raise TypeError("__init__() missing required keyward-only \
                argument: 'strategy' !")
        else:
            print("=> Strategy = {}".format(self.strategy))
        self.optimizer = self._init_optimizer()
        self.cls_num_list = self.cfg.cls_num_list
        self.best_acc1 = 0.

    def get_criterion(self):
        return NotImplemented

    def train_one_epoch(self):
        return NotImplemented

    def _init_optimizer(self):
        if self.cfg.optimizer == 'sgd':
            print("=> Initialize optimizer {}".format(self.cfg.optimizer))
            optimizer = optim.SGD(self.model.parameters(),
                                  self.cfg.learning_rate,
                                  momentum=self.cfg.momentum,
                                  weight_decay=self.cfg.weight_decay)
            return optimizer
        else:
            raise ValueError("[Warning] Selected Optimizer not supported !")

    def adjust_learning_rate(self):
        """Sets the learning rate"""
        # total 200 epochs scheme
        if self.cfg.epochs == 200:
            epoch = self.epoch + 1
            if epoch <= 5:
                lr = self.cfg.learning_rate * epoch / 5
            elif epoch > 180:
                lr = self.cfg.learning_rate * 0.0001
            elif epoch > 160:
                lr = self.cfg.learning_rate * 0.01
            else:
                lr = self.cfg.learning_rate
        # total 300 epochs scheme
        elif self.cfg.epochs == 300:
            epoch = self.epoch + 1
            if epoch <= 5:
                lr = self.cfg.learning_rate * epoch / 5
            elif epoch > 250:
                lr = self.cfg.learning_rate * 0.01
            elif epoch > 150:
                lr = self.cfg.learning_rate * 0.1
            else:
                lr = self.cfg.learning_rate
        else:
            raise ValueError(
                "[Warning] Total epochs {} not supported !".format(
                    self.cfg.epochs))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def do_train_val(self):
        for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
            self.epoch = epoch

            # learning rate control
            self.adjust_learning_rate()

            # criterion
            self.get_criterion()
            assert self.criterion is not None, "No criterion !"
            self.train_one_epoch()
            acc1 = self.validate()
            # remember best acc@1 and save checkpoint
            is_best = acc1 > self.best_acc1
            self.best_acc1 = max(acc1, self.best_acc1)

            self.tf_writer.add_scalar('acc/test_top1_best', self.best_acc1,
                                      self.epoch)
            output_best = 'Best Prec@1: %.3f\n' % (self.best_acc1)
            print(output_best)
            self.log_testing.write(output_best + '\n')
            self.log_testing.flush()

            if epoch == self.cfg.epochs - 1:
                collect_result(self.cfg, output_best)

            save_checkpoint(
                self.cfg, {
                    'epoch': self.epoch + 1,
                    'backbone': self.cfg.backbone,
                    'classifier': self.cfg.classifier,
                    'state_dict': self.model.state_dict(),
                    'best_acc1': self.best_acc1,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, self.epoch)

    def eval_best_model(self):
        assert self.cfg.best_model is not None, "[Warning] Best Model \
                                                    must be loaded !"

        assert 'best' in self.cfg.best_model, "[Need Best Model]"

        if os.path.isfile(self.cfg.best_model):
            print("=> [Loading Best Model] '{}'".format(self.cfg.best_model))
            checkpoint = torch.load(self.cfg.best_model, map_location='cuda:0')
            self.epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if self.cfg.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(self.cfg.gpu)
            self.model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> [Loaded Best Model] '{}' (epoch {})".format(
                self.cfg.best_model, checkpoint['epoch']))
        else:
            print("=> [No Trained Model Path found at '{}'".format(
                self.cfg.best_model))
            raise ValueError("[Warning] No Trained Model Path Found !!!")

        self.get_criterion()
        assert self.criterion is not None, "No criterion !"
        acc1, cls_acc_string = self.validate()
        output_best = 'Best Prec@1: %.3f' % (acc1)
        print(output_best)
        print(cls_acc_string)
        print("[Done] with evaluating with best model of {}".format(
            self.cfg.best_model))
        return

    def validate(self):
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        # switch to evaluate mode
        self.model.eval()
        all_preds = list()
        all_targets = list()

        with torch.no_grad():
            for i, (_input, target) in enumerate(self.val_loader):

                _input = _input.cuda(self.cfg.gpu, non_blocking=True)
                target = target.cuda(self.cfg.gpu, non_blocking=True)

                # compute output
                output, _ = self.model(_input)
                loss = self.criterion(output, target).mean()

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), _input.size(0))
                top1.update(acc1[0], _input.size(0))
                top5.update(acc5[0], _input.size(0))

                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

                if i % self.cfg.print_freq == 0:
                    output = ('Test: [{0}/{1}]\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                  i,
                                  len(self.val_loader),
                                  loss=losses,
                                  top1=top1,
                                  top5=top5))
                    print(output)

            cls_acc_string = self.compute_metrics_and_record(all_preds,
                                            all_targets,
                                            losses,
                                            top1,
                                            top5,
                                            flag='Testing')

        if cls_acc_string is not None:
            return top1.avg, cls_acc_string
        else:
            return top1.avg
