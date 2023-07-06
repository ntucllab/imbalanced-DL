import os
import torch
import torch.optim as optim
import numpy as np
from imbalanceddl.utils.utils import AverageMeter, save_checkpoint, collect_result
from imbalanceddl.utils.metrics import accuracy
from .base import BaseTrainer
from imbalanceddl.utils.m2m_utils import Logger
from torchmetrics import F1Score
from torchmetrics.functional import precision_recall

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
        self.img_num_per_cls = self.cfg.cls_num_list
        self.best_acc1 = 0.
        self.best_val_acc1 = 0.

        if self.cfg.strategy == "M2m":
            LOGFILE_BASE = f"S{self.cfg.seed}_{self.cfg.strategy}_" \
                f"L{self.cfg.lam}_W{self.cfg.warm}_" \
                f"E{self.cfg.step_size}_I{self.cfg.attack_iter}_" \
                f"{self.cfg.dataset}_R{int(1/self.cfg.imb_factor)}_{self.cfg.backbone}_G{self.cfg.gamma}_B{self.cfg.beta}"
                
            LOGNAME = 'Imbalance_' + LOGFILE_BASE
            self.logger = Logger(LOGNAME)
            self.LOGDIR = self.logger.logdir
            self.LOG_CSV = os.path.join(self.LOGDIR, f'log_{self.cfg.seed}.csv')
            self.LOG_CSV_HEADER = [
                'epoch', 'train loss', 'gen loss', 'train acc', 'gen_acc', 'prob_orig', 'prob_targ',
                'test loss', 'major test acc', 'neutral test acc', 'minor test acc', 'test acc', 'f1 score'
            ]

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
        # total 400 epochs scheme for testing phase -> should be removed then
        elif self.cfg.epochs == 400:
            epoch = self.epoch + 1
            if epoch <=5:
                lr = self.cfg.learning_rate * epoch / 5
            elif epoch > 320:
                lr = self.cfg.learning_rate * 0.01
            elif epoch > 250:
                lr = self.cfg.learning_rate * 0.1
            else:
                lr = self.cfg.learning_rate
        else:
            raise ValueError(
                "[Warning] Total epochs {} not supported !".format(
                    self.cfg.epochs))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    # Do and train M2m Strategy here
    def do_train_val_m2m(self):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = self.cfg.gpu
        self.logger.log('==> Building model: %s' % self.cfg.backbone)
        net = self.model
        net_seed = self.model
        # net = models.__dict__[self.cfg.backbone](self.cfg.num_classes)
        # net_seed = models.__dict__[self.cfg.backbone](self.cfg.num_classes)

        net, net_seed = net.to(device), net_seed.to(device)
        optimizer = optim.SGD(net.parameters(), lr=self.cfg.learning_rate, momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
        
        SUCCESS = torch.zeros(self.cfg.epochs, self.cfg.num_classes, 2)
        self.train_oversamples = []

        if self.cfg.over:
            # Stage 2: Train f model with synthetic dataset.
            for epoch in range(self.cfg.start_epoch, self.cfg.epochs):
                self.epoch = epoch
                self.train_one_epoch(net, net_seed, optimizer, SUCCESS)

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
                    'optimizer': self.optimizer.state_dict()
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
        f1 = F1Score(num_classes=self.cfg.num_classes).to(self.cfg.gpu)

        # switch to evaluate mode
        self.model.eval()

        all_preds = list()
        all_targets = list()
        all_f1_scores = []

        all_precisions = []
        all_recalls = []

        with torch.no_grad():
            for i, (_input, target) in enumerate(self.val_loader):

                _input = _input.cuda(self.cfg.gpu, non_blocking=True)
                target = target.cuda(self.cfg.gpu, non_blocking=True)

                # compute output
                output, _ = self.model(_input)
                # output = self.model(_input)
                loss = self.criterion(output, target).mean()

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), _input.size(0))
                top1.update(acc1[0], _input.size(0))
                top5.update(acc5[0], _input.size(0))

                _, pred = torch.max(output, 1)
                F1_value = f1(pred, target)
                precision_value, recall_value = precision_recall(pred, target, average='macro', num_classes=self.cfg.num_classes)

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_f1_scores.append(F1_value.cpu().numpy())
                all_precisions.append(precision_value.cpu().numpy())
                all_recalls.append(recall_value.cpu().numpy())

                if i % self.cfg.print_freq == 0:
                    output = ('Epoch: [{0}][{1}/{2}]\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                  self.epoch,
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

            all_f1_scores = np.array(all_f1_scores)
            f1=np.mean(all_f1_scores)
            print("==========F1_Score of TESTING dataset: {:.4f}% =============".format(f1*100))

        if cls_acc_string is not None:
            return top1.avg, cls_acc_string
        else:
            return top1.avg
