from __future__ import print_function
import csv
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .trainer import Trainer
from tqdm import tqdm

from imbalanceddl.utils.m2m_utils import random_perturb, make_step, sum_t, classwise_loss, adjust_learning_rate, LDAMLoss, FocalLoss, InputNormalize

TEST_ACC = 0  # best test accuracy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def classwise_loss(outputs, targets):
    out_1hot = torch.zeros_like(outputs)
    out_1hot.scatter_(1, targets.view(-1, 1), 1)
    return (outputs * out_1hot).sum(1).mean()

class M2mTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalizer = self.get_normalize()

    def get_normalize(self):
        # Data
        print('==> Preparing data: %s' % self.cfg.dataset)
        if self.cfg.dataset == 'cifar100':
            mean = torch.tensor([0.5071, 0.4867, 0.4408])
            std = torch.tensor([0.2675, 0.2565, 0.2761])
        elif self.cfg.dataset == 'cifar10':
            mean = torch.tensor([0.4914, 0.4822, 0.4465])
            std = torch.tensor([0.2023, 0.1994, 0.2010])
        elif self.cfg.dataset == 'svhn10':
            mean = torch.tensor([.5, .5, .5])
            std = torch.tensor([.5, .5, .5])
        elif self.cfg.dataset == 'tiny200':
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
        elif self.cfg.dataset == 'cinic10':
            mean = torch.tensor([0.47889522, 0.47227842, 0.43047404])
            std = torch.tensor([0.24205776, 0.23828046, 0.25874835])
        else:
            raise NotImplementedError()

        normalizer = InputNormalize(mean, std).to(device)
        return normalizer

    def get_criterion(self):
        if self.strategy == 'M2m':
            ## For Cost-Sensitive Learning ##
            if self.cfg.reweight and self.epoch >= self.cfg.warm:

                # Dont apply reweight for M2m generation method
                beta = self.cfg.eff_beta
                if beta < 1:
                    effective_num = 1.0 -np.power(beta, self.img_num_per_cls)
                    per_cls_weights = (1.0 - beta) / np.array(effective_num)
                else:
                    per_cls_weights = 1 / np.array(self.img_num_per_cls)
                
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.img_num_per_cls)
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(self.cfg.gpu)
                print("=> Per Class Weight = {}".format(per_cls_weights))
            else:
                per_cls_weights = torch.ones(self.cfg.num_classes).to(self.cfg.gpu)
                print("=> Per Class Weight = {}".format(per_cls_weights))

            ## Choos a loss function ##
            if self.cfg.loss_type == 'CE':
                self.criterion = nn.CrossEntropyLoss(weight=per_cls_weights, reduction='none').cuda(self.cfg.gpu)
            elif self.cfg.loss_type == 'Focal':
                self.criterion = FocalLoss(weight=per_cls_weights, gamma=0.9).cuda(self.cfg.gpu)
            elif self.cfg.loss_type == 'LDAM':
                self.criterion = LDAMLoss(cls_num_list=self.cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(self.cfg.gpu)
            else:
                raise ValueError("Wrong Loss Type")

        else:
            raise ValueError("[Warning] Strategy is not supported !")

    def train_gen_epoch(self, net_t, net_g, criterion, optimizer, data_loader, logger):
        net_t.train()
        net_g.eval()

        oth_loss, gen_loss = 0, 0
        correct_oth = 0
        correct_gen = 0
        total_oth, total_gen = 1e-6, 1e-6
        p_g_orig, p_g_targ = 0, 0
        t_success = torch.zeros(self.cfg.num_classes, 2)
        n_samples_per_cls = torch.Tensor(self.img_num_per_cls).to(device)
        
        for inputs, targets in tqdm(data_loader):
            # classes, class_counts = np.unique(targets, return_counts=True)
            # print(classes)
            # print(class_counts)
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(self.cfg.gpu), targets.to(self.cfg.gpu)

            # Set a generation target for current batch with re-sampling
            if self.cfg.imb_type != 'none':  # Imbalanced
                # Keep the sample with this probability
                if n_samples_per_cls[0] > n_samples_per_cls[1]:
                    num_gen_probs = n_samples_per_cls[0]
                else:
                    num_gen_probs = n_samples_per_cls[1]

                gen_probs = n_samples_per_cls[targets] / num_gen_probs
                gen_index = (1 - torch.bernoulli(gen_probs)).nonzero()    # Generation index
                gen_index = gen_index.view(-1)
                gen_targets = targets[gen_index]
            else:   # Balanced
                gen_index = torch.arange(batch_size).view(-1)
                gen_targets = torch.randint(self.cfg.num_classes, (batch_size,)).to(self.cfg.gpu).long()

            t_loss, g_loss, num_others, num_correct, num_gen, num_gen_correct, p_g_orig_batch, p_g_targ_batch, success \
                = self.train_net(net_t, net_g, criterion, optimizer, inputs, targets, gen_index, gen_targets)
            
            oth_loss += t_loss
            gen_loss += g_loss
            total_oth += num_others
            correct_oth += num_correct
            total_gen += num_gen
            correct_gen += num_gen_correct
            p_g_orig += p_g_orig_batch
            p_g_targ += p_g_targ_batch
            t_success += success

        res = {
            'train_loss': oth_loss / total_oth,
            'gen_loss': gen_loss / total_gen,
            'train_acc': 100. * correct_oth / total_oth,
            'gen_acc': 100. * correct_gen / total_gen,
            'p_g_orig': p_g_orig / total_gen,
            'p_g_targ': p_g_targ / total_gen,
            't_success': t_success
        }

        msg = 't_Loss: %.3f | g_Loss: %.3f | Acc: %.3f%% (%d/%d) | Acc_gen: %.3f%% (%d/%d) ' \
            '| Prob_orig: %.3f | Prob_targ: %.3f' % (
            res['train_loss'], res['gen_loss'],
            res['train_acc'], correct_oth, total_oth,
            res['gen_acc'], correct_gen, total_gen,
            res['p_g_orig'], res['p_g_targ']
        )
        if logger:
            logger.log(msg)
        else:
            print(msg)

        return res

    def train_epoch(self, net, criterion, optimizer, data_loader, logger=None):
        net.train()

        train_loss = 0
        correct = 0
        total = 0

        for inputs, targets in tqdm(data_loader):

            inputs, targets = inputs.to(self.cfg.gpu), targets.to(self.cfg.gpu)
            batch_size = inputs.size(0)

            outputs, _ = net(self.normalizer(inputs))
            # outputs, _ = net(inputs)
            loss = criterion(outputs, targets).mean()

            train_loss += loss.item() * batch_size
            predicted = outputs.max(1)[1]
            total += batch_size
            correct += sum_t(predicted.eq(targets))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        msg = 'Loss: %.3f| Acc: %.3f%% (%d/%d)' % \
            (train_loss / total, 100. * correct / total, correct, total)
        if logger:
            logger.log(msg)
        else:
            print(msg)

        return train_loss / total, 100. * correct / total
    
    def train_net(self, net, net_seed, criterion, optimizer_train, inputs_orig, targets_orig, gen_idx, gen_targets):
        batch_size = inputs_orig.size(0)

        inputs = inputs_orig.clone()
        targets = targets_orig.clone()

        ########################
        n_samples_per_cls = torch.Tensor(self.img_num_per_cls).to(device)
        bs = n_samples_per_cls[targets_orig].repeat(gen_idx.size(0), 1)
        gs = n_samples_per_cls[gen_targets].view(-1, 1)

        delta = F.relu(bs - gs)
        p_accept = 1 - self.cfg.beta ** delta
        mask_valid = (p_accept.sum(1) > 0)

        gen_idx = gen_idx[mask_valid]
        gen_targets = gen_targets[mask_valid]
        p_accept = p_accept[mask_valid]

        select_idx = torch.multinomial(p_accept, 1, replacement=True).view(-1)
        p_accept = p_accept.gather(1, select_idx.view(-1, 1)).view(-1)

        seed_targets = targets_orig[select_idx]
        seed_images = inputs_orig[select_idx]

        gen_inputs, correct_mask = self.generation(net_seed, net, seed_images, seed_targets, gen_targets, p_accept,
                                            self.cfg.gamma, self.cfg.lam, self.cfg.step_size, True, self.cfg.attack_iter)

        ########################
        # Only change the correctly generated samples
        num_gen = sum_t(correct_mask)
        num_others = batch_size - num_gen

        gen_c_idx = gen_idx[correct_mask]
        others_mask = torch.ones(batch_size, dtype=torch.bool, device=self.cfg.gpu)
        others_mask[gen_c_idx] = 0
        others_idx = others_mask.nonzero().view(-1)

        if num_gen > 0:
            gen_inputs_c = gen_inputs[correct_mask]
            gen_targets_c = gen_targets[correct_mask]

            inputs[gen_c_idx] = gen_inputs_c
            targets[gen_c_idx] = gen_targets_c

        outputs, _ = net(self.normalizer(inputs))
        # outputs, _ = net(inputs)
        loss = criterion(outputs, targets)

        optimizer_train.zero_grad()
        loss.mean().backward()
        optimizer_train.step()

        # For logging the training
        oth_loss_total = sum_t(loss[others_idx])
        gen_loss_total = sum_t(loss[gen_c_idx])

        _, predicted = torch.max(outputs[others_idx].data, 1)
        num_correct_oth = sum_t(predicted.eq(targets[others_idx]))

        num_correct_gen, p_g_orig, p_g_targ = 0, 0, 0
        success = torch.zeros(self.cfg.num_classes, 2)

        if num_gen > 0:
            _, predicted_gen = torch.max(outputs[gen_c_idx].data, 1)
            num_correct_gen = sum_t(predicted_gen.eq(targets[gen_c_idx]))
            probs = torch.softmax(outputs[gen_c_idx], 1).data

            p_g_orig = probs.gather(1, seed_targets[correct_mask].view(-1, 1))
            p_g_orig = sum_t(p_g_orig)

            p_g_targ = probs.gather(1, gen_targets_c.view(-1, 1))
            p_g_targ = sum_t(p_g_targ)

        for i in range(self.cfg.num_classes):
            if num_gen > 0:
                success[i, 0] = sum_t(gen_targets_c == i)
            success[i, 1] = sum_t(gen_targets == i)

        return oth_loss_total, gen_loss_total, num_others, num_correct_oth, num_gen, num_correct_gen, p_g_orig, p_g_targ, success

    
    def generation(self, net_seed, net, inputs, seed_targets, targets, p_accept,
                gamma, lam, step_size, random_start=True, max_iter=10):  # model_r = model train, model_g = model generation
        net_seed.eval()
        net.eval()
        criterion = nn.CrossEntropyLoss()

        if random_start:
            random_noise = random_perturb(inputs, 'l2', 0.5)
            inputs = torch.clamp(inputs + random_noise, 0, 1)

        for _ in range(max_iter):
            inputs = inputs.clone().detach().requires_grad_(True)
            outputs_g, _ = net_seed(self.normalizer(inputs))
            outputs_r, _ = net(self.normalizer(inputs))
            # outputs_g, _ = net_seed(inputs)
            # outputs_r, _ = net(inputs)

            loss = criterion(outputs_g, targets) + lam * classwise_loss(outputs_r, seed_targets)
            grad, = torch.autograd.grad(loss, [inputs])

            inputs = inputs - make_step(grad, 'l2', step_size)
            inputs = torch.clamp(inputs, 0, 1)

        inputs = inputs.detach()

        outputs_g, _ = net_seed(self.normalizer(inputs))
        # outputs_g, _ = net_seed(inputs)

        one_hot = torch.zeros_like(outputs_g)
        one_hot.scatter_(1, targets.view(-1, 1), 1)
        probs_g = torch.softmax(outputs_g, dim=1)[one_hot.to(torch.bool)]

        correct = (probs_g >= gamma) * torch.bernoulli(p_accept).byte().to(self.cfg.gpu)
        net.train()

        return inputs, correct

    def save_m2m_checkpoint(self, acc, model, optim, epoch, index=False):
    # Save checkpoint.
        print('Saving..')

        if isinstance(model, nn.DataParallel):
            model = model.module

        state = {
            'net': model.state_dict(),
            'optimizer': optim.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'rng_state': torch.get_rng_state()
        }

        if index:
            ckpt_name = 'ckpt_epoch' + str(epoch) + str(self.cfg.strategy) + '_' + str(self.cfg.seed) + '_' + str(self.cfg.dataset) + '.t7'
        else:
            ckpt_name = 'ckpt_' + str(self.cfg.strategy) + '_' + str(self.cfg.seed) + '_' + str(self.cfg.dataset) + '.t7'

        ckpt_path = os.path.join(self.LOGDIR, ckpt_name)
        torch.save(state, ckpt_path)
    
    def load_best_first_200_epochs(self, net, net_seed, optimizer):
        LOGFILE_BASE = f"S{self.cfg.seed}_{self.cfg.strategy}_" \
        f"L{self.cfg.lam}_W{self.cfg.warm}_" \
        f"E{self.cfg.step_size}_I{self.cfg.attack_iter}_" \
        f"{self.cfg.dataset}_R{int(1/self.cfg.imb_factor)}_{self.cfg.backbone}_G{self.cfg.gamma}_B{self.cfg.beta}"
        LOGNAME = 'Imbalance_' + LOGFILE_BASE
        ckpt_name = 'ckpt_' + str(self.cfg.strategy) + '_' + str(self.cfg.seed) + '_' + str(self.cfg.dataset) + '.t7'
        ckpt_g = f'./logs/{LOGNAME}/{ckpt_name}'

        if self.cfg.net_both is not None:
            ckpt_t = torch.load(self.cfg.net_both)
            net.load_state_dict(ckpt_t['net'])
            optimizer.load_state_dict(ckpt_t['optimizer'])
            START_EPOCH = ckpt_t['epoch'] + 1
            net_seed.load_state_dict(ckpt_t['net2'])
        else:
            if self.cfg.net_t is not None:
                ckpt_t = torch.load(self.cfg.net_t)
                net.load_state_dict(ckpt_t['net'])
                optimizer.load_state_dict(ckpt_t['optimizer'])
                START_EPOCH = ckpt_t['epoch'] + 1

            if self.cfg.net_g is not None:
                ckpt_g = self.cfg.net_g
                print(ckpt_g)
                ckpt_g = torch.load(ckpt_g)
                net_seed.load_state_dict(ckpt_g['net'])
            else:
                print("==> [Loading the best model of first 200 epochs]")
                print(ckpt_g)
                ckpt_g = torch.load(ckpt_g)
                net_seed.load_state_dict(ckpt_g['net'])
    
    def evaluate(self, net, dataloader, logger=None):
        is_training = net.training
        net.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct, total = 0.0, 0.0
        major_correct, neutral_correct, minor_correct = 0.0, 0.0, 0.0
        major_total, neutral_total, minor_total = 0.0, 0.0, 0.0

        class_correct = torch.zeros(self.cfg.num_classes)
        class_total = torch.zeros(self.cfg.num_classes)

        for inputs, targets in dataloader:
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, _ = net(self.normalizer(inputs))
            # outputs, _ = net(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * batch_size
            predicted = outputs[:, :self.cfg.num_classes].max(1)[1]
            total += batch_size
            correct_mask = (predicted == targets)
            correct += sum_t(correct_mask)

            # For accuracy of minority / majority classes.
            major_mask = targets < (self.cfg.num_classes // 3)
            major_total += sum_t(major_mask)
            major_correct += sum_t(correct_mask * major_mask)

            minor_mask = targets >= (self.cfg.num_classes - (self.cfg.num_classes // 3))
            minor_total += sum_t(minor_mask)
            minor_correct += sum_t(correct_mask * minor_mask)

            neutral_mask = ~(major_mask + minor_mask)
            neutral_total += sum_t(neutral_mask)
            neutral_correct += sum_t(correct_mask * neutral_mask)

            for i in range(self.cfg.num_classes):
                class_mask = (targets == i)
                class_total[i] += sum_t(class_mask)
                class_correct[i] += sum_t(correct_mask * class_mask)

        if major_total == 0:
            major_acc = 0
        else:
            major_acc = 100. * major_correct / major_total

        if neutral_total == 0:
            neutral_acc = 0
        else:
            neutral_acc = 100. * neutral_correct / neutral_total
        
        if minor_total == 0:
            minor_acc = 0
        else:
            minor_acc = 100. * minor_correct / minor_total

        results = {
            'loss': total_loss / total,
            'acc': 100. * correct / total,
            'major_acc': major_acc,
            'neutral_acc': neutral_acc,
            'minor_acc': minor_acc,
            'class_acc': 100. * class_correct / class_total,
        }

        msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Major_ACC: %.3f%% | Neutral_ACC: %.3f%% | Minor ACC: %.3f%% ' % \
            (
                results['loss'], results['acc'], correct, total,
                results['major_acc'], results['neutral_acc'], results['minor_acc']
            )
        if logger:
            logger.log(msg)
        else:
            print(msg)

        net.train(is_training)
        return results

    def train_one_epoch(self, net, net_seed, optimizer, SUCCESS):
        
        if (self.epoch == 0):
            if self.cfg.dataset == 'cifar100':  # Update below later
                # To avoid it will process all datasets when running M2m method frist time, just import specific dataset.
                from imbalanceddl.dataset.m2m_imbalance_cifar100 import cifar100_train_val_oversamples
                train_in_loader, val_in_loader, train_oversamples_loader = cifar100_train_val_oversamples(self.cfg.cifar_root, self.cfg.batch_size)
                self.train_loader, self.val_loader, self.train_oversamples = train_in_loader, val_in_loader, train_oversamples_loader
            elif self.cfg.dataset == 'cifar10':
                # To avoid it will process all datasets when running M2m method frist time, just import specific dataset.
                from imbalanceddl.dataset.m2m_imbalance_cifar10 import cifar10_train_val_oversamples
                train_in_loader, val_in_loader, train_oversamples_loader = cifar10_train_val_oversamples(self.cfg.cifar_root, self.cfg.batch_size)
                self.train_loader, self.val_loader, self.train_oversamples = train_in_loader, val_in_loader, train_oversamples_loader
            elif self.cfg.dataset == 'svhn10':
                # To avoid it will process all datasets when running M2m method frist time, just import specific dataset.
                from imbalanceddl.dataset.m2m_imbalance_svhn import svhn_train_val_oversamples
                train_in_loader, val_in_loader, train_oversamples_loader = svhn_train_val_oversamples(self.cfg.svhn_root, self.cfg.batch_size)
                self.train_loader, self.val_loader, self.train_oversamples = train_in_loader, val_in_loader, train_oversamples_loader
            elif self.cfg.dataset == 'tiny200':
                # To avoid it will process all datasets when running M2m method frist time, just import specific dataset.
                from imbalanceddl.dataset.m2m_imbalance_tinyimagenet import tinyimagenet_train_val_oversamples
                train_in_loader, val_in_loader, train_oversamples_loader = tinyimagenet_train_val_oversamples(self.cfg.tiny_root, self.cfg.batch_size)
                self.train_loader, self.val_loader, self.train_oversamples = train_in_loader, val_in_loader, train_oversamples_loader
            elif self.cfg.dataset == 'cinic10':
                # To avoid it will process all datasets when running M2m method frist time, just import specific dataset.
                from imbalanceddl.dataset.m2m_imbalance_cinic import cinic_train_val_oversamples
                train_in_loader, val_in_loader, train_oversamples_loader = cinic_train_val_oversamples(self.cfg.cinic_root, self.cfg.batch_size)
                self.train_loader, self.val_loader, self.train_oversamples = train_in_loader, val_in_loader, train_oversamples_loader
            else:
                raise NotImplementedError()

        # Load checkpoint.
        if self.epoch == self.cfg.warm and self.cfg.resume:
            self.logger.log('==> Resuming from checkpoint..')
            self.load_best_first_200_epochs(net, net_seed, optimizer)

        self.logger.log(' * Epoch %d: %s' % (self.epoch, self.LOGDIR))

        adjust_learning_rate(optimizer, self.cfg.learning_rate, self.epoch, self.cfg.epochs)

        ## For Cost-Sensitive Learning ##
        if self.cfg.reweight and self.epoch >= self.cfg.warm:
            beta = self.cfg.eff_beta
            if beta < 1:
                effective_num = 1.0 -np.power(beta, self.img_num_per_cls)
                per_cls_weights = (1.0 - beta) / np.array(effective_num)
            else:
                per_cls_weights = 1 / np.array(self.img_num_per_cls)
            
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.img_num_per_cls)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(self.cfg.gpu)
            print("=> Per Class Weight = {}".format(per_cls_weights))
        else:
            per_cls_weights = torch.ones(self.cfg.num_classes).to(self.cfg.gpu)
            print("=> Per Class Weight = {}".format(per_cls_weights))

        ## Choos a loss function ##
        if self.cfg.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights, reduction='none').to(self.cfg.gpu)
        elif self.cfg.loss_type == 'Focal':
            criterion = FocalLoss(weight=per_cls_weights, gamma=self.cfg.gamma, reduction='none').to(self.cfg.gpu)
        elif self.cfg.loss_type == 'LDAM':
            criterion = LDAMLoss(cls_num_list=self.img_num_per_cls, max_m=0.5, s=30, weight=per_cls_weights,
                                 reduction='none').to(self.cfg.gpu)
        else:
            raise ValueError("Wrong Loss Type")

        ## Training ( ARGS.warm is used for deferred re-balancing ) ##
        if self.epoch >= self.cfg.warm and self.cfg.gen:
            train_stats = self.train_gen_epoch(net, net_seed, criterion, optimizer, self.train_oversamples, self.logger)
            SUCCESS[self.epoch, :, :] = train_stats['t_success'].float()
            self.logger.log(SUCCESS[self.epoch, -self.cfg.num_classes:, :])
            np.save(self.LOGDIR + '/success.npy', SUCCESS.cpu().numpy())

        else:
            train_loss, train_acc = self.train_epoch(net, criterion, optimizer, self.train_loader, self.logger)
            train_stats = {'train_loss': train_loss, 'train_acc': train_acc}
            if self.epoch == self.cfg.warm:
                self.save_m2m_checkpoint(train_acc, net, optimizer, self.epoch, True)

        ## Evaluation ##
        val_stats = self.evaluate(net, self.val_loader, logger=self.logger)
        val_acc = val_stats['acc']

        if val_acc >= self.best_val_acc1:
            self.best_val_acc1 = val_acc
            TEST_ACC_CLASS = val_stats['class_acc']

            self.logger.log("==========Class-wise test performance (avg:{:.3f}%)==========".format(self.best_val_acc1))
            np.save(self.LOGDIR + '/classwise_acc.npy', TEST_ACC_CLASS.cpu())

            # Saving the best accuracy model
            self.save_m2m_checkpoint(TEST_ACC, net, optimizer, self.epoch)

        def _convert_scala(x):
            if hasattr(x, 'item'):
                x = x.item()
            return x
        
        log_tr = ['train_loss', 'gen_loss', 'train_acc', 'gen_acc', 'p_g_orig', 'p_g_targ']
        log_te = ['loss', 'major_acc', 'neutral_acc', 'minor_acc', 'acc', 'f1_score']

        log_vector = [self.epoch] + [train_stats.get(k, 0) for k in log_tr] + [val_stats.get(k, 0) for k in log_te]
        log_vector = list(map(_convert_scala, log_vector))

        with open(self.LOG_CSV, 'a') as f:
            logwriter = csv.writer(f, delimiter=',')
            logwriter.writerow(log_vector)
        
        # Print the best accuracy
        self.logger.log(' * %s' % self.LOGDIR)
        self.logger.log("Best Accuracy of Test: {:.3f}%".format(self.best_val_acc1))