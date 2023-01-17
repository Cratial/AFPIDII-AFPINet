import os
import time

import numpy as np
import pandas as pd
import torch
from torchvision.utils import make_grid

from base import BaseTrainer
from models.metric import MetricTracker
from utils import inf_loop
import copy
import wandb


class AFPINetTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, torch_args: dict, save_dir, resume, device, **kwargs):
        self.device = device
        super().__init__(torch_args, save_dir, **kwargs)

        if resume is not None:
            self._resume_checkpoint(resume, finetune=self.finetune)

        # data_loaders
        self.do_validation = self.valid_data_loaders['data'] is not None
        if self.len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_data_loaders['data'])
        else:
            # iteration-based training
            self.train_data_loaders['data'] = inf_loop(self.train_data_loaders['data'])
        self.log_step = int(np.sqrt(self.train_data_loaders['data'].batch_size))

        # losses
        self.criterion = self.losses['loss']

        # metrics
        keys_loss = ['loss']
        keys_iter = [m.__name__ for m in self.metrics_iter]
        keys_epoch = [m.__name__ for m in self.metrics_epoch]
        self.train_metrics = MetricTracker(keys_loss + keys_iter, keys_epoch, writer=self.writer)
        self.valid_metrics = MetricTracker(keys_loss + keys_iter, keys_epoch, writer=self.writer)
        
        # init wandb model watch
        # wandb.watch(self.models['model'])
        
#         wandb.save("trainers/cr_trainer.py", base_path="../")
        
        # init history logging of best/worst metrics
        self.best_val_metrics = {}
        for index, row in self.valid_metrics.metrics_epoch.iterrows():
            self.best_val_metrics['val_'+index] = {'min': 99999999, 'max': -99999999}

        # max_lr
        if kwargs.get('max_lr'):  # 0.0268269
            max_lr = kwargs.get('max_lr')
        else:
            # ## Calculate the learning rate
            if kwargs.get('find_lr'):
                print('Finding optimal LR...')
                self.backup_model = copy.deepcopy(self.models['model'].state_dict())
                self.backup_opt = copy.deepcopy(self.optimizers['model'].state_dict())
                max_lr = np.median([self._find_lr() for i in range(7)])
            
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizers['model'],
                                                                max_lr=max_lr,
                                                                steps_per_epoch=len(self.train_data_loaders['data']),
                                                                epochs=self.epochs)
        print(f'Max LR: {max_lr}')
        # wandb.run.summary['OneCycle Max LR'] = max_lr

        # learning rate schedulers
        # self.do_lr_scheduling = kwargs.get('find_lr')
        self.do_lr_scheduling = True
        # user defined lr scheduler strategy
        # self.lr_scheduler = self.lr_schedulers['model']

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        start = time.time()
        self.models['model'].train()
        self.train_metrics.reset()
        if len(self.metrics_epoch) > 0:
            outputs = torch.FloatTensor().to(self.device)
            targets = torch.FloatTensor().to(self.device)
        for batch_idx, (data, target) in enumerate(self.train_data_loaders['data']):
            if isinstance(data, dict):
                data = {k: v.to(self.device) for k, v in data.items()}
            else:
                data = data.to(self.device)
                
            target = target.to(self.device)

            self.optimizers['model'].zero_grad()
            output = self.models['model'](data)
            if len(self.metrics_epoch) > 0:
                outputs = torch.cat((outputs, output))
                targets = torch.cat((targets, target))
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizers['model'].step()

            # update learning rate
            if self.do_lr_scheduling:
                self.lr_scheduler.step()
            
            # log loss and lr
            # wandb.log({'loss': loss})
            # wandb.log({'learning_rate': self.optimizers['model'].param_groups[0]['lr']})

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.iter_update('loss', loss.item())
            for met in self.metrics_iter:
                self.train_metrics.iter_update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                epoch_debug = f"Train Epoch: {epoch} {self._progress(batch_idx)} "
                current_metrics = self.train_metrics.current()
                metrics_debug = ", ".join(f"{key}: {value:.6f}" for key, value in current_metrics.items())
                self.logger.debug(epoch_debug + metrics_debug)
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        for met in self.metrics_epoch:
            self.train_metrics.epoch_update(met.__name__, met(outputs, targets))
            # log the training metrics
            # wandb.log({met.__name__: met(outputs, targets)})

        train_log = self.train_metrics.result()
        
        # wandb.run.summary['epochs_trained'] = epoch
        
        if self.do_validation:
            valid_log = self._valid_epoch(epoch)
            valid_log.set_index('val_' + valid_log.index.astype(str), inplace=True)
            
            # log the validation metrics
            # wandb.log({str(index) : row['mean'] for index, row in valid_log.iterrows()})
            
            # update best/worst metric results
            for index, row in valid_log.iterrows():
                if index in self.best_val_metrics:
                    if row['mean'] < self.best_val_metrics[index]['min']:
                        self.best_val_metrics[index]['min'] = row['mean']
                    if row['mean'] > self.best_val_metrics[index]['max']:
                        self.best_val_metrics[index]['max'] = row['mean']
                    
                    # wandb.run.summary['lowest_'+index] = self.best_val_metrics[index]['min']
                    # wandb.run.summary['highest_'+index] = self.best_val_metrics[index]['max']

#         if self.do_lr_scheduling:
#             self.lr_scheduler.step()

        log = pd.concat([train_log, valid_log])
        end = time.time()
        ty_res = time.gmtime(end - start)
        res = time.strftime("%H hours, %M minutes, %S seconds", ty_res)
        epoch_log = {'epochs': epoch,
                     'iterations': self.len_epoch * epoch,
                     'Runtime': res}
        epoch_info = ', '.join(f"{key}: {value}" for key, value in epoch_log.items())
        logger_info = f"{epoch_info}\n{log}"
        self.logger.info(logger_info)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.models['model'].eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            if len(self.metrics_epoch) > 0:
                outputs = torch.FloatTensor().to(self.device)
                targets = torch.FloatTensor().to(self.device)
            for batch_idx, (data, target) in enumerate(self.valid_data_loaders['data']):
                if isinstance(data, dict):
                    data = {k: v.to(self.device) for k, v in data.items()}
                else:
                    data = data.to(self.device)

                target = target.to(self.device)

                output = self.models['model'](data)
                loss = self.criterion(output, target)
                if len(self.metrics_epoch) > 0:
                    outputs = torch.cat((outputs, output))
                    targets = torch.cat((targets, target))

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, 'valid')
                self.valid_metrics.iter_update('loss', loss.item())
                for met in self.metrics_iter:
                    self.valid_metrics.iter_update(met.__name__, met(output, target))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        for met in self.metrics_epoch:
            self.valid_metrics.epoch_update(met.__name__, met(outputs, targets))

        # # add histogram of model parameters to the tensorboard
        # for name, param in self.models['model'].named_parameters():
        #     self.writer.add_histogram(name, param, bins='auto')

        valid_log = self.valid_metrics.result()

        return valid_log
    
    def _find_lr(self):
        lrs = np.logspace(-7, 2, base=10, num=50)
        losses = []
        
        lr_idx = 0
        
        self.models['model'].train()
        
        while lr_idx < len(lrs):
            for batch_idx, (data, target) in enumerate(self.train_data_loaders['data']):
                if lr_idx == len(lrs):
                    break
                    
                lr = lrs[lr_idx]
                self.optimizers['model'].param_groups[0]['lr'] = lr
                
                if isinstance(data, dict):
                    data = {k: v.to(self.device) for k, v in data.items()}
                else:
                    data = data.to(self.device)

                target = target.to(self.device)

                self.optimizers['model'].zero_grad()
                output = self.models['model'](data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizers['model'].step()
                
                losses += [loss.item()]
                
                lr_idx += 1
                
        best_idx = np.argmin(losses)
        best_loss = losses[best_idx]
        best_lr = lrs[best_idx]
        rec_lr = best_lr / 10.0
        
        self.logger.debug(f'Best LR: {best_lr} Recommended 1-Cycle max_lr: {rec_lr}')
        
#         self.models['model'].eval()
        self.models['model'].load_state_dict(self.backup_model)
        self.optimizers['model'].load_state_dict(self.backup_opt)
        
        return rec_lr
    
    def _progress(self, batch_idx):
        ratio = '[{}/{} ({:.0f}%)]'
        return ratio.format(batch_idx, self.len_epoch, 100.0 * batch_idx / self.len_epoch)


class AFPINet_DANN_Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, torch_args: dict, save_dir, resume, device, **kwargs):
        self.device = device
        super().__init__(torch_args, save_dir, **kwargs)

        if resume is not None:
            self._resume_checkpoint(resume, finetune=self.finetune)

        # data_loaders
        self.do_validation = self.valid_data_loaders['data'] is not None
        if self.len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_data_loaders['data'])
        else:
            # iteration-based training
            self.train_data_loaders['data'] = inf_loop(self.train_data_loaders['data'])
        self.log_step = int(np.sqrt(self.train_data_loaders['data'].batch_size))

        # losses
        self.criterion = self.losses['loss']

        # self.loss_domain = torch.nn.NLLLoss()
        self.loss_domain = self.losses['loss']
        # self.criterion = torch.nn.NLLLoss()

        # metrics
        keys_loss = ['loss']
        keys_iter = [m.__name__ for m in self.metrics_iter]
        keys_epoch = [m.__name__ for m in self.metrics_epoch]
        self.train_metrics = MetricTracker(keys_loss + keys_iter, keys_epoch, writer=self.writer)
        self.valid_metrics = MetricTracker(keys_loss + keys_iter, keys_epoch, writer=self.writer)

        # init wandb model watch
        # wandb.watch(self.models['model'])

        #         wandb.save("trainers/cr_trainer.py", base_path="../")

        # init history logging of best/worst metrics
        self.best_val_metrics = {}
        for index, row in self.valid_metrics.metrics_epoch.iterrows():
            self.best_val_metrics['val_' + index] = {'min': 99999999, 'max': -99999999}

        # max_lr
        if kwargs.get('max_lr'):  # 0.0268269
            max_lr = kwargs.get('max_lr')
        else:
            # ## Calculate the learning rate
            if kwargs.get('find_lr'):
                print('Finding optimal LR...')
                self.backup_model = copy.deepcopy(self.models['model'].state_dict())
                self.backup_opt = copy.deepcopy(self.optimizers['model'].state_dict())
                max_lr = np.median([self._find_lr() for i in range(7)])

        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizers['model'],
                                                                max_lr=max_lr,
                                                                steps_per_epoch=len(self.train_data_loaders['data']),
                                                                epochs=self.epochs)
        print(f'Max LR: {max_lr}')
        # wandb.run.summary['OneCycle Max LR'] = max_lr

        # learning rate schedulers
        # self.do_lr_scheduling = kwargs.get('find_lr')
        self.do_lr_scheduling = True
        # user defined lr scheduler strategy
        # self.lr_scheduler = self.lr_schedulers['model']

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        n_epoch = 20
        start = time.time()
        self.models['model'].train()
        self.train_metrics.reset()
        if len(self.metrics_epoch) > 0:
            outputs = torch.FloatTensor().to(self.device)
            targets = torch.FloatTensor().to(self.device)

        len_dataloader = min(len(self.train_data_loaders['data']), len(self.valid_data_loaders['data']))
        data_source_iter = iter(self.train_data_loaders['data'])
        data_target_iter = iter(self.valid_data_loaders['data'])

        # for batch_idx, (data, target) in enumerate(self.train_data_loaders['data']):
        batch_idx = 0
        for i in range(len_dataloader):
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            batch_idx += 1

            # training model using source data
            data_source = next(data_source_iter)
            s_img, s_label = data_source

            self.optimizers['model'].zero_grad()
            batch_size = len(s_label)
            domain_label = torch.zeros(batch_size).long()

            if isinstance(s_img, dict):
                s_img = {k: v.to(self.device) for k, v in s_img.items()}
            else:
                s_img = s_img.to(self.device)

            s_label = s_label.to(self.device)
            domain_label = domain_label.to(self.device)

            class_output, domain_output = self.models['model'](s_img, alpha)

            if len(self.metrics_epoch) > 0:
                outputs = torch.cat((outputs, class_output))
                targets = torch.cat((targets, s_label))

            err_s_label = self.criterion(class_output, s_label)
            err_s_domain = self.loss_domain(domain_output, domain_label)

            # training model using target data
            data_target = next(data_target_iter)
            t_img, t_labels = data_target

            batch_size = len(t_labels)
            domain_label = torch.ones(batch_size).long()

            if isinstance(t_img, dict):
                t_img = {k: v.to(self.device) for k, v in t_img.items()}
            else:
                t_img = t_img.to(self.device)
            domain_label = domain_label.to(self.device)

            _, domain_output = self.models['model'](t_img, alpha)
            err_t_domain = self.loss_domain(domain_output, domain_label)
            loss = err_t_domain + err_s_domain + err_s_label
            loss.backward()
            self.optimizers['model'].step()

            # update learning rate
            # if self.do_lr_scheduling:
            #     self.lr_scheduler.step()

            # log loss and lr
            # wandb.log({'loss': loss})
            # wandb.log({'learning_rate': self.optimizers['model'].param_groups[0]['lr']})

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.iter_update('loss', loss.item())
            for met in self.metrics_iter:
                self.train_metrics.iter_update(met.__name__, met(class_output, s_label))

            if batch_idx % self.log_step == 0:
                epoch_debug = f"Train Epoch: {epoch} {self._progress(batch_idx)} "
                current_metrics = self.train_metrics.current()
                metrics_debug = ", ".join(f"{key}: {value:.6f}" for key, value in current_metrics.items())
                self.logger.debug(epoch_debug + metrics_debug)
                # print('err_t_domain:{:.4f}, err_s_domain:{:.4f}, err_s_label:{:.4f}'.format(err_t_domain, err_s_domain, err_s_label))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        for met in self.metrics_epoch:
            self.train_metrics.epoch_update(met.__name__, met(outputs, targets))
            # log the training metrics
            # wandb.log({met.__name__: met(outputs, targets)})

        train_log = self.train_metrics.result()

        # wandb.run.summary['epochs_trained'] = epoch

        if self.do_validation:
            valid_log = self._valid_epoch(epoch)
            valid_log.set_index('val_' + valid_log.index.astype(str), inplace=True)

            # log the validation metrics
            # wandb.log({str(index) : row['mean'] for index, row in valid_log.iterrows()})

            # update best/worst metric results
            for index, row in valid_log.iterrows():
                if index in self.best_val_metrics:
                    if row['mean'] < self.best_val_metrics[index]['min']:
                        self.best_val_metrics[index]['min'] = row['mean']
                    if row['mean'] > self.best_val_metrics[index]['max']:
                        self.best_val_metrics[index]['max'] = row['mean']

                    # wandb.run.summary['lowest_'+index] = self.best_val_metrics[index]['min']
                    # wandb.run.summary['highest_'+index] = self.best_val_metrics[index]['max']

        #         if self.do_lr_scheduling:
        #             self.lr_scheduler.step()

        log = pd.concat([train_log, valid_log])
        end = time.time()
        ty_res = time.gmtime(end - start)
        res = time.strftime("%H hours, %M minutes, %S seconds", ty_res)
        epoch_log = {'epochs': epoch,
                     'iterations': self.len_epoch * epoch,
                     'Runtime': res}
        epoch_info = ', '.join(f"{key}: {value}" for key, value in epoch_log.items())
        logger_info = f"{epoch_info}\n{log}"
        self.logger.info(logger_info)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        alpha = 0
        self.models['model'].eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            if len(self.metrics_epoch) > 0:
                outputs = torch.FloatTensor().to(self.device)
                targets = torch.FloatTensor().to(self.device)
            for batch_idx, (data, target) in enumerate(self.valid_data_loaders['data']):
                if isinstance(data, dict):
                    data = {k: v.to(self.device) for k, v in data.items()}
                else:
                    data = data.to(self.device)

                target = target.to(self.device)

                output, _ = self.models['model'](data, alpha)

                loss = self.criterion(output, target)
                if len(self.metrics_epoch) > 0:
                    outputs = torch.cat((outputs, output))
                    targets = torch.cat((targets, target))

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, 'valid')
                self.valid_metrics.iter_update('loss', loss.item())
                for met in self.metrics_iter:
                    self.valid_metrics.iter_update(met.__name__, met(output, target))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        for met in self.metrics_epoch:
            self.valid_metrics.epoch_update(met.__name__, met(outputs, targets))

        # # add histogram of model parameters to the tensorboard
        # for name, param in self.models['model'].named_parameters():
        #     self.writer.add_histogram(name, param, bins='auto')

        valid_log = self.valid_metrics.result()

        return valid_log

    def _find_lr(self):
        lrs = np.logspace(-7, 2, base=10, num=50)
        losses = []

        lr_idx = 0

        self.models['model'].train()

        while lr_idx < len(lrs):
            for batch_idx, (data, target) in enumerate(self.train_data_loaders['data']):
                if lr_idx == len(lrs):
                    break

                lr = lrs[lr_idx]
                self.optimizers['model'].param_groups[0]['lr'] = lr

                if isinstance(data, dict):
                    data = {k: v.to(self.device) for k, v in data.items()}
                else:
                    data = data.to(self.device)

                target = target.to(self.device)

                self.optimizers['model'].zero_grad()
                output, _ = self.models['model'](data, 0)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizers['model'].step()

                losses += [loss.item()]

                lr_idx += 1

        best_idx = np.argmin(losses)
        best_loss = losses[best_idx]
        best_lr = lrs[best_idx]
        rec_lr = best_lr / 10.0

        self.logger.debug(f'Best LR: {best_lr} Recommended 1-Cycle max_lr: {rec_lr}')

        #         self.models['model'].eval()
        self.models['model'].load_state_dict(self.backup_model)
        self.optimizers['model'].load_state_dict(self.backup_opt)

        return rec_lr

    def _progress(self, batch_idx):
        ratio = '[{}/{} ({:.0f}%)]'
        return ratio.format(batch_idx, self.len_epoch, 100.0 * batch_idx / self.len_epoch)

