# Trainer for image classification task

import logging
import time
from collections import OrderedDict
from contextlib import suppress

import torch
from timm.models import model_parameters
from timm.utils import AverageMeter, dispatch_clip_grad, reduce_tensor, accuracy, distribute_bn

import core.plugin
from core.registry import registerTrainer
from core.trainer import Trainer


@registerTrainer
class imgCls(Trainer):
    @property
    def trainerName(self):
        return "ImageClassification"

    def __init__(self, config, optimizer, scheduler, logger, saver, train_loss, eval_loss, train_loader=None,
                 device='cuda', plugins=[], test_loader=None, verbose=False):
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.verbose = verbose
        self.saver = saver

        self.device = device

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_loss = train_loss
        self.eval_loss = eval_loss

        self.plugins: list[core.plugin.Plugin] = plugins

        self.cmd_logger = logging.getLogger("ImageClsTrainer")
        self.eval_metric = config.Trainer.eval_metric

        self.loss_scaler = torch.cuda.amp.GradScaler() if config.Trainer.amp else None
        self.loss_scaler.state_dict_key = "amp_scaler"
        if saver is not None:
            saver.amp_scaler = self.loss_scaler
            saver.desceasing = (self.eval_metric == 'loss')
        self.cmd_logger.debug(self.plugins)

    def trainModel(self, model, **kwargs):
        assert self.train_loader is not None
        best_epoch = None
        best_metric = None
        if self.scheduler is not None and self.config.Trainer.start_epoch > 0:
            self.scheduler.step(self.config.Trainer.start_epoch)
        if self.verbose:
            self.cmd_logger.info("Starting from {0} epoch".format(self.config.Trainer.start_epoch))
        for epoch in range(self.config.Trainer.start_epoch, self.config.Trainer.scheduled_epochs):
            for plg in self.plugins:
                plg.epochHeadHook(model, epoch)
            if self.config.Experiment.dist and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            train_metrics = self._train_one_epoch(model, epoch)

            if self.config.Experiment.dist and self.config.Experiment.dist_bn in ('broadcast', 'reduce'):
                if self.verbose:
                    self.cmd_logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, torch.distributed.get_world_size(), self.config.Experiment.dist_bn == 'reduce')

            eval_metrics = self.evalModel(model, epoch=epoch)
            if self.scheduler is not None:
                self.scheduler.step(epoch + 1, eval_metrics[self.eval_metric])
            if self.saver is not None:
                save_metric = eval_metrics[self.eval_metric]
                best_metric, best_epoch = self.saver.save_checkpoint(epoch, metric=save_metric)
            for plg in self.plugins:
                plg.epochTailHook(model, epoch)
        if best_metric is not None:
            self.cmd_logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

    def evalModel(self, model, **kwargs):
        assert self.test_loader is not None
        batch_time_m = AverageMeter()
        losses_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()
        model.eval()
        amp_autocast = torch.cuda.amp.autocast if self.config.Trainer.amp else suppress

        end = time.time()
        last_idx = len(self.test_loader) - 1
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(self.test_loader):
                last_batch = batch_idx == last_idx
                if not self.config.Trainer.prefetcher:
                    input = input.to(self.device)
                    target = target.to(self.device)
                if self.config.Experiment.channel_last:
                    input = input.contiguous(memory_format=torch.channels_last)

                with amp_autocast():
                    output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                loss = self.eval_loss(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                if self.config.Experiment.dist:
                    reduced_loss = reduce_tensor(loss.data, torch.distributed.get_world_size())
                    acc1 = reduce_tensor(acc1, torch.distributed.get_world_size())
                    acc5 = reduce_tensor(acc5, torch.distributed.get_world_size())
                else:
                    reduced_loss = loss.data

                torch.cuda.synchronize()

                losses_m.update(reduced_loss.item(), input.size(0))
                top1_m.update(acc1.item(), output.size(0))
                top5_m.update(acc5.item(), output.size(0))

                batch_time_m.update(time.time() - end)
                end = time.time()

                for plg in self.plugins:
                    plg.evalIterHook(model, batch_idx, logger=self.logger)
                if self.verbose and (last_batch or batch_idx % self.config.Experiment.log_interval == 0):
                    log_name = 'Test'
                    self.cmd_logger.info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                        'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                            log_name, batch_idx, last_idx, batch_time=batch_time_m,
                            loss=losses_m, top1=top1_m, top5=top5_m))
                #Stop after certain iterations
                if "n_iter" in kwargs.keys() and batch_idx>=kwargs["n_iter"]:
                    self.cmd_logger.info("Stop evaluation in iteration {0} as directed!")
                    break
        for plg in self.plugins:
            plg.evalTailHook(model, logger=self.logger,
                             epoch_id=None if 'epoch' not in kwargs.keys() else kwargs['epoch'])

        if self.logger is not None and 'epoch' in kwargs.keys() and self.verbose:
            self.logger.log_scalar(losses_m.avg, "loss", "Test", kwargs['epoch'])
            self.logger.log_scalar(top1_m.avg, "Top-1_Acc", "Test", kwargs['epoch'])
            self.logger.log_scalar(top5_m.avg, "Top-5_Acc", "Test", kwargs['epoch'])
        metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

        return metrics

    def _train_one_epoch(self, model, epoch):
        if self.config.Data.augs.mixup.mixup_off_epoch and epoch >= self.config.Data.augs.mixup.mixup_off_epoch:
            if self.config.Trainer.prefetcher and self.train_loader.mixup_enabled:
                self.train_loader.mixup_enabled = False
            elif self.train_loader.mixup_fn is not None:
                self.train_loader.mixup_fn.mixup_enabled = False
        second_order = hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order
        batch_time_m = AverageMeter()
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()
        amp_autocast = torch.cuda.amp.autocast if self.config.Trainer.amp else suppress

        model.train()
        end = time.time()
        last_idx = len(self.train_loader) - 1
        num_updates = epoch * len(self.train_loader)

        for batch_idx, (input, target) in enumerate(self.train_loader):
            self.cmd_logger.debug("Iteration: {0}".format(batch_idx))
            last_batch = batch_idx == last_idx
            data_time_m.update(time.time() - end)
            
            if not self.config.Trainer.prefetcher:
                input, target = input.to(self.device), target.to(self.device)
                if self.train_loader.mixup_fn is not None:
                    input, target = self.train_loader.mixup_fn(input, target)
                    
            if target.shape[1]>1:   #If using mixup
                acc_target = torch.clone(target).argmax(dim=1)
                #Extract the label from mixup version
            else:
                acc_target = target

            if self.config.Experiment.channel_last:
                input = input.contiguous(memory_format=torch.channels_last)

            for plg in self.plugins:
                plg.preForwardHook(model, input, target, batch_idx)

            with amp_autocast():
                output = model(input)
                loss = self.train_loss(output, target)

            acc1, acc5 = accuracy(output, acc_target, topk=(1, 5))

            if not self.config.Experiment.dist:
                losses_m.update(loss.item(), input.size(0))
                top1_m.update(acc1.item(), output.size(0))
                top5_m.update(acc5.item(), output.size(0))

            for plg in self.plugins:
                plg.preBackwardHook(model, input, target, loss, epoch)

            self.optimizer.zero_grad()
            if self.loss_scaler is not None:
                self.loss_scaler.scale(loss).backward(create_graph=second_order)
            else:
                loss.backward(create_graph=second_order)

            for plg in self.plugins:
                plg.preUpdateHook(model, input, target, loss, epoch)

            if self.config.Trainer.opt.params.clip_grad is not None:
                if self.loss_scaler is not None:
                    self.loss_scaler.unscale_(self.optimizer)
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in self.config.Trainer.opt.params.clip_mode),
                    value=self.config.Trainer.opt.params.clip_grad, mode=self.config.Trainer.opt.params.clip_mode)
            torch.distributed.barrier()
            if self.loss_scaler is not None:
                self.loss_scaler.step(self.optimizer)
                self.loss_scaler.update()
            else:
                self.optimizer.step()

            torch.cuda.synchronize()
            num_updates += 1
            batch_time_m.update(time.time() - end)
            if last_batch or batch_idx % self.config.Experiment.log_interval == 0:
                lrl = [param_group['lr'] for param_group in self.optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                if self.config.Experiment.dist:
                    reduced_loss = reduce_tensor(loss.data, torch.distributed.get_world_size())
                    losses_m.update(reduced_loss.item(), input.size(0))
                    acc1 = reduce_tensor(acc1, torch.distributed.get_world_size())
                    acc5 = reduce_tensor(acc5, torch.distributed.get_world_size())
                    top1_m.update(acc1.item(), output.size(0))
                    top5_m.update(acc5.item(), output.size(0))

                if self.verbose:
                    self.cmd_logger.info(
                        'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                        'Acc@1: {top1.val:>5.2f} ({top1.avg:>5.2f})  '
                        'Acc@5: {top5.val:>5.2f} ({top5.avg:>5.2f})'
                        'Time: {batch_time.val:.3f}s, {rate:>5.2f}/s  '
                        '({batch_time.avg:.3f}s, {rate_avg:>5.2f}/s)  '
                        'LR: {lr:.3e}  '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            epoch,
                            batch_idx, len(self.train_loader),
                            100. * batch_idx / last_idx,
                            loss=losses_m,
                            top1=top1_m,
                            top5=top5_m,
                            batch_time=batch_time_m,
                            rate=input.size(
                                0) * torch.distributed.get_world_size() if self.config.Experiment.dist else 1.0 / batch_time_m.val,
                            rate_avg=input.size(
                                0) * torch.distributed.get_world_size() if self.config.Experiment.dist else 1.0 / batch_time_m.avg,
                            lr=lr,
                            data_time=data_time_m))

                    if self.logger is not None:
                        self.logger.log_scalar(losses_m.avg, "loss", "Train", epoch, batch_idx)
                        self.logger.log_scalar(top1_m.avg, "Top-1_Acc", "Train", epoch, batch_idx)
                        self.logger.log_scalar(top5_m.avg, "Top-5_Acc", "Train", epoch, batch_idx)
                        self.logger.log_scalar(lr, "LR", "Train", epoch, batch_idx)
            if self.saver is not None and self.config.Experiment.recovery_interval and (
                    last_batch or (batch_idx + 1) % self.config.Experiment.recovery_interval == 0):
                self.saver.save_recovery(epoch, batch_idx=batch_idx)

            if self.scheduler is not None:
                self.scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

            for plg in self.plugins:
                plg.iterTailHook(model, input, target, self.logger, batch_idx)

            end = time.time()
            # end for

        if hasattr(self.optimizer, 'sync_lookahead'):
            self.optimizer.sync_lookahead()

        return OrderedDict([('loss', losses_m.avg)])
