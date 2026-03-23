import argparse
import datetime
import time
import shutil

import yaml
import os
import traceback

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from abc import ABC, abstractmethod
from tqdm.autonotebook import tqdm

from runners.base.EMA import EMA
from runners.utils import make_save_dirs, make_dir, get_dataset, remove_file

import wandb

class BaseRunner(ABC):
    def __init__(self, config):
        self.net = None  # Neural Network
        self.optimizer = None  # optimizer
        self.scheduler = None  # scheduler
        self.config = config  # config from configuration file

        # set training params
        self.global_epoch = 0  # global epoch
        if config.args.sample_at_start:
            self.global_step = -1  # global step
        else:
            self.global_step = 0

        self.GAN_buffer = {}  # GAN buffer for Generative Adversarial Network
        self.topk_checkpoints = {}  # Top K checkpoints
        self.is_main_process = not config.training.use_DDP or getattr(config.training, 'local_rank', 0) == 0

        # set log and save destination
        self.config.result = argparse.Namespace()
        self.config.result.image_path, \
        self.config.result.ckpt_path, \
        self.config.result.log_path, \
        self.config.result.sample_path, \
        self.config.result.sample_to_eval_path = make_save_dirs(self.config.args,
                                                                prefix=self.config.data.dataset_name,
                                                                suffix=self.config.model.model_name)

        self.save_config()  # save configuration file
        self.writer = SummaryWriter(self.config.result.log_path)  # initialize SummaryWriter

        # initialize model
        self.net, self.optimizer, self.scheduler = self.initialize_model_optimizer_scheduler(self.config)

        self.print_model_summary(self.net)

        # initialize EMA
        self.use_ema = False if not self.config.model.__contains__('EMA') else self.config.model.EMA.use_ema
        if self.use_ema:
            self.ema = EMA(self.config.model.EMA.ema_decay)
            self.update_ema_interval = self.config.model.EMA.update_ema_interval
            self.start_ema_step = self.config.model.EMA.start_ema_step
            self.ema.register(self.net)

        # load model from checkpoint
        self.load_model_from_checkpoint()

        # initialize DDP
        if self.config.training.use_DDP:
            self.net = DDP(self.net, device_ids=[self.config.training.local_rank], output_device=self.config.training.local_rank)
        else:
            self.net = self.net.to(self.config.training.device[0])
        # self.ema.reset_device(self.net)

        # torch.compile: store uncompiled reference for validation/EMA/checkpointing
        self._uncompiled_net = self.net
        if getattr(self.config.training, 'use_compile', False):
            print("Compiling model with torch.compile...")
            self.net = torch.compile(self.net)
            print("Model compiled successfully.")

        # wandb model watching (gradients + parameters) — only on rank 0
        if wandb.run is not None:
            wandb.watch(self._uncompiled_net, log='all', log_freq=500)

    # save configuration file
    def save_config(self):
        if self.config.args.sample_to_eval:
            return
            
        backup_path = os.path.join(self.config.result.ckpt_path, 'config_backup.yaml')
        shutil.copyfile(self.config.args.config, backup_path)

        save_path = os.path.join(self.config.result.ckpt_path, 'config.yaml')
        save_config = self.config
        with open(save_path, 'w') as f:
            yaml.dump(save_config, f)

    def initialize_model_optimizer_scheduler(self, config, is_test=False):
        """
        get model, optimizer, scheduler
        :param args: args
        :param config: config
        :param is_test: is_test
        :return: net: Neural Network, nn.Module;
                 optimizer: a list of optimizers;
                 scheduler: a list of schedulers or None;
        """
        net = self.initialize_model(config)
        optimizer, scheduler = None, None
        if not is_test:
            optimizer, scheduler = self.initialize_optimizer_scheduler(net, config)
        return net, optimizer, scheduler

    # load model, EMA, optimizer, scheduler from checkpoint
    def load_model_from_checkpoint(self):
        model_states = None
        if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
            print(f"load model {self.config.model.model_name} from {self.config.model.model_load_path}")
            model_states = torch.load(self.config.model.model_load_path, map_location='cpu')

            self.global_epoch = model_states['epoch']
            self.global_step = model_states['step']

            # load model (filter out shape-mismatched keys before loading)
            current_state = self.net.state_dict()
            filtered_state = {}
            shape_mismatch = []
            for k, v in model_states['model'].items():
                if k in current_state and current_state[k].shape != v.shape:
                    shape_mismatch.append(k)
                else:
                    filtered_state[k] = v
            missing, unexpected = self.net.load_state_dict(filtered_state, strict=False)
            if shape_mismatch:
                print(f"  Shape-mismatched keys (randomly initialized): {shape_mismatch}")
            if missing:
                print(f"  Missing keys (randomly initialized): {missing}")
            if unexpected:
                print(f"  Unexpected keys (ignored): {unexpected}")

            # load ema (rebuild shadow from current model, then overwrite matching keys)
            if self.use_ema:
                new_shadow = {}
                for name, param in self.net.named_parameters():
                    if name in model_states['ema'] and model_states['ema'][name].shape == param.shape:
                        new_shadow[name] = model_states['ema'][name]
                    else:
                        new_shadow[name] = param.data.clone()
                self.ema.shadow = new_shadow
                self.ema.reset_device(self.net)

            # load optimizer and scheduler
            if self.config.args.train:
                if self.config.model.__contains__('optim_sche_load_path') and self.config.model.optim_sche_load_path is not None:
                    optimizer_scheduler_states = torch.load(self.config.model.optim_sche_load_path, map_location='cpu')
                    for i in range(len(self.optimizer)):
                        self.optimizer[i].load_state_dict(optimizer_scheduler_states['optimizer'][i])

                    if self.scheduler is not None:
                        for i in range(len(self.optimizer)):
                            self.scheduler[i].load_state_dict(optimizer_scheduler_states['scheduler'][i])
        return model_states

    def get_checkpoint_states(self, stage='epoch_end'):
        optimizer_state = []
        for i in range(len(self.optimizer)):
            optimizer_state.append(self.optimizer[i].state_dict())

        scheduler_state = []
        for i in range(len(self.scheduler)):
            scheduler_state.append(self.scheduler[i].state_dict())

        optimizer_scheduler_states = {
            'optimizer': optimizer_state,
            'scheduler': scheduler_state
        }

        model_states = {
            'step': self.global_step,
        }

        if self.config.training.use_DDP:
            model_states['model'] = self._uncompiled_net.module.state_dict()
        else:
            model_states['model'] = self._uncompiled_net.state_dict()

        if stage == 'exception':
            model_states['epoch'] = self.global_epoch
        else:
            model_states['epoch'] = self.global_epoch + 1

        if self.use_ema:
            model_states['ema'] = self.ema.shadow
        return model_states, optimizer_scheduler_states

    # EMA part
    def step_ema(self):
        with_decay = False if self.global_step < self.start_ema_step else True
        if self.config.training.use_DDP:
            self.ema.update(self._uncompiled_net.module, with_decay=with_decay)
        else:
            self.ema.update(self._uncompiled_net, with_decay=with_decay)

    def apply_ema(self):
        if self.use_ema:
            if self.config.training.use_DDP:
                self.ema.apply_shadow(self._uncompiled_net.module)
            else:
                self.ema.apply_shadow(self._uncompiled_net)

    def restore_ema(self):
        if self.use_ema:
            if self.config.training.use_DDP:
                self.ema.restore(self._uncompiled_net.module)
            else:
                self.ema.restore(self._uncompiled_net)

    # Evaluation and sample part
    @torch.no_grad()
    def validation_step(self, val_batch, epoch, step):
        self.apply_ema()
        self._uncompiled_net.eval()
        with torch.cuda.amp.autocast():
            loss = self.loss_fn(net=self._uncompiled_net,
                                batch=val_batch,
                                epoch=epoch,
                                step=step,
                                opt_idx=0,
                                stage='val_step')
            if len(self.optimizer) > 1:
                loss = self.loss_fn(net=self._uncompiled_net,
                                    batch=val_batch,
                                    epoch=epoch,
                                    step=step,
                                    opt_idx=1,
                                    stage='val_step')
        self.restore_ema()

    @torch.no_grad()
    def validation_epoch(self, val_loader, epoch):
        self.apply_ema()
        self._uncompiled_net.eval()

        pbar = tqdm(val_loader, total=len(val_loader), smoothing=0.01)
        step = 0
        loss_sum = 0.
        dloss_sum = 0.
        for val_batch in pbar:
            with torch.cuda.amp.autocast():
                loss = self.loss_fn(net=self._uncompiled_net,
                                    batch=val_batch,
                                    epoch=epoch,
                                    step=step,
                                    opt_idx=0,
                                    stage='val',
                                    write=False)
                loss_sum += loss
                if len(self.optimizer) > 1:
                    loss = self.loss_fn(net=self._uncompiled_net,
                                        batch=val_batch,
                                        epoch=epoch,
                                        step=step,
                                        opt_idx=1,
                                        stage='val',
                                        write=False)
                    dloss_sum += loss
            step += 1
        average_loss = loss_sum / step
        self.writer.add_scalar(f'val_epoch/loss', average_loss, epoch)
        try:
            wandb.log({f'val_epoch/loss': average_loss}, step=epoch)
        except:
            print('Could not log loss/val_epoch to wandb')
        if len(self.optimizer) > 1:
            average_dloss = dloss_sum / step
            self.writer.add_scalar(f'val_dloss_epoch/loss', average_dloss, epoch)
            try:
                wandb.log({f'loss/val_dloss_epoch': average_dloss}, step=epoch)
            except:
                print('Could not log loss/val_dloss_epoch to wandb')

        self.restore_ema()
        return average_loss

    @torch.no_grad()
    def sample_step(self, train_batch, val_batch):
        self.apply_ema()
        self._uncompiled_net.eval()
        sample_path = make_dir(os.path.join(self.config.result.image_path, str(self.global_step)))
        if self.config.training.use_DDP:
            self.sample(self._uncompiled_net.module, train_batch, sample_path, stage='train')
            self.sample(self._uncompiled_net.module, val_batch, sample_path, stage='val')
        else:
            self.sample(self._uncompiled_net, train_batch, sample_path, stage='train')
            self.sample(self._uncompiled_net, val_batch, sample_path, stage='val')
        self.restore_ema()

    # abstract methods
    @abstractmethod
    def print_model_summary(self, net):
        pass

    @abstractmethod
    def initialize_model(self, config):
        """
        initialize model
        :param config: config
        :return: nn.Module
        """
        pass

    @abstractmethod
    def initialize_optimizer_scheduler(self, net, config):
        """
        initialize optimizer and scheduler
        :param net: nn.Module
        :param config: config
        :return: a list of optimizers; a list of schedulers
        """
        pass

    @abstractmethod
    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        """
        loss function
        :param net: nn.Module
        :param batch: batch
        :param epoch: global epoch
        :param step: global step
        :param opt_idx: optimizer index, default is 0; set it to 1 for GAN discriminator
        :param stage: train, val, test
        :param write: write loss information to SummaryWriter
        :return: a scalar of loss
        """
        pass

    @abstractmethod
    def sample(self, net, batch, sample_path, stage='train'):
        """
        sample a single batch
        :param net: nn.Module
        :param batch: batch
        :param sample_path: path to save samples
        :param stage: train, val, test
        :return:
        """
        pass

    @abstractmethod
    def sample_to_eval(self, net, test_dataset, sample_path):
        """
        sample among the test dataset to calculate evaluation metrics
        :param net: nn.Module
        :param test_dataset: test dataset
        :param sample_path: path to save samples
        :return:
        """
        pass

    def on_save_checkpoint(self, net, train_loader, val_loader, epoch, step):
        """
        additional operations whilst saving checkpoint
        :param net: nn.Module
        :param train_loader: train data loader
        :param val_loader: val data loader
        :param epoch: epoch
        :param step: step
        :return:
        """
        pass

    def train(self):
        print(self.__class__.__name__)

        train_dataset, val_dataset = get_dataset(self.config.data, test=False)
        train_sampler = None
        val_sampler = None
        # test_sampler = None
        if self.config.training.use_DDP:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            train_loader = DataLoader(train_dataset,
                                      batch_size=self.config.data.train.batch_size,
                                      num_workers=4,
                                      pin_memory=True,
                                      persistent_workers=True,
                                      drop_last=True,
                                      sampler=train_sampler)
            val_loader = DataLoader(val_dataset,
                                    batch_size=self.config.data.val.batch_size,
                                    num_workers=2,
                                    pin_memory=True,
                                    persistent_workers=True,
                                    drop_last=True,
                                    sampler=val_sampler)
            # test_loader = DataLoader(test_dataset,
            #                          batch_size=self.config.data.test.batch_size,
            #                          num_workers=1,
            #                          drop_last=True,
            #                          sampler=test_sampler)
        else:
            train_loader = DataLoader(train_dataset,
                                      batch_size=self.config.data.train.batch_size,
                                      shuffle=self.config.data.train.shuffle,
                                      num_workers=4,
                                      pin_memory=True,
                                      persistent_workers=True,
                                      drop_last=False)
            val_loader = DataLoader(val_dataset,
                                    batch_size=self.config.data.val.batch_size,
                                    shuffle=self.config.data.val.shuffle,
                                    num_workers=2,
                                    pin_memory=True,
                                    persistent_workers=True,
                                    drop_last=False)
            # test_loader = DataLoader(test_dataset,
            #                          batch_size=self.config.data.test.batch_size,
            #                          shuffle=False,
            #                          num_workers=1,
            #                          drop_last=False)

        epoch_length = len(train_loader)
        start_epoch = self.global_epoch
        print(
            f"start training {self.config.model.model_name} on {self.config.data.dataset_name}, {len(train_loader)} iters per epoch")

        try:
            scaler = torch.cuda.amp.GradScaler()
            accumulate_grad_batches = self.config.training.accumulate_grad_batches
            average_loss = None
            for epoch in range(start_epoch, self.config.training.n_epochs):
                if self.global_step > self.config.training.n_steps:
                    break

                if self.config.training.use_DDP:
                    train_sampler.set_epoch(epoch)
                    val_sampler.set_epoch(epoch)

                pbar = tqdm(train_loader, total=len(train_loader), smoothing=0.01)
                self.global_epoch = epoch
                start_time = time.time()
                val_iter = iter(val_loader)
                grad_norm = None
                for train_batch in pbar:
                    self.global_step += 1
                    self.net.train()

                    losses = []
                    for i in range(len(self.optimizer)):
                        with torch.cuda.amp.autocast():
                            loss = self.loss_fn(net=self.net,
                                                batch=train_batch,
                                                epoch=epoch,
                                                step=self.global_step,
                                                opt_idx=i,
                                                stage='train')

                        scaler.scale(loss).backward()
                        if self.global_step % accumulate_grad_batches == 0:
                            scaler.unscale_(self.optimizer[i])
                            grad_norm = torch.nn.utils.clip_grad_norm_(self._uncompiled_net.parameters(), max_norm=1.0)
                            scaler.step(self.optimizer[i])
                            self.optimizer[i].zero_grad(set_to_none=True)
                            if self.scheduler is not None:
                                self.scheduler[i].step(loss)
                        losses.append(loss.detach().mean())

                    if self.global_step % accumulate_grad_batches == 0:
                        scaler.update()

                    # wandb logging: LR, GPU memory, throughput, gradient norm (rank 0 only)
                    if self.is_main_process and self.global_step % 100 == 0:
                        try:
                            log_dict = {
                                'train/grad_norm': grad_norm.item() if grad_norm is not None else 0,
                                'train/lr': self.optimizer[0].param_groups[0]['lr'],
                                'train/gpu_mem_allocated_mb': torch.cuda.memory_allocated() / 1e6,
                                'train/gpu_mem_reserved_mb': torch.cuda.memory_reserved() / 1e6,
                                'train/epoch': epoch + 1,
                            }
                            wandb.log(log_dict, step=self.global_step)
                        except:
                            pass

                    if self.use_ema and self.global_step % (self.update_ema_interval*accumulate_grad_batches) == 0:
                        self.step_ema()

                    if len(self.optimizer) > 1:
                        pbar.set_description(
                            (
                                f'Epoch: [{epoch + 1} / {self.config.training.n_epochs}] '
                                f'iter: {self.global_step} loss-1: {losses[0]:.4f} loss-2: {losses[1]:.4f}'
                            )
                        )
                    else:
                        pbar.set_description(
                            (
                                f'Epoch: [{epoch + 1} / {self.config.training.n_epochs}] '
                                f'iter: {self.global_step} loss: {losses[0]:.4f}'
                            )
                        )

                    with torch.no_grad():
                        if self.global_step % 500 == 0:
                            try:
                                val_batch = next(val_iter)
                            except StopIteration:
                                val_iter = iter(val_loader)
                                val_batch = next(val_iter)
                            self.validation_step(val_batch=val_batch, epoch=epoch, step=self.global_step)

                        if self.global_step % int(self.config.training.sample_interval * epoch_length) == 0:
                            if not self.config.training.use_DDP or \
                                    (self.config.training.use_DDP and self.config.training.local_rank) == 0:
                                try:
                                    val_batch = next(val_iter)
                                except StopIteration:
                                    val_iter = iter(val_loader)
                                    val_batch = next(val_iter)
                                self.sample_step(val_batch=val_batch, train_batch=train_batch)
                                torch.cuda.empty_cache()

                end_time = time.time()
                elapsed_rounded = int(round((end_time-start_time)))
                print("training time: " + str(datetime.timedelta(seconds=elapsed_rounded)))

                # validation
                if (epoch + 1) % self.config.training.validation_interval == 0 or (
                        epoch + 1) == self.config.training.n_epochs:
                    if not self.config.training.use_DDP or \
                            (self.config.training.use_DDP and self.config.training.local_rank) == 0:
                        with torch.no_grad():
                            print("validating epoch...")
                            average_loss = self.validation_epoch(val_loader, epoch)
                            torch.cuda.empty_cache()
                            print("validating epoch success")

                # save checkpoint
                if (epoch + 1) % self.config.training.save_interval == 0 or \
                        (epoch + 1) == self.config.training.n_epochs or \
                        self.global_step > self.config.training.n_steps:
                    if not self.config.training.use_DDP or \
                            (self.config.training.use_DDP and self.config.training.local_rank) == 0:
                        with torch.no_grad():
                            print("saving latest checkpoint...")
                            self.on_save_checkpoint(self.net, train_loader, val_loader, epoch, self.global_step)
                            model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='epoch_end')

                            # save latest checkpoint
                            temp = 0
                            while temp < epoch + 1:
                                remove_file(os.path.join(self.config.result.ckpt_path, f'latest_model_{temp}.pth'))
                                remove_file(
                                    os.path.join(self.config.result.ckpt_path, f'latest_optim_sche_{temp}.pth'))
                                temp += 1
                            torch.save(model_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'latest_model_{epoch + 1}.pth'))
                            torch.save(optimizer_scheduler_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'latest_optim_sche_{epoch + 1}.pth'))
                            torch.save(model_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'last_model.pth'))
                            torch.save(optimizer_scheduler_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'last_optim_sche.pth'))

                            # save top_k checkpoints
                            model_ckpt_name = f'top_model_epoch_{epoch + 1}.pth'
                            optim_sche_ckpt_name = f'top_optim_sche_epoch_{epoch + 1}.pth'

                            if self.config.args.save_top and average_loss is not None:
                                print("save top model start...")
                                top_key = 'top'
                                if top_key not in self.topk_checkpoints:
                                    print('top key not in topk_checkpoints')
                                    self.topk_checkpoints[top_key] = {"loss": average_loss,
                                                                      'model_ckpt_name': model_ckpt_name,
                                                                      'optim_sche_ckpt_name': optim_sche_ckpt_name}

                                    print(f"saving top checkpoint: average_loss={average_loss} epoch={epoch + 1}")
                                    torch.save(model_states,
                                               os.path.join(self.config.result.ckpt_path, model_ckpt_name))
                                    torch.save(optimizer_scheduler_states,
                                               os.path.join(self.config.result.ckpt_path, optim_sche_ckpt_name))
                                else:
                                    if average_loss < self.topk_checkpoints[top_key]["loss"]:
                                        print("remove " + self.topk_checkpoints[top_key]["model_ckpt_name"])
                                        remove_file(os.path.join(self.config.result.ckpt_path,
                                                                 self.topk_checkpoints[top_key]['model_ckpt_name']))
                                        remove_file(os.path.join(self.config.result.ckpt_path,
                                                                 self.topk_checkpoints[top_key]['optim_sche_ckpt_name']))

                                        print(
                                            f"saving top checkpoint: average_loss={average_loss} epoch={epoch + 1}")

                                        self.topk_checkpoints[top_key] = {"loss": average_loss,
                                                                          'model_ckpt_name': model_ckpt_name,
                                                                          'optim_sche_ckpt_name': optim_sche_ckpt_name}

                                        torch.save(model_states,
                                                   os.path.join(self.config.result.ckpt_path, model_ckpt_name))
                                        torch.save(optimizer_scheduler_states,
                                                   os.path.join(self.config.result.ckpt_path, optim_sche_ckpt_name))
        except BaseException as e:
            if not self.config.training.use_DDP or (self.config.training.use_DDP and self.config.training.local_rank) == 0:
                print("exception save model start....")
                print(self.__class__.__name__)
                model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='exception')
                torch.save(model_states,
                           os.path.join(self.config.result.ckpt_path, f'last_model.pth'))
                torch.save(optimizer_scheduler_states,
                           os.path.join(self.config.result.ckpt_path, f'last_optim_sche.pth'))

                print("exception save model success!")

            print('str(Exception):\t', str(Exception))
            print('str(e):\t\t', str(e))
            print('repr(e):\t', repr(e))
            print('traceback.print_exc():')
            traceback.print_exc()
            print('traceback.format_exc():\n%s' % traceback.format_exc())

        try: 
            wandb.finish()
        except:
            print('Could not finish wandb')
        
    @torch.no_grad()
    def test(self):
        test_dataset = get_dataset(self.config.data, test=True)
        # test_dataset = val_dataset
        if self.config.training.use_DDP:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.config.data.test.batch_size,
                                     shuffle=False,
                                     num_workers=4,
                                     pin_memory=True,
                                     persistent_workers=True,
                                     drop_last=True,
                                     sampler=test_sampler)
        else:
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.config.data.test.batch_size,
                                     shuffle=False,
                                     num_workers=4,
                                     pin_memory=True,
                                     persistent_workers=True,
                                     drop_last=False)

        if self.use_ema:
            self.apply_ema()

        self._uncompiled_net.eval()
        if self.config.args.sample_to_eval:
            sample_path = self.config.result.sample_to_eval_path
            if self.config.training.use_DDP:
                self.sample_to_eval(self._uncompiled_net.module, test_dataset, sample_path)
            else:
                self.sample_to_eval(self._uncompiled_net, test_dataset, sample_path)

        else:
            test_iter = iter(test_loader)
            for i in tqdm(range(1), initial=0, dynamic_ncols=True, smoothing=0.01):
                test_batch = next(test_iter)
                sample_path = os.path.join(self.config.result.sample_path, str(i))
                if self.config.training.use_DDP:
                    self.sample(self._uncompiled_net.module, test_batch, sample_path, stage='test')
                else:
                    self.sample(self._uncompiled_net, test_batch, sample_path, stage='test')

        try: 
            wandb.finish()
        except:
            print('Could not finish wandb')
