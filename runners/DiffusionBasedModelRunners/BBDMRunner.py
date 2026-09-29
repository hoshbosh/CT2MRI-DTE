import os
import numpy as np
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from PIL import Image
from tqdm.autonotebook import tqdm
from Register import Registers
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from runners.DiffusionBasedModelRunners.DiffusionBaseRunner import DiffusionBaseRunner
from runners.utils import *
from runners.eval import calcul_metrics, save_exp_result, foreground_mask

import nibabel as nib
from collections import defaultdict
import pandas as pd

import time
import wandb


def unpack_batch(batch):
    """(x, x_name), (x_cond, x_cond_name), context, labels -- with explicit arity.

    Every call site used to spell this as a starred rest-capture followed by
    taking element [0], which silently discarded anything past the third item.
    That was harmless while batches were 3-tuples and became a real hazard once
    the label-carrying dataset started emitting a 4th: the label map would have
    been dropped with no error, and a structure-weighted training run would have
    completed unweighted while reporting a clean negative result.

    `labels` is None for the 2- and 3-tuple datasets. Sampling paths legitimately
    ignore it -- there is no loss to weight -- but they now do so visibly.
    """
    if len(batch) == 4:
        (x, x_name), (x_cond, x_cond_name), context, labels = batch
    elif len(batch) == 3:
        (x, x_name), (x_cond, x_cond_name), context = batch
        labels = None
    elif len(batch) == 2:
        (x, x_name), (x_cond, x_cond_name) = batch
        context, labels = None, None
    else:
        raise ValueError(
            f"batch has {len(batch)} elements; expected 2 (image pair), "
            f"3 (+ histogram context) or 4 (+ label map)."
        )
    return (x, x_name), (x_cond, x_cond_name), context, labels


@Registers.runners.register_with_name('BBDMRunner')
class BBDMRunner(DiffusionBaseRunner):
    def __init__(self, config):
        super().__init__(config)

    def initialize_model(self, config):
        if config.model.model_type == "BBDM":
            bbdmnet = BrownianBridgeModel(config.model).to(config.training.device[0])
        # elif config.model.model_type == "LBBDM":
        #     bbdmnet = LatentBrownianBridgeModel(config.model).to(config.training.device[0])
        else:
            raise NotImplementedError
        bbdmnet.apply(weights_init)
        return bbdmnet

    def load_model_from_checkpoint(self):
        states = None
        if self.config.model.only_load_latent_mean_std:
            if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
                states = torch.load(self.config.model.model_load_path, map_location='cpu')
        else:
            states = super().load_model_from_checkpoint()

        if self.config.model.normalize_latent:
            if states is not None:
                self.net.ori_latent_mean = states['ori_latent_mean'].to(self.config.training.device[0])
                self.net.ori_latent_std = states['ori_latent_std'].to(self.config.training.device[0])
                self.net.cond_latent_mean = states['cond_latent_mean'].to(self.config.training.device[0])
                self.net.cond_latent_std = states['cond_latent_std'].to(self.config.training.device[0])
            else:
                if self.config.args.train:
                    self.get_latent_mean_std()

    def print_model_summary(self, net):
        def get_parameter_number(model):
            total_num = sum(p.numel() for p in model.parameters())
            trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_num, trainable_num

        total_num, trainable_num = get_parameter_number(net)
        print("Total Number of parameter: %.2fM" % (total_num / 1e6))
        print("Trainable Number of parameter: %.2fM" % (trainable_num / 1e6))

    def initialize_optimizer_scheduler(self, net, config):
        optim_config = config.model.BB.optimizer
        cross_attn_lr = getattr(optim_config, 'cross_attn_lr', None)

        if cross_attn_lr is not None:
            param_groups = net.get_parameter_groups(
                base_lr=optim_config.lr,
                cross_attn_lr=cross_attn_lr,
            )
            print(f"Using separate LRs: base={optim_config.lr}, cross_attn={cross_attn_lr}")
            print(f"  Base params: {sum(p.numel() for p in param_groups[0]['params'])/ 1e6:.2f}M")
            print(f"  Cross-attn params: {sum(p.numel() for p in param_groups[1]['params'])/ 1e6:.2f}M")
            optimizer = torch.optim.Adam(
                param_groups,
                lr=optim_config.lr,
                weight_decay=optim_config.weight_decay,
                betas=(optim_config.beta1, 0.999),
            )
        else:
            optimizer = get_optimizer(optim_config, net.get_parameters())

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               threshold_mode='rel',
                                                               **vars(config.model.BB.lr_scheduler))
        return [optimizer], [scheduler]

    @torch.no_grad()
    def get_checkpoint_states(self, stage='epoch_end'):
        model_states, optimizer_scheduler_states = super().get_checkpoint_states()
        if self.config.model.normalize_latent:
            if self.config.training.use_DDP:
                model_states['ori_latent_mean'] = self.net.module.ori_latent_mean
                model_states['ori_latent_std'] = self.net.module.ori_latent_std
                model_states['cond_latent_mean'] = self.net.module.cond_latent_mean
                model_states['cond_latent_std'] = self.net.module.cond_latent_std
            else:
                model_states['ori_latent_mean'] = self.net.ori_latent_mean
                model_states['ori_latent_std'] = self.net.ori_latent_std
                model_states['cond_latent_mean'] = self.net.cond_latent_mean
                model_states['cond_latent_std'] = self.net.cond_latent_std
        return model_states, optimizer_scheduler_states

    def get_latent_mean_std(self):
        train_dataset, val_dataset, test_dataset = get_dataset(self.config.data)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.data.train.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True)

        total_ori_mean = None
        total_ori_var = None
        total_cond_mean = None
        total_cond_var = None
        max_batch_num = 30000 // self.config.data.train.batch_size

        def calc_mean(batch, total_ori_mean=None, total_cond_mean=None):
            (x, x_name), (x_cond, x_cond_name), context, _ = unpack_batch(batch)

            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            x_mean = x_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_ori_mean = x_mean if total_ori_mean is None else x_mean + total_ori_mean

            x_cond_mean = x_cond_latent.mean(axis=[0, 2, 3], keepdim=True)
            total_cond_mean = x_cond_mean if total_cond_mean is None else x_cond_mean + total_cond_mean
            return total_ori_mean, total_cond_mean

        def calc_var(batch, ori_latent_mean=None, cond_latent_mean=None, total_ori_var=None, total_cond_var=None):
            (x, x_name), (x_cond, x_cond_name), context, _ = unpack_batch(batch)
            
            x = x.to(self.config.training.device[0])
            x_cond = x_cond.to(self.config.training.device[0])

            x_latent = self.net.encode(x, cond=False, normalize=False)
            x_cond_latent = self.net.encode(x_cond, cond=True, normalize=False)
            x_var = ((x_latent - ori_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_ori_var = x_var if total_ori_var is None else x_var + total_ori_var

            x_cond_var = ((x_cond_latent - cond_latent_mean) ** 2).mean(axis=[0, 2, 3], keepdim=True)
            total_cond_var = x_cond_var if total_cond_var is None else x_cond_var + total_cond_var
            return total_ori_var, total_cond_var

        print(f"start calculating latent mean")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            batch_count += 1
            total_ori_mean, total_cond_mean = calc_mean(train_batch, total_ori_mean, total_cond_mean)

        ori_latent_mean = total_ori_mean / batch_count
        self.net.ori_latent_mean = ori_latent_mean

        cond_latent_mean = total_cond_mean / batch_count
        self.net.cond_latent_mean = cond_latent_mean

        print(f"start calculating latent std")
        batch_count = 0
        for train_batch in tqdm(train_loader, total=len(train_loader), smoothing=0.01):
            batch_count += 1
            total_ori_var, total_cond_var = calc_var(train_batch,
                                                     ori_latent_mean=ori_latent_mean,
                                                     cond_latent_mean=cond_latent_mean,
                                                     total_ori_var=total_ori_var,
                                                     total_cond_var=total_cond_var)

        ori_latent_var = total_ori_var / batch_count
        cond_latent_var = total_cond_var / batch_count

        self.net.ori_latent_std = torch.sqrt(ori_latent_var)
        self.net.cond_latent_std = torch.sqrt(cond_latent_var)
        print(self.net.ori_latent_mean)
        print(self.net.ori_latent_std)
        print(self.net.cond_latent_mean)
        print(self.net.cond_latent_std)

    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        (x, x_name), (x_cond, x_cond_name), context, labels = unpack_batch(batch)

        x = x.to(self.config.training.device[0], non_blocking=True)
        x_cond = x_cond.to(self.config.training.device[0], non_blocking=True)
        if context is not None:
            context = context.to(self.config.training.device[0], non_blocking=True)
        if labels is not None:
            labels = labels.to(self.config.training.device[0], non_blocking=True)

        loss, additional_info = net(x, x_cond, context=context, labels=labels)
        if write:
            self.writer.add_scalar(f'loss/{stage}', loss, step)
            # Also to tensorboard, not just wandb: wandb runs offline when the
            # node has no key, and this scalar is how a null result gets told
            # apart from "the weighting never engaged".
            if 'deepgray_frac' in additional_info:
                self.writer.add_scalar(f'deepgray_frac/{stage}',
                                       additional_info['deepgray_frac'], step)
            try:
                log_dict = {f"loss/{stage}": loss}
                if 'recloss_l1' in additional_info:
                    log_dict[f"loss_l1/{stage}"] = additional_info['recloss_l1']
                if 'perceptual_loss' in additional_info:
                    log_dict[f"loss_perceptual/{stage}"] = additional_info['perceptual_loss']
                if 'frequency_loss' in additional_info:
                    log_dict[f"loss_frequency/{stage}"] = additional_info['frequency_loss']
                if 'deepgray_frac' in additional_info:
                    log_dict[f"deepgray_frac/{stage}"] = additional_info['deepgray_frac']
                wandb.log(log_dict, step=step)
            except:
                print(f'Could not log loss to wandb')
            if additional_info.__contains__('recloss_noise'):
                self.writer.add_scalar(f'recloss_noise/{stage}', additional_info['recloss_noise'], step)

            if additional_info.__contains__('recloss_xy'):
                self.writer.add_scalar(f'recloss_xy/{stage}', additional_info['recloss_xy'], step)
        return loss

    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage='train'):
        sample_path = make_dir(os.path.join(sample_path, f'{stage}_sample'))

        # Labels carry no information for sampling; dropped deliberately.
        (x, x_name), (x_cond, x_cond_name), context, _ = unpack_batch(batch)

        batch_size = x.shape[0] if x.shape[0] < 4 else 4

        x = x[0:batch_size].to(self.config.training.device[0])
        x_cond = x_cond[0:batch_size].to(self.config.training.device[0])
        if context is not None:
            context = context[0:batch_size].to(self.config.training.device[0])

        grid_size = 4
        sample = net.sample(x, x_cond, context=context, clip_denoised=self.config.testing.clip_denoised, config=self.config, device=self.config.training.device[0]).to('cpu')
        image_grid = get_image_grid(sample, grid_size, to_normal=self.config.data.dataset_config.to_normal)
        mid_slice_index = image_grid.shape[-1] // 2
        image_grid = image_grid[:,:,mid_slice_index:mid_slice_index+1]
        im = Image.fromarray(image_grid[:,:,0])
        im.save(os.path.join(sample_path, 'skip_sample.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_skip_sample', image_grid, self.global_step, dataformats='HWC')
            try:
                wandb.log({f'{stage}_skip_sample': [wandb.Image(image_grid, caption=f'{stage}_skip_sample')]}, step=self.global_step)
            except:
                print(f'Could not log {stage}_skip_sample to wandb')
            
        image_grid = get_image_grid(x_cond.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        image_grid = image_grid[:,:,mid_slice_index:mid_slice_index+1]
        im = Image.fromarray(image_grid[:,:,0])
        im.save(os.path.join(sample_path, 'condition.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_condition', image_grid, self.global_step, dataformats='HWC')
            try:
                wandb.log({f'{stage}_condition': [wandb.Image(image_grid, caption=f'{stage}_condition')]}, step=self.global_step)
            except:
                print(f'Could not log {stage}_condition to wandb')
                
        image_grid = get_image_grid(x.to('cpu'), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        image_grid = image_grid[:,:,mid_slice_index:mid_slice_index+1]
        im = Image.fromarray(image_grid[:,:,0])
        im.save(os.path.join(sample_path, 'ground_truth.png'))
        if stage != 'test':
            self.writer.add_image(f'{stage}_ground_truth', image_grid, self.global_step, dataformats='HWC')
            try:
                wandb.log({f'{stage}_ground_truth': [wandb.Image(image_grid, caption=f'{stage}_ground_truth')]}, step=self.global_step)
            except:
                print(f'Could not log {stage}_ground_truth to wandb')

    @torch.no_grad()
    def sample_to_eval(self, net, test_dataset, sample_path):
        start_time = time.time()

        mid_slice = self.config.data.dataset_config.channels // 2
        H = self.config.data.dataset_config.image_size
        sample_step = self.config.model.BB.params.sample_step
        inference_type = self.config.model.BB.params.inference_type
        num_ISTA_step = self.config.model.BB.params.num_ISTA_step
        ISTA_step_size = self.config.model.BB.params.ISTA_step_size
        dataset_type = self.config.data.dataset_type

        sample_path = os.path.join(sample_path, f"{inference_type}_{sample_step}")
        if 'ISTA' in inference_type:
            sample_path = os.path.join(sample_path, f"{inference_type}_{sample_step}_{ISTA_step_size}_{num_ISTA_step}")

        if "colin" in dataset_type:
            sample_path += '_colin'
        elif "best" in dataset_type:
            sample_path += '_best_meanmax'
        elif "average" in dataset_type:
            sample_path += '_average'

        # Uncertainty quantification: draw `uq_samples` stochastic samples per subject and
        # report their per-voxel mean (ensemble prediction) and std (predictive uncertainty).
        # Sampling is only stochastic when eta > 0 (see p_sample: sigma_t scales with eta),
        # so at eta == 0 every member would be bit-identical and the std map would be zero.
        uq_samples = getattr(self.config.model.BB.params, 'uq_samples', 1) or 1
        if uq_samples > 1:
            if self.config.model.BB.params.eta <= 0:
                raise ValueError(
                    f"uq_samples={uq_samples} requires ddim_eta > 0; got eta={self.config.model.BB.params.eta}. "
                    "At eta=0 sampling is deterministic and all members would be identical."
                )
            # Deliberately not keyed by N: members are cached per index, so a later run with
            # a larger uq_samples reuses the members already on disk and only adds the new ones.
            sample_path += '_uq'

        print(f"sample_path: {sample_path}")
        os.makedirs(sample_path, exist_ok=True)

        # Group slices by patient, keeping both data and ground truth
        batch_dict = defaultdict(list)
        gt_dict = defaultdict(list)
        for idx in range(len(test_dataset)):
            sample_item = test_dataset[idx]
            pid = sample_item[0][1].decode('utf-8')
            batch_dict[pid].append(sample_item)
            # Extract ground truth MR mid-slice from the dataset
            gt_slice = sample_item[0][0][mid_slice]  # ground truth MR, mid channel
            gt_dict[pid].append(gt_slice)

        metrics_dict = defaultdict(dict)
        single_metrics_dict = defaultdict(dict)
        for pid in tqdm(batch_dict.keys()):
            # One cached .nii per ensemble member, so a job killed mid-run resumes at
            # member granularity rather than redoing the whole subject.
            if uq_samples > 1:
                member_paths = [os.path.join(sample_path, f'{pid}_s{k}.nii') for k in range(uq_samples)]
            else:
                member_paths = [os.path.join(sample_path, f'{pid}.nii')]

            # Build ground truth volume from dataset slices
            gt_slices = np.stack(gt_dict[pid], axis=0)  # [num_slices, H, W]
            if self.config.data.dataset_config.to_normal:
                gt_slices = gt_slices * 0.5 + 0.5  # denormalize [-1,1] -> [0,1]
            gt_slices = np.clip(gt_slices, 0, 1)
            gt_volume = np.transpose(gt_slices, (1, 2, 0))  # [H, W, num_slices]

            batch_gpu = None
            members = []
            for k, member_path in enumerate(member_paths):
                if os.path.exists(member_path):
                    print(f'resuming: {pid} member {k} (loading cached .nii)')
                    members.append(nib.load(member_path).get_fdata())
                    continue

                # Conditioning is identical across members; only the sampling noise differs.
                if batch_gpu is None:
                    test_batch = default_collate(batch_dict[pid])
                    (x, x_name), (x_cond, x_cond_name), context, _ = unpack_batch(test_batch)
                    x_cond = x_cond.to(self.config.training.device[0], non_blocking=True)
                    if context is not None:
                        context = context.to(self.config.training.device[0], non_blocking=True)
                    batch_gpu = (x, x_cond, x_cond_name, context)

                x, x_cond, x_cond_name, context = batch_gpu
                sample = net.sample(x, x_cond, x_cond_name, context=context, clip_denoised=False, path=sample_path, save=False, config=self.config, device=self.config.training.device[0])
                sample = sample[:, mid_slice].detach().clone().cpu().mul_(0.5).add_(0.5).clamp_(0, 1.)

                member = sample.numpy().transpose(1, 2, 0)  # [H, W, num_slices]
                nib.save(nib.Nifti1Image(member, np.eye(4)), member_path)
                members.append(member)
                print(f'saved_id: {pid} member {k}')

            if uq_samples > 1:
                stack = np.stack(members, axis=0)  # [N, H, W, num_slices]
                syn_img = stack.mean(axis=0)       # ensemble prediction
                unc_img = stack.std(axis=0)        # per-voxel predictive std
                nib.save(nib.Nifti1Image(syn_img, np.eye(4)), os.path.join(sample_path, f'{pid}_mean.nii'))
                nib.save(nib.Nifti1Image(unc_img, np.eye(4)), os.path.join(sample_path, f'{pid}_std.nii'))
            else:
                syn_img = members[0]
                unc_img = None

            print(f"  syn range: [{syn_img.min():.4f}, {syn_img.max():.4f}], mean: {syn_img.mean():.4f}")
            print(f"  gt  range: [{gt_volume.min():.4f}, {gt_volume.max():.4f}], mean: {gt_volume.mean():.4f}")
            mask = foreground_mask(gt_volume)
            calcul_metrics(metrics_dict, pid, syn_img, gt_volume, mask=mask, device=self.config.training.device[0])

            if unc_img is not None:
                metrics_dict[pid]['unc_mean'] = float(unc_img.mean())
                metrics_dict[pid]['unc_mean_mask'] = float(unc_img[mask].mean()) if mask.any() else np.nan
                # Member 0 scored on its own, so the ensemble gain can be separated from
                # the effect of moving eta off 0 to make sampling stochastic at all.
                calcul_metrics(single_metrics_dict, pid, members[0], gt_volume, mask=mask, device=self.config.training.device[0])

        df = pd.DataFrame.from_dict(metrics_dict, orient='index')
        means = df.mean()
        df.loc['mean'] = means

        df.to_csv(os.path.join(sample_path, 'results.csv'), index_label='pa_id')

        if single_metrics_dict:
            df_single = pd.DataFrame.from_dict(single_metrics_dict, orient='index')
            df_single.loc['mean'] = df_single.mean()
            df_single.to_csv(os.path.join(sample_path, 'results_single_member.csv'), index_label='pa_id')
            print("\nensemble vs single member (mean over subjects):")
            for m in ['ssim', 'ssim_mask', 'psnr', 'psnr_mask', 'lpips', 'nrmse']:
                print(f"  {m:10s} ensemble={means[m]:.4f}  single={df_single.loc['mean', m]:.4f}")
            print(f"  predictive std: overall={means['unc_mean']:.4f}  in-brain={means['unc_mean_mask']:.4f}")

        results_file = os.path.join(sample_path, 'test_results.csv')
        save_exp_result(results_file, self.config, means)

        end_time = time.time()
        print_runtime(start_time, end_time)

