import numpy as np
import torch
import pytorch_lightning as pl
import os
from torch.optim.lr_scheduler import MultiStepLR
import wandb
from pytorch_lightning.loggers import WandbLogger
from ..models_2d.loss import JointGNLLLoss
from .utils.visualization import compare_results_denseldmks2d
from ...datasets import MixedWebDataset
from lib.models.models_2d.pose_hrnet import get_pose_net

import yaml


class DenseLdmks2DHRNet(pl.LightningModule):
    def __init__(self, cfg, viz_dir=None):
        super(DenseLdmks2DHRNet, self).__init__()

        self.cfg = cfg
        self.optimizer_cfg = cfg.optimizer.optimizer
        self.lr_cfg = cfg.optimizer.lr_config
        self.train_data_cfg = cfg.data['train']
        self.val_data_cfg = cfg.data['val'] if 'val' in cfg.data else None
        self.use_mask = False
        self.visibility = True

        with open(cfg.model['cfg'], 'r') as f:
            cfg_model = yaml.safe_load(f)
        self.model = get_pose_net(cfg_model, is_train=True)

        self.criterion = JointGNLLLoss(loss_weights=cfg.loss_weights)
        self.vis_criteria = torch.nn.BCEWithLogitsLoss()
        self.viz_dir = viz_dir
        self.validation_outputs = []
        self.show_results = None

    def configure_optimizers(self):
        # get params and set different learning rate for the backbone ones
        param_groups = [{'params': filter(lambda p: p.requires_grad, self.model.parameters()), 'lr': self.optimizer_cfg['lr']}]
        # if not self.freeze_backbone:

        optimizer = eval(f'torch.optim.{self.optimizer_cfg["type"]}')(
            param_groups,
            lr=self.optimizer_cfg['lr'],
            betas=self.optimizer_cfg['betas'],
            weight_decay=self.optimizer_cfg['weight_decay']
        )

        # Learning rate scheduler (MultiStepLR)
        milestones = self.lr_cfg['step']
        gamma = 0.1
        scheduler = MultiStepLR(optimizer, milestones, gamma)

        return [optimizer], [scheduler]

    def forward(self, x, masks):
        pred = self.model(x)
        return pred

    def training_step(self, batch, batch_idx):
        target = batch
        images = target['image'].to(self.device)
        target_weights = target['target_weight'].to(self.device)
        masks = target['mask'].to(self.device) if self.use_mask else None

        pred = self(images, masks)

        loss = self.criterion(pred, target, target_weights)
        if self.visibility:
            loss['loss_visibility'] = 0.0
            loss['loss_visibility'] = self.vis_criteria(pred['visibility'], target["joints_visibility"])
            loss['loss'] = loss['loss'] + loss['loss_visibility']

        steps = 10 if self.cfg.debug_mode else 2000
        train_path = os.path.join(self.viz_dir, 'train')
        os.makedirs(train_path, exist_ok=True)
        if self.global_step > 0 and self.global_step % steps == 0:
            with torch.no_grad():
                video_path = compare_results_denseldmks2d(images, pred, target, masks, self.global_step, train_path, self.train_data_cfg["normalize_plus_min_one"])
                self.wandb_video_log(video_path, 'train')
        if self.global_step > 0 and self.global_step % self.cfg.log_steps == 0:
            self.tensorboard_logging(loss, self.global_step, train=True)
        self.log('train/loss', loss["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        self.log('train/loss_joints2d', loss["loss_joints2d"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        self.log('train/loss_sigma', loss["loss_sigma"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        if "loss_visibility" in loss:
            self.log('train/loss_visibility', loss["loss_visibility"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        target = batch
        images = target['image'].to(self.device)
        target_weights = target['target_weight'].to(self.device)
        masks = target['mask'].to(self.device) if self.use_mask else None

        pred = self(images, masks)

        loss = self.criterion(pred, target, target_weights)
        if self.visibility:
            loss['loss_visibility'] = 0.0
            loss['loss_visibility'] = self.vis_criteria(pred['visibility'], target["joints_visibility"])
            loss['loss'] = loss['loss'] + loss['loss_visibility']

        self.tensorboard_logging(loss, self.global_step, train=False,)
        self.log('val/loss', loss["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True, batch_size=images.size(0))  # NOTE: , sync_dist=True MAYBE?
        self.log('val/loss_joints2d', loss["loss_joints2d"], on_step=True, on_epoch=True, prog_bar=True, logger=False, sync_dist=True, batch_size=images.size(0))
        self.log('val/loss_sigma', loss["loss_sigma"], on_step=True, on_epoch=True, prog_bar=True, logger=False, sync_dist=True, batch_size=images.size(0))
        if "loss_visibility" in loss:
            self.log('val/loss_visibility', loss["loss_visibility"], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=images.size(0))
        self.show_results = [images, pred, target, masks]
        return loss

    def on_validation_epoch_end(self):
        val_path = os.path.join(self.viz_dir, 'val')
        os.makedirs(val_path, exist_ok=True)
        if self.show_results is not None:
            with torch.no_grad():
                images, pred, target, masks = self.show_results
                video_path = compare_results_denseldmks2d(images, pred, target, masks, self.global_step, val_path, self.val_data_cfg["normalize_plus_min_one"])
                self.wandb_video_log(video_path, 'val')
            self.show_results = None
        self.validation_outputs.clear()

    def wandb_video_log(self, video_path, log_name='train') -> None:
        video = wandb.Video(video_path, format="mp4")
        if isinstance(self.loggers, WandbLogger):
            self.loggers.experiment.log({f"{log_name}/video": video})
        elif isinstance(self.loggers, list):  # in case you're using multiple loggers
            for l in self.loggers:
                if isinstance(l, WandbLogger):
                    l.experiment.log({f"{log_name}/video": video})

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self.train_data_cfg['type'] == 'BEDLAM_WD':
                # with_epoch tells how many data samples are in the dataset per each gpu and it's multiplied by the number of workers
                # To find num of iterations per epoch, we need to divide
                # we divided by the number of GPUs as well to get the number of iterations per epoch
                self.mamma_train = MixedWebDataset(
                                             self.train_data_cfg,
                                             is_train=True,
                                             ).with_epoch(1_000_000//(self.cfg.workers_per_gpu*self.cfg.gpus_n)).shuffle(2000)
                print("Number of iterations per epoch: ", self.mamma_train.nsamples,
                      np.ceil(1_000_000/(self.cfg.workers_per_gpu*self.cfg.gpus_n*self.cfg.samples_per_gpu)*self.cfg.workers_per_gpu))

                self.mamma_val = MixedWebDataset(
                                             self.val_data_cfg,
                                             is_train=False,)  # change so only augmentation is activated
            else:
                raise NotImplementedError(f"Dataset type {self.train_data_cfg['type']} not implemented yet!")

    def train_dataset(self):
        return self.mamma_train

    def train_dataloader(self):
        self.train_ds = self.train_dataset()
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.cfg.samples_per_gpu,
            num_workers=self.cfg.workers_per_gpu,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataset(self):
        if self.train_data_cfg['type'] == 'BEDLAM_WD':
            return self.mamma_val

        return torch.utils.data.TensorDataset(torch.empty(0))

    def val_dataloader(self):
        self.val_ds = self.val_dataset()
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=16 if self.cfg.samples_per_gpu > 20 else 1,#self.cfg.data['samples_per_gpu'],
            num_workers=2 if self.cfg.samples_per_gpu > 20 else 1,# #self.cfg.data['workers_per_gpu'],
            pin_memory=True
        )

    # Tensoroboard logging should run from first rank only
    @pl.utilities.rank_zero.rank_zero_only
    def tensorboard_logging(self, losses, step_count: int, train: bool = True, write_to_summary_writer: bool = True) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'

        if write_to_summary_writer:
            summary_writer = self.logger.experiment
            for loss_name, val in losses.items():
                summary_writer.add_scalar(mode +'/' + loss_name, val.detach().item(), step_count)