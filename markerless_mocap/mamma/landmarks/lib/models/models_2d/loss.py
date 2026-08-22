# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

class JointGNLLLoss(nn.Module):
    """GNLL loss for SMPL-X parameters.

    Args:
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, loss_weights=1.):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.criterion_noreduce = nn.MSELoss(reduction='none')

        for name, weight in loss_weights.items():
            setattr(self, f'lw_{name}', weight)

    def forward(self, output, target, weights=None, with_sigma=False, kpts_loss_thresh=25):
        """Compute a (probabilistic) landmark/keypoint loss.

        Args:
            pred: Tensor shape (Batch, num_Keypoints * 3) if probabilistic and (Batch, num_Keypoints * 2) otherwise
            label_coords: Tensor shaped (Batch, num-Keypoints, 2) of keypoint coordinates
            weights: Tensor shaped (Batch, num-Keypoints) of per-keypoint weights
            with_sigma: Should probabilistic version of the loss be computed
            kpts_loss_thresh: Max value kpts_loss entries should take (set to very high if don't want thresholding)
        Returns:
            scalar loss
        """
        batch_size = len(output['joints2d'])

        # if with_sigma:
        num_landmarks = output['joints2d'].shape[1]
        num_dims = output['joints2d'].shape[2]
        pred_coords = output['joints2d']

        kpts_diffs = target['joints'] - pred_coords[:, :, :2]  # shape: (B, K, 2)
        kpts_sq_diffs = torch.sum(torch.square(kpts_diffs), axis=-1)  # shape: (B, K)
        if weights is None:
            kpts_sq_diffs_weighted = kpts_sq_diffs
        else:
            kpts_sq_diffs_weighted = torch.mul(kpts_sq_diffs, weights[:,:,0])

        if num_dims == 3:
            eps = torch.tensor(1e-6).to(pred_coords.device)
            pred_log_sigmas = pred_coords[:, :, -1]

            #clip sigmas
            pred_log_sigmas = torch.clip(pred_log_sigmas, min=torch.log(eps), max=None)
            pred_sigmas = torch.exp(pred_log_sigmas)

            keypoint_2_sigma_sq = 2.0 * torch.square(pred_sigmas)
            kpts_sq_diffs_over_2sigmasq = kpts_sq_diffs_weighted * (1.0 / keypoint_2_sigma_sq)

            # Clip kpts_loss to a maximum value as it can be very unstable due to (1.0 / keypoint_2_sigma_sq)
            kpts_loss = torch.mean(torch.clip(kpts_sq_diffs_over_2sigmasq, min=None, max=kpts_loss_thresh))
            if weights is None:
                sigmas_loss = torch.mean(2 * pred_log_sigmas)
            else:
                sigmas_loss = torch.mean(2 * torch.mul(pred_log_sigmas, weights[:,:,0]))
            loss = self.lw_joints2d*(kpts_loss + sigmas_loss)

        else:
            kpts_loss = torch.mean(kpts_sq_diffs_weighted)
            loss = self.lw_joints2d*kpts_loss
            sigmas_loss = torch.tensor(0.0).to(kpts_loss.device)

        total_loss = dict(
            loss_joints2d = torch.mean(kpts_sq_diffs_weighted),
            loss = loss*self.lw_total,
            loss_sigma=sigmas_loss,
        )
        return total_loss

def focal_loss(pred, target, alpha=0.9, gamma=2.0):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p = torch.sigmoid(pred)
    pt = target * p + (1 - target) * (1 - p)
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()