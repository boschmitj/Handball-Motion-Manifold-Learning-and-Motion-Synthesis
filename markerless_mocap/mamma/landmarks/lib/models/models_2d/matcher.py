"""
Instance Sequence Matching
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

INF = 1e8

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, num_frames : int = 36, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_frames = num_frames
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the sequence level matching
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []
        for i in range(bs):
            out_prob = outputs["pred_logits"][i].softmax(-1)
            out_bbox = outputs["pred_boxes"][i]
            tgt_ids = targets[i]["labels"]
            tgt_bbox = targets[i]["boxes"]
            tgt_valid = targets[i]["valid"]
            num_out = 10
            num_tgt = len(tgt_ids)//self.num_frames
            out_prob_split = out_prob.reshape(self.num_frames,num_out,out_prob.shape[-1]).permute(1,0,2)
            out_bbox_split = out_bbox.reshape(self.num_frames,num_out,out_bbox.shape[-1]).permute(1,0,2).unsqueeze(1)
            tgt_bbox_split = tgt_bbox.reshape(num_tgt,self.num_frames,4).unsqueeze(0)
            tgt_valid_split = tgt_valid.reshape(num_tgt,self.num_frames)
            frame_index = torch.arange(start=0,end=self.num_frames).repeat(num_tgt).long()
            class_cost = -1 * out_prob_split[:,frame_index,tgt_ids].view(num_out,num_tgt,self.num_frames).mean(dim=-1)
            bbox_cost = (out_bbox_split-tgt_bbox_split).abs().mean((-1,-2))
            iou_cost = -1 * multi_iou(box_cxcywh_to_xyxy(out_bbox_split),box_cxcywh_to_xyxy(tgt_bbox_split)).mean(-1)
            cost = self.cost_class*class_cost + self.cost_bbox*bbox_cost + self.cost_giou*iou_cost
            out_i, tgt_i = linear_sum_assignment(cost.cpu())
            index_i,index_j = [],[]
            for j in range(len(out_i)):
                tgt_valid_ind_j = tgt_valid_split[j].nonzero().flatten()
                index_i.append(tgt_valid_ind_j*num_out + out_i[j])
                index_j.append(tgt_valid_ind_j + tgt_i[j]* self.num_frames)
            if index_i==[] or index_j==[]:
                indices.append((torch.tensor([]).long().to(out_prob.device),torch.tensor([]).long().to(out_prob.device)))
            else:
                index_i = torch.cat(index_i).long()
                index_j = torch.cat(index_j).long()
                indices.append((index_i,index_j))
        return indices

# def build_matcher(args):


class LandmarkHungarianMatcher(nn.Module):
    """
    Hungarian matcher for multi-view 2D landmark predictions.
    Matches predicted persons ↔ GT persons over all views using
    classification (presence) and landmark alignment costs.
    """

    def __init__(self, cost_class=1.0, cost_l1=1.0, sigma=None):
        super().__init__()
        self.cost_class = cost_class
        self.cost_l1 = cost_l1
        self.sigma = sigma  # per-landmark tolerance (optional, for OKS-like)
        self.num_frames = 8

        assert cost_class != 0 or cost_l1 != 0, "all costs cannot be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs:
          - "joints2d": [B, V, P_pred, N, 2]
          - "visibility": [B, V, P_pred]  (presence prob per view)
        targets:
          - "joints": [B, V, P_gt, N, 2]
          - "is_visible":  [B, V, P_gt] (0/1)
        returns:
          list of (index_pred, index_gt) tuples per batch
        """

        bs, V, P_pred, N, _ = outputs["joints2d"].shape
        indices = []

        for b in range(bs):
            pred_xy = outputs["joints2d"][b]     # [V, P_pred, N, 2]
            pred_prob = nn.functional.sigmoid(outputs["visibility"][b])      # [V, P_pred]
            gt_xy = targets["joints"][b]         # [V, P_gt, N, 2]
            gt_vis = targets["is_visible"][b]          # [V, P_gt]
            tgt_valid_split = gt_vis.transpose(1,0).contiguous()

            V, P_gt = gt_vis.shape

            # ---------- Classification cost ----------
            # For each (pred_person, gt_person), compute -log(pred_prob) averaged over visible views
            # pred_prob: [V, P_pred], gt_vis: [V, P_gt]
            prob = pred_prob.clamp(1e-6, 1 - 1e-6)
            class_cost = -(gt_vis[:, None, :] * torch.log(prob[:, :, None])).sum(0)  # [P_pred, P_gt]

            class_denom = gt_vis.sum(0).clamp_min(1.0)
            class_cost = class_cost / class_denom  # average across visible views

            # ---------- Landmark cost ----------
            # L1 distance per landmark averaged over visible views
            # pred_xy: [V, P_pred, N, 2], gt_xy: [V, P_gt, N, 2]
            diff = (pred_xy[:, :, None] - gt_xy[:, None])  # [V, P_pred, P_gt, N, 2]
            l1 = diff.abs().sum(-1)                        # [V, P_pred, P_gt, N]

            vis = gt_vis[:, None, :, None]                 # [V, 1, P_gt, 1]
            l1 = (l1 * vis).sum((0, 3)) / vis.sum((0, 3)).clamp_min(1.0)  # [P_pred, P_gt]
            l1_cost = l1

            # ---------- Total cost ----------

            # No class cost as the class is always "person" and it is in reality a presence score
            cost = self.cost_l1 * l1_cost

            # Hungarian assignment
            cost_cpu = cost.detach().cpu()
            pred_ind, gt_ind = linear_sum_assignment(cost_cpu)
            index_i, index_j = [], []

            for j in range(len(pred_ind)):
                tgt_valid_ind_j = tgt_valid_split[j].nonzero().flatten()
                index_i.append(tgt_valid_ind_j*2 + pred_ind[j])
                index_j.append(tgt_valid_ind_j + gt_ind[j]* self.num_frames)

            if index_i==[] or index_j==[]:
                indices.append((torch.tensor([]).long().to(tgt_valid_split.device),torch.tensor([]).long().to(tgt_valid_split.device)))
            else:
                index_i = torch.cat(index_i).long()
                index_j = torch.cat(index_j).long()
                indices.append((index_i, index_j))
        return indices


def build_matcher(cost_class, cost_l1):
    return LandmarkHungarianMatcher(
        cost_class=cost_class,
        cost_l1=cost_l1
    )