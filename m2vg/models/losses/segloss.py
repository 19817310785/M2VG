import torch
from torch.nn import functional as F
import torch


def boundary_dou_loss(inputs, targets, alpha=0.8):
    """
    Boundary DoU Loss (MICCAI 2023)
    替代传统的 Dice Loss，极致增强边缘与小目标分割
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)

    # 计算交集与并集
    intersection = (inputs * targets).sum(1)
    union = inputs.sum(1) + targets.sum(1) - intersection

    # 差异集 (Difference) = 并集 - 交集 (即预测错误的所有像素，绝大部分分布在边界)
    difference = union - intersection

    # Boundary DoU 公式: difference / (difference + (1 - alpha) * intersection + 1e-8)
    # alpha 越大，网络对内部像素的容忍度越高，从而将所有梯度注意力逼迫到边界上
    loss = difference / (difference + (1.0 - alpha) * intersection + 1e-8)

    return loss.mean()


def poly_bce_loss(inputs, targets, epsilon=2.0):
    """
    PolyLoss (ICLR 2022)
    基于泰勒展开改进 BCE，增强对 Hard Pixels 的学习
    """
    # 基础的 BCE Loss
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

    # 计算预测概率 pt
    prob = inputs.sigmoid()
    pt = targets * prob + (1 - targets) * (1 - prob)

    # PolyLoss 公式: CE + epsilon * (1 - pt)
    # 增加多项式领先项，强化对预测不准像素的梯度
    poly_loss = ce_loss + epsilon * (1 - pt)

    return poly_loss.mean()



def refer_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor):

    loss = F.cross_entropy(inputs, targets, weight=weight)
    return loss


def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """

    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


def sigmoid_ce_loss(inputs, targets):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")
    return ce_loss


def seg_loss(inputs, target, loss_info):
    loss_seg = torch.tensor([0.0], device=inputs.device)
    target = target.float().unsqueeze(1)
    assert target.shape == inputs.shape

    # --- 传统的 Baseline Loss (通过 config 控制是否开启) ---
    if "dice" in loss_info:
        loss_seg += dice_loss(inputs, target) * loss_info["dice"]
    if "bce" in loss_info:
        loss_seg += sigmoid_ce_loss(inputs, target) * loss_info["bce"]

    # --- 你的创新组合 Loss (通过 config 控制是否开启) ---
    if "dou" in loss_info:
        loss_seg += boundary_dou_loss(inputs, target) * loss_info["dou"]
    if "poly" in loss_info:
        loss_seg += poly_bce_loss(inputs, target) * loss_info["poly"]

    return loss_seg

def part_seg_loss(inputs, targets, indices, loss_info):
    loss_seg = 0.0
    total_pred_masks_pos = []
    total_gt_masks_pos = []
    total_pred_masks_neg = []
    total_gt_mask_neg = []
    for pred_mask, gt_part_mask, indice in zip(inputs, targets, indices[-1]):
        neg_mask = torch.ones(pred_mask.size(0), dtype=torch.bool)
        neg_mask[indice[0]] = False
        pred_mask_neg = pred_mask[neg_mask]
        gt_masks_neg = torch.zeros((pred_mask.shape[0] - len(indice[0]), pred_mask.shape[-2], pred_mask.shape[-1]), dtype=torch.float).to(pred_mask.device)
        total_pred_masks_neg.append(pred_mask_neg)
        total_gt_mask_neg.append(gt_masks_neg)
        if len(indice) == 0:
            total_pred_masks_pos.append(torch.zeros((0, pred_mask.shape[-2], pred_mask.shape[-1]), dtype=torch.float).to(pred_mask.device))
            total_gt_masks_pos.append(torch.zeros((0, pred_mask.shape[-2], pred_mask.shape[-1]), dtype=torch.float).to(pred_mask.device))
            continue
        pred_mask_pos = pred_mask[indice[0]]
        gt_mask_pos = torch.tensor(gt_part_mask[indice[1].cpu().detach().numpy().tolist()]).to(pred_mask.device)
        total_pred_masks_pos.append(pred_mask_pos)
        total_gt_masks_pos.append(gt_mask_pos)
    pred_masks_pos = torch.concat(total_pred_masks_pos, dim=0)
    pred_mask_neg = torch.concat(total_pred_masks_neg, dim=0)
    gt_masks_pos = torch.concat(total_gt_masks_pos, dim=0)
    gt_masks_neg = torch.concat(total_gt_mask_neg, dim=0)
    assert pred_masks_pos.shape == gt_masks_pos.shape
    assert pred_mask_neg.shape == gt_masks_neg.shape
    loss_seg_pos = seg_loss(pred_masks_pos.unsqueeze(1), gt_masks_pos, loss_info)
    if loss_info["neg"] ==0:
        loss_seg = loss_seg_pos
    else:
        loss_seg_neg = seg_loss(pred_mask_neg.unsqueeze(1), gt_masks_neg, loss_info) * loss_info["neg"]
        loss_seg = loss_seg_pos + loss_seg_neg
    return loss_seg