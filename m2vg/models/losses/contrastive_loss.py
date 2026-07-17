# 文件路径：m2vg/models/losses/contrastive_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class WordRegionContrastiveLoss(nn.Module):
    def __init__(self, temp=0.07):
        super().__init__()
        self.temp = temp  # 温度超参数，用于缩放相似度

    def forward(self, img_feat, text_feat, text_attention_mask, gt_exist):
        """
        img_feat: (B, C, H, W) - 视觉特征
        text_feat: (B, L, C) - 文本Token特征
        text_attention_mask: (B, L) - 文本有效Mask (1为有效，0为Pad)
        gt_exist: (B, 1) 或 List[Tensor] - 存在性标签 (1: 匹配, 0: 难负样本)
        """
        # 1. 统一 gt_exist 格式并送到正确设备
        if isinstance(gt_exist, list):
            gt_exist = torch.stack(gt_exist).float().view(-1).to(text_feat.device)
        else:
            gt_exist = gt_exist.float().view(-1).to(text_feat.device)

        B = text_feat.shape[0]

        # 2. 视觉全局特征提取 (Region -> Global Image Feature)
        if img_feat.dim() == 4:  # 形状为 (B, C, H, W)
            v_feat = img_feat.flatten(2).mean(dim=2)  # (B, C)
        else:
            v_feat = img_feat.mean(dim=1)

        # 3. 文本特征提取 (Word -> Global Text Feature)
        if text_attention_mask is not None:
            mask = text_attention_mask.unsqueeze(-1).float()
            # 屏蔽掉 Padding 的无效词，求有效词的平均特征
            t_feat = (text_feat * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)  # (B, C)
        else:
            t_feat = text_feat.mean(dim=1)

        # 4. L2 特征归一化 (将特征投影到单位球面上)
        v_feat = F.normalize(v_feat, p=2, dim=-1)
        t_feat = F.normalize(t_feat, p=2, dim=-1)

        # 5. 计算直接对齐损失 (Alignment Loss) —— 难负样本的克星！
        # 计算每对图像-文本的余弦相似度（对角线元素）
        sim_diag = (v_feat * t_feat).sum(dim=-1)  # (B,)
        logit_diag = sim_diag / self.temp

        # 使用 BCE：如果 gt_exist=1，逼迫相似度向正无穷(1)靠拢；
        # 如果 gt_exist=0，逼迫相似度向负无穷(-1)靠拢，彻底推开不匹配的特征！
        loss_align = F.binary_cross_entropy_with_logits(logit_diag, gt_exist)

        # 6. 计算 InfoNCE 对比损失 (Batch Contrastive Loss)
        # 仅利用 Batch 内的正样本，促使整个特征空间排布更合理
        valid_mask = (gt_exist == 1.0)
        loss_nce = torch.tensor(0.0, device=text_feat.device)

        if valid_mask.sum() > 0:
            # 计算全局相似度矩阵 (B, B)
            sim_matrix = torch.matmul(v_feat, t_feat.transpose(0, 1)) / self.temp
            labels = torch.arange(B, dtype=torch.long, device=text_feat.device)

            # 双向对比学习：图找文，文找图
            loss_i2t = F.cross_entropy(sim_matrix[valid_mask], labels[valid_mask])
            loss_t2i = F.cross_entropy(sim_matrix.transpose(0, 1)[valid_mask], labels[valid_mask])
            loss_nce = (loss_i2t + loss_t2i) / 2.0

        # 综合返回：既有强力的对角线推斥，又有全局的 InfoNCE 拉扯
        return loss_align + 0.5 * loss_nce