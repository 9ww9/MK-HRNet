import torch


class KpLoss(object):
    def __init__(self, leg_weight=0.5):
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.leg_criterion = torch.nn.MSELoss(reduction='mean')
        self.leg_weight = leg_weight

    def __call__(self, outputs, targets):
        # 现在outputs可能是元组(heatmaps, leg_lengths)或者仅热图
        if isinstance(outputs, tuple) and len(outputs) == 2:
            logits, leg_lengths = outputs
        else:
            # 如果不是元组，假设仅提供热图
            raise ValueError("模型输出必须是包含热图和腿长的元组 (logits, leg_lengths)")

        # 验证热图形状
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        
        device = logits.device
        bs = logits.shape[0]
        # [num_kps, H, W] -> [B, num_kps, H, W]
        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])
        # [num_kps] -> [B, num_kps]
        kps_weights = torch.stack([t["kps_weights"].to(device) for t in targets])

        # [B, num_kps, H, W] -> [B, num_kps]
        loss = self.criterion(logits, heatmaps).mean(dim=[2, 3])
        loss = torch.sum(loss * kps_weights) / bs
        
        # 处理腿长损失，并确保腿长标签存在
        if "leg_lengths" not in targets[0]:
            raise ValueError("训练样本中缺少腿长标签 'leg_lengths'，请检查数据处理步骤")
            
        target_leg_lengths = torch.stack([t["leg_lengths"].to(device) for t in targets])
        leg_loss = self.leg_criterion(leg_lengths, target_leg_lengths)
        
        # 组合两种损失
        total_loss = loss + self.leg_weight * leg_loss
        return total_loss
