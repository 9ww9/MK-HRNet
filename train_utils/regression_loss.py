import torch
import torch.nn as nn
import torch.nn.functional as F


class MRSEWithVisibility(nn.Module):
    def __init__(self, use_target_weight=True):
        super(MRSEWithVisibility, self).__init__()
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        """
        计算带权重的均方根误差
        Args:
            output: 模型输出，形状为 (batch_size, num_joints, 2)
            target: 真实坐标，形状为 (batch_size, num_joints, 2)
            target_weight: 关键点权重，形状为 (batch_size, num_joints, 1)
        Returns:
            loss: 均方根误差
        """
        batch_size = output.size(0)
        num_joints = output.size(1)
        
        # 计算欧氏距离
        diff = output - target
        diff = diff.pow(2).sum(dim=2).sqrt()  # (batch_size, num_joints)
        
        if self.use_target_weight and target_weight is not None:
            diff = diff * target_weight.squeeze(2)
        
        # 计算每个样本的平均误差
        loss = diff.sum() / (batch_size * num_joints)
        
        return loss


class MRSE(nn.Module):
    def __init__(self):
        super(MRSE, self).__init__()

    def forward(self, output, target):
        """
        计算均方根误差
        Args:
            output: 模型输出，形状为 (batch_size, num_joints, 2)
            target: 真实坐标，形状为 (batch_size, num_joints, 2)
        Returns:
            loss: 均方根误差
        """
        batch_size = output.size(0)
        num_joints = output.size(1)
        
        # 计算欧氏距离
        diff = output - target
        diff = diff.pow(2).sum(dim=2).sqrt()  # (batch_size, num_joints)
        
        # 计算每个样本的平均误差
        loss = diff.sum() / (batch_size * num_joints)
        
        return loss 