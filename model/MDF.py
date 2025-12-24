import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelModule(nn.Module):  # Channel attention module
    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    # Forward propagation for channel attention
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        y = self.relu(y)
        y = nn.functional.interpolate(y, size=(x.size(2), x.size(3)), mode='nearest')
        return x * y.expand_as(x)  # Feature recalibration with channel weights

class SpatialModule(nn.Module):  # Spatial attention module
    def __init__(self, in_channel):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    # Forward propagation for spatial attention
    def forward(self, x):
        y = self.Conv1x1(x)
        y = self.norm(y)
        return x * y  # Spatial excitation with spatial weights

class MergeModule(nn.Module):  # Merge channel and spatial attention
    def __init__(self, in_channel):
        super().__init__()
        self.channel_module = ChannelModule(in_channel)
        self.spatial_module = SpatialModule(in_channel)

    # Forward propagation for attention fusion
    def forward(self, U):
        U_spatial = self.spatial_module(U)
        U_channel = self.channel_module(U)
        return torch.max(U_channel, U_spatial)  # Element-wise max fusion

class MDF(nn.Module):  # Multi-scale Dilated Fusion Attention Module
    # dim_in: input channels, dim_out: output channels, rate: dilation rate, bn_mom: BN momentum
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(MDA, self).__init__()
        # Branch 1: 1x1 conv (no dilation)
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # Branch 2: 3x3 conv (dilation rate = 2*rate)
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=2 * rate, dilation=2 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # Branch 3: 3x3 conv (dilation rate = 4*rate)
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=4 * rate, dilation=4 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # Branch 4: 3x3 conv (dilation rate = 8*rate)
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=8 * rate, dilation=8 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # Branch 5: Global feature extraction
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        # Concatenate and reduce dimension
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.merge_module = MergeModule(in_channel=dim_out*5)

    def forward(self, x):
        [b, c, row, col] = x.size()
        # Multi-scale feature extraction
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        
        # Global feature extraction
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        
        # Feature concatenation
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        
        # Attention fusion
        combine = self.merge_module(feature_cat)
        combine_feature_cat = combine * feature_cat
        
        # Final dimension reduction
        result = self.conv_cat(combine_feature_cat)

        return result
