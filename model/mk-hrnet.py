import torch.nn as nn
import torch
import torch.nn.functional as F


BN_MOMENTUM = 0.1
################################MDFUA########
class tongdao(nn.Module):  #处理通道部分
    # 通道模块初始化，输入通道数为in_channel
    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化，输出大小为1x1
        self.fc = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)  # 1x1卷积用于降维
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数，就地操作以节省内存

    # 前向传播函数
    def forward(self, x):
        b, c, _, _ = x.size()  # 提取批次大小和通道数
        y = self.avg_pool(x)  # 应用自适应平均池化
        y = self.fc(y)  # 应用1x1卷积
        y = self.relu(y)  # 应用ReLU激活
        y = nn.functional.interpolate(y, size=(x.size(2), x.size(3)), mode='nearest')  # 调整y的大小以匹配x的空间维度
        return x * y.expand_as(x)  # 将计算得到的通道权重应用到输入x上，实现特征重校准
class kongjian(nn.Module):
    # 空间模块初始化，输入通道数为in_channel
    def __init__(self, in_channel):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)  # 1x1卷积用于产生空间激励
        self.norm = nn.Sigmoid()  # Sigmoid函数用于归一化

    # 前向传播函数
    def forward(self, x):
        y = self.Conv1x1(x)  # 应用1x1卷积
        y = self.norm(y)  # 应用Sigmoid函数
        return x * y  # 将空间权重应用到输入x上，实现空间激励
class hebing(nn.Module):
    # 合并模块初始化，输入通道数为in_channel
    def __init__(self, in_channel):
        super().__init__()
        self.tongdao = tongdao(in_channel)  # 创建通道子模块
        self.kongjian = kongjian(in_channel)  # 创建空间子模块

    # 前向传播函数
    def forward(self, U):
        U_kongjian = self.kongjian(U)  # 通过空间模块处理输入U
        U_tongdao = self.tongdao(U)  # 通过通道模块处理输入U
        return torch.max(U_tongdao, U_kongjian)  # 取两者的逐元素最大值，结合通道和空间激励

class MDFUA(nn.Module): ##多尺度空洞融合注意力模块。
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):# 初始化多尺度空洞卷积结构模块
        super(MDFUA, self).__init__()
        self.branch1 = nn.Sequential(# 第一分支：使用1x1卷积，保持通道维度不变，不使用空洞
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential( # 第二分支：使用3x3卷积，空洞率为2，可以增加感受野
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=2 * rate, dilation=2 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential( # 第三分支：使用3x3卷积，空洞率为4，进一步增加感受野
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=4 * rate, dilation=4 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(# 第四分支：使用3x3卷积，空洞率为12，最大化感受野的扩展
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=8 * rate, dilation=8 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True) # 第五分支：全局特征提取，使用全局平均池化后的1x1卷积处理
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential( # 合并所有分支的输出，并通过1x1卷积降维
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.Hebing=hebing(in_channel=dim_out*5)# 整合通道和空间特征的合并模块

    def forward(self, x):
        [b, c, row, col] = x.size()
        # 应用各分支
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # 全局特征提取
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        # 合并所有特征
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # 应用合并模块进行通道和空间特征增强
        combine=self.Hebing(feature_cat)
        combine_feature_cat=combine*feature_cat
        # 最终输出经过降维处理
        result = self.conv_cat(combine_feature_cat)

        return result
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class StageModule(nn.Module):
    def __init__(self, input_branches, output_branches, c):
        """
        构建对应stage，即用来融合不同尺度的实现
        :param input_branches: 输入的分支数，每个分支对应一种尺度
        :param output_branches: 输出的分支数
        :param c: 输入的第一个分支通道数
        """
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.input_branches):  # 每个分支上都先通过4个BasicBlock
            w = c * (2 ** i)  # 对应第i个分支的通道数
            branch = nn.Sequential(
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w)
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()  # 用于融合每个分支上的输出
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    # 当输入、输出为同一个分支时不做任何处理
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # 当输入分支j大于输出分支i时(即输入分支下采样率大于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及上采样，方便后续相加
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')
                        )
                    )
                else:  # i > j
                    # 当输入分支j小于输出分支i时(即输入分支下采样率小于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及下采样，方便后续相加
                    # 注意，这里每次下采样2x都是通过一个3x3卷积层实现的，4x就是两个，8x就是三个，总共i-j个
                    ops = []
                    # 前i-j-1个卷积层不用变通道，只进行下采样
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    # 最后一个卷积层不仅要调整通道，还要进行下采样
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 每个分支通过对应的block
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

        # 接着融合不同尺寸信息
        x_fused = []
        for i in range(len(self.fuse_layers)):
            x_fused.append(
                self.relu(
                    sum([self.fuse_layers[i][j](x[j]) for j in range(len(self.branches))])
                )
            )

        return x_fused


class MK_HRNet(nn.Module):
    def __init__(self, base_channel: int = 32, num_joints: int = 24):
        super().__init__()
        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # Stage1
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )

        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, base_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )
        ])

        # Stage2
        self.stage2 = nn.Sequential(
            StageModule(input_branches=2, output_branches=2, c=base_channel)
        )

        # 注意力机制添加到Stage2输出的两个分支上
        self.attention1 = MDFUA(base_channel, base_channel)  # 第一分支的注意力机制
        self.attention2 = MDFUA(base_channel * 2, base_channel * 2)  # 第二分支的注意力机制

        # transition2
        self.transition2 = nn.ModuleList([
            nn.Identity(),  # 直接传递第一个分支
            nn.Identity(),  # 直接传递第二个分支
            nn.Sequential(
                nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )
        ])

        # Stage3
        self.stage3 = nn.Sequential(
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel)
        )

        self.attention3_branch1 = MDFUA(base_channel, base_channel)  # 64通道分支的注意力
        self.attention3_branch2 = MDFUA(base_channel * 2, base_channel * 2)  # 32通道分支的注意力
        self.attention3_branch3 = MDFUA(base_channel * 4, base_channel * 4)  # 16通道分支的注意力

    # transition3
        self.transition3 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(base_channel * 4, base_channel * 8, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channel * 8, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )
        ])

        # self.attention4_branch1 = CoordAtt(base_channel, base_channel)  # 64通道分支的注意力
        # self.attention4_branch2 = CoordAtt(base_channel * 2, base_channel * 2)  # 32通道分支的注意力
        # self.attention4_branch3 = CoordAtt(base_channel * 4, base_channel * 4)  # 16通道分支的注意力
        # self.attention4_branch4 = CoordAtt(base_channel * 8, base_channel * 8)  # 8通道分支的注意力

        # Stage4
        self.stage4 = nn.Sequential(
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=1, c=base_channel)
        )

        # self.stage5 = nn.Sequential(
        #     StageModule(input_branches=4, output_branches=1, c=base_channel)
        # )

        # Final layer
        self.final_layer = nn.Conv2d(base_channel, num_joints, kernel_size=1, stride=1)

    def calculate_leg_length(self, x, reverse_trans):
        # 假设我们有一个从热图中提取关键点的函数

        # keypoints = self.extract_keypoints_from_heatmap(x)
        # keypoints = self.extract_keypoints_from_heatmap(x)*4
        import transforms
        reverse_trans = [reverse_trans]
        keypoints, _ = transforms.get_final_preds(x, reverse_trans, post_processing=True)
        keypoints = torch.from_numpy(keypoints)
        # 从 JSON 文件中获取关节对
        skeleton = [
            [0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7],
            [8, 9], [9, 10], [10, 11], [12, 13], [13, 14], [14, 15],
            [16, 17], [17, 18], [18, 19], [20, 21], [21, 22], [22, 23]
        ]

        # 计算腿长
        batch_size = keypoints.shape[0]
        leg_lengths = torch.zeros((batch_size, len(skeleton)), device=keypoints.device)

        for b in range(batch_size):
            for i, (joint1, joint2) in enumerate(skeleton):
                # 检查索引是否超出范围
                if joint1 < keypoints.shape[1] and joint2 < keypoints.shape[1]:
                    # 计算欧氏距离
                    point1 = keypoints[b, joint1]
                    point2 = keypoints[b, joint2]
                    distance = torch.sqrt(torch.sum((point1 - point2) ** 2))
                    leg_lengths[b, i] = distance
                else:
                    leg_lengths[b, i] = 0.0

        return leg_lengths

    def extract_keypoints_from_heatmap(self, heatmaps, threshold=0.0):
        """从热图中提取关键点坐标"""
        batch_size, num_joints, height, width = heatmaps.shape
        keypoints = torch.zeros((batch_size, num_joints, 2), device=heatmaps.device)

        for b in range(batch_size):
            for j in range(num_joints):
                heatmap = heatmaps[b, j]
                # 找出热图中置信度最高的位置
                confidence, ind = torch.max(heatmap.view(-1), dim=0)

                # 将索引转换为坐标
                y = ind // width
                x = ind % width

                # 如果置信度低于阈值，则认为该关键点无效
                if confidence < threshold:
                    keypoints[b, j] = torch.tensor([0, 0], device=heatmaps.device)
                else:
                    keypoints[b, j] = torch.tensor([x.float(), y.float()], device=heatmaps.device)

        return keypoints

    def forward(self, x, reverse_trans = None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list

        x = self.stage2(x)
        # 开始应用注意力机制
        x[0] = self.attention1(x[0])  # 对第一个分支应用注意力机制
        x[1] = self.attention2(x[1])  # 对第二个分支应用注意力机制

        # 后续阶段
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]

        x = self.stage3(x)
        x[0] = self.attention3_branch1(x[0])  # 对 64 通道分支应用注意力机制
        x[1] = self.attention3_branch2(x[1])  # 对 32 通道分支应用注意力机制
        x[2] = self.attention3_branch3(x[2])  # 对 16 通道分支应用注意力机制

        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1]),
        ]

        x = self.stage4(x)
        x = self.final_layer(x[0])
        len = self.calculate_leg_length(x,reverse_trans)
        return x, len
