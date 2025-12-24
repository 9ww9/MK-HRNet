import torch.nn as nn
import torch
import torch.nn.functional as F
import MDF

BN_MOMENTUM = 0.1

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
        Build the stage module for multi-scale feature fusion
        :param input_branches: Number of input branches (each for one scale)
        :param output_branches: Number of output branches
        :param c: Channel number of the first input branch
        """
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        # Each branch passes through 4 BasicBlocks
        for i in range(self.input_branches):
            w = c * (2 ** i)  # Channel number for the i-th branch
            branch = nn.Sequential(
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w)
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()  # Layers for multi-branch fusion
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    # Identity mapping for same input/output branch
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # Upsample + channel adjustment (input branch j has higher downsampling rate)
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')
                        )
                    )
                else:  # i > j
                    # Downsample + channel adjustment (input branch j has lower downsampling rate)
                    ops = []
                    # First (i-j-1) conv layers: downsample only (no channel change)
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    # Last conv layer: downsample + channel adjustment
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Process each branch with corresponding BasicBlocks
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

        # Fuse multi-scale features
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

        self.attention1 = MDF(base_channel, base_channel)
        self.attention2 = MDF(base_channel * 2, base_channel * 2) 

        # transition2
        self.transition2 = nn.ModuleList([
            nn.Identity(),  
            nn.Identity(), 
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

        self.attention3_branch1 = MDF(base_channel, base_channel)  
        self.attention3_branch2 = MDF(base_channel * 2, base_channel * 2) 
        self.attention3_branch3 = MDF(base_channel * 4, base_channel * 4)  

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
        keypoints = self.extract_keypoints_from_heatmap(x)*4
        # Get joint pairs from JSON file
        skeleton = [
            [0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7],
            [8, 9], [9, 10], [10, 11], [12, 13], [13, 14], [14, 15],
            [16, 17], [17, 18], [18, 19], [20, 21], [21, 22], [22, 23]
        ]

        # Calculate leg lengths
        batch_size = keypoints.shape[0]
        leg_lengths = torch.zeros((batch_size, len(skeleton)), device=keypoints.device)

        for b in range(batch_size):
            for i, (joint1, joint2) in enumerate(skeleton):
                # Check if indices are out of range
                if joint1 < keypoints.shape[1] and joint2 < keypoints.shape[1]:
                    # Calculate Euclidean distance
                    point1 = keypoints[b, joint1]
                    point2 = keypoints[b, joint2]
                    distance = torch.sqrt(torch.sum((point1 - point2) ** 2))
                    leg_lengths[b, i] = distance
                else:
                    leg_lengths[b, i] = 0.0

        return leg_lengths

    def extract_keypoints_from_heatmap(self, heatmaps, threshold=0.0):
        batch_size, num_joints, height, width = heatmaps.shape
        keypoints = torch.zeros((batch_size, num_joints, 2), device=heatmaps.device)

        for b in range(batch_size):
            for j in range(num_joints):
                heatmap = heatmaps[b, j]
                # Find the position with the highest confidence in the heatmap
                confidence, ind = torch.max(heatmap.view(-1), dim=0)

                # Convert index to coordinates
                y = ind // width
                x = ind % width

                # If confidence is below the threshold, the keypoint is considered invalid
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
        x[0] = self.attention1(x[0]) 
        x[1] = self.attention2(x[1]) 

        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]

        x = self.stage3(x)
        x[0] = self.attention3_branch1(x[0]) 
        x[1] = self.attention3_branch2(x[1]) 
        x[2] = self.attention3_branch3(x[2]) 

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
