import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.modules.utils import _triple
from .superpoint import Superpoint

NLMODE = 'dot'  # v['gaussian', 'embedded_gaussian', 'dot', 'concatenate']


class SpatioTemporalConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SpatioTemporalConv, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
        spatial_stride = [1, stride[1], stride[2]]
        spatial_padding = [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride = [stride[0], 1, 1]
        temporal_padding = [padding[0], 0, 0]

        intermed_channels = int(
            math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) / \
                       (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                      stride=spatial_stride, padding=spatial_padding, bias=bias)

        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()

        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                       stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x


class SpatioTemporalResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()

        self.downsample = downsample

        padding = kernel_size // 2

        if self.downsample:
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)

            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)


class SpatioTemporalResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, layer_size, non_local_mode=NLMODE,
                 block_type=SpatioTemporalResBlock, downsample=False, non_local=False):

        super(SpatioTemporalResLayer, self).__init__()

        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        self.blocks = nn.ModuleList([])

        last_idx = layer_size
        if non_local:
            last_idx = layer_size - 1

        for i in range(1, last_idx):
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

        # add non-local block here
        # if non_local:
        #     self.blocks += [NONLocalBlock3D(in_channels=out_channels, mode=non_local_mode)]
        #     self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class R2Plus1DNet(nn.Module):

    def __init__(self, input_channel, non_local, SPtype, layer_sizes=[2, 2, 2, 2],
                 block_type=SpatioTemporalResBlock):
        super(R2Plus1DNet, self).__init__()
        if SPtype == 'None':
            self.conv1 = SpatioTemporalConv(input_channel, 64, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        else:
            self.conv1 = SpatioTemporalConv(input_channel, 64, [3, 3, 3], stride=[1, 1, 1], padding=[1, 3, 3])
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True,
                                            non_local=non_local)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True,
                                            non_local=non_local)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)
        self.pool = nn.AdaptiveAvgPool3d(1)


class BasicBlock_2d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_2d, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_2d(nn.Module):
    def __init__(self, block=BasicBlock_2d, num_blocks=[2, 2, 2, 2]):
        super(ResNet_2d, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5,
                               stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


class DevianceNet(nn.Module):
    def __init__(self, args, device, Res2d=ResNet_2d()):
        super(DevianceNet, self).__init__()
        self.classifier_type = args.classifier_type
        self.drop_p = args.dropout
        self.SPtype = args.superpoint_type
        self.nl = args.non_local

        config_sp = {
            'top_k_keypoints': args.num_kps,
            'height': args.img_y,
            'width': args.img_x,
            'align_corners': args.align_corners,
            'detection_threshold': args.detection_threshold,
            'frac_superpoint': args.frac_superpoint,
            'nms_radius': args.nms_radius,
        }

        if self.SPtype == 'mul' or self.SPtype == 'con':
            self.supernet = Superpoint(config_sp).to(device)
            if self.SPtype == 'mul':
                self.input_channel = 256
            else:
                self.input_channel = 705
        else:
            self.input_channel = 3

        # load pretrained-supernet
        if args.weight_load_pth is None:
            weights = torch.load('../weight_file/pretrained_superpoint.pth.tar', map_location=torch.device('cpu'))
            self.supernet.load_state_dict(weights['state_dict'], strict=True)

        Res2p1d = R2Plus1DNet(self.input_channel, self.nl, self.SPtype)

        self.in_planes = 64
        if self.SPtype == 'None':
            self.conv1_2d = nn.Conv2d(self.input_channel, self.in_planes, kernel_size=(7, 7), stride=(2, 2),
                                      padding=(3, 3), bias=False)
        else:
            self.conv1_2d = nn.Conv2d(self.input_channel, self.in_planes, kernel_size=(3, 3), stride=(1, 1),
                                      padding=(3, 3), bias=False)

        self.bn1_2d = nn.BatchNorm2d(self.in_planes)

        self.maxpool_2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool_3d = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1_2d = Res2d.layer1
        self.layer2_2d = Res2d.layer2
        self.layer3_2d = Res2d.layer3
        self.layer4_2d = Res2d.layer4

        self.layer0_3d = Res2p1d.conv1
        self.layer1_3d = Res2p1d.conv2
        self.layer2_3d = Res2p1d.conv3
        self.layer3_3d = Res2p1d.conv4
        self.layer4_3d = Res2p1d.conv5

        self.reduction_conv3d_layer1 = nn.Conv3d(64 * 2, 64, kernel_size=(1, 1, 1))
        self.reduction_conv3d_layer2 = nn.Conv3d(128 * 2, 128, kernel_size=(1, 1, 1))
        self.reduction_conv3d_layer3 = nn.Conv3d(256 * 2, 256, kernel_size=(1, 1, 1))
        self.reduction_conv3d_layer4 = nn.Conv3d(512 * 2, 512, kernel_size=(1, 1, 1))

        self.avgpool_2d = Res2d.avgpool
        self.avgpool_3d = Res2p1d.pool

        self.relu = nn.ReLU(inplace=True)

        # superpoint
        self.merge_features = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1, 1))
        self.merge_descriptors = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.merge_scores = nn.Sequential(
            nn.Conv2d(in_channels=65, out_channels=256, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.merge_conv1 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1))

        self.fc1 = nn.Linear(512, 64)

        if self.classifier_type == 'SEA':
            self.classification = nn.Linear(64, 5)
        elif self.classifier_type == 'DIA':
            self.detection = nn.Linear(64, 2)
        elif self.classifier_type == 'SEA_DIA':
            self.classification = nn.Linear(64, 5)
            self.detection = nn.Linear(64, 2)
        else:
            sys.exit()

    def conv_2D_block(self, x, x_3d, f):
        x_2d = []

        for t in range(x.size(2)):
            tmp = f(x[:, :, t, :, :])
            x_2d.append(tmp)
        x_2d = torch.stack(x_2d, dim=0)
        x_2d = x_2d.permute(1, 2, 0, 3, 4)
        x_2d = F.adaptive_avg_pool3d(x_2d, (x_3d.size(2), x_2d.size(3), x_2d.size(4)))
        return x_2d

    def MR_block(self, x_2d, x_3d, reductionconv):
        return reductionconv(torch.cat((x_2d, x_3d), dim=1))

    def forward(self, x):
        if self.SPtype == 'mul':
            XX = []
            for idx in range(x.size(0)):
                data_sp = {'img': x[idx].permute(1, 0, 2, 3), 'process_tsp': 'ts'}
                pred = self.supernet(data_sp)
                X_1 = self.merge_features(pred['skip_eight'])
                X_2 = self.merge_scores(pred['scores'])
                X_3 = self.merge_descriptors(pred['descriptors'])
                X_merge = X_1.mul(X_2).mul(X_3)
                XX.append(X_merge)
            XX = torch.stack(XX, dim=0)
            x = XX.permute(0, 2, 1, 3, 4)

        if self.SPtype == 'con':
            XX = []
            for idx in range(x.size(0)):
                data_sp = {'img': x[idx].permute(1, 0, 2, 3), 'process_tsp': 'ts'}
                pred = self.supernet(data_sp)
                X_1 = self.merge_conv1(pred['features'])
                X_1 = F.interpolate(X_1, scale_factor=4.0, mode='bilinear', align_corners=False)
                X_merge = torch.cat([X_1, pred['descriptors'], pred['scores']], dim=1)

                XX.append(X_merge)
            XX = torch.stack(XX, dim=0)
            x = XX.permute(0, 2, 1, 3, 4)

        # layer0
        if self.SPtype == 'None':
            x_3d = self.maxpool_3d(self.layer0_3d(x))
            x_2d = self.conv_2D_block(x, x_3d, nn.Sequential(self.conv1_2d, self.bn1_2d, self.relu, self.maxpool_2d))
        else:
            x_3d = self.layer0_3d(x)
            x_2d = self.conv_2D_block(x, x_3d, nn.Sequential(self.conv1_2d, self.bn1_2d, self.relu))

        # layer1
        x_3d = self.layer1_3d(x_3d)  #
        x_2d = self.conv_2D_block(x_2d, x_3d, self.layer1_2d)
        x_3d = self.MR_block(x_2d, x_3d, self.reduction_conv3d_layer1)

        # layer2
        x_3d = self.layer2_3d(x_3d)
        x_2d = self.conv_2D_block(x_2d, x_3d, self.layer2_2d)
        x_3d = self.MR_block(x_2d, x_3d, self.reduction_conv3d_layer2)

        # layer3
        x_3d = self.layer3_3d(x_3d)
        x_2d = self.conv_2D_block(x_2d, x_3d, self.layer3_2d)
        x_3d = self.MR_block(x_2d, x_3d, self.reduction_conv3d_layer3)

        # layer4
        x_3d = self.layer4_3d(x_3d)
        x_2d = self.conv_2D_block(x_2d, x_3d, self.layer4_2d)
        x_3d = self.MR_block(x_2d, x_3d, self.reduction_conv3d_layer4)

        # layer5 - FC
        out = self.avgpool_3d(x_3d)

        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=self.drop_p, training=self.training)
        out = self.fc1(out)
        out = F.dropout(out, p=self.drop_p / 2, training=self.training)
        if self.classifier_type == 'SEA':
            output_classification = self.classification(out)
            return output_classification
        elif self.classifier_type == 'DIA':
            output_detection = self.detection(out)
            return output_detection
        elif self.classifier_type == 'SEA_DIA':
            output_classification = self.classification(out)
            output_detection = self.detection(out)
            return output_classification, output_detection
