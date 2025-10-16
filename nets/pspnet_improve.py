from copy import deepcopy

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torchstat import stat

# from torchsummary import summary

from nets.Attention_improve import SE_Block, CBAM_block
from nets.mobilenetv2 import mobilenetv2
from nets.pspnet_training import CE_Loss
from nets.resnet import resnet50


class Resnet(nn.Module):
    def __init__(self, dilate_scale=8, pretrained=True):
        super(Resnet, self).__init__()
        from functools import partial
        model = resnet50(pretrained)

        # --------------------------------------------------------------------------------------------#
        #   根据下采样因子修改卷积的步长与膨胀系数
        #   当downsample_factor=16的时候，我们最终获得两个特征层，shape分别是：30,30,1024和30,30,2048
        # --------------------------------------------------------------------------------------------#
        if dilate_scale == 8:
            model.layer3.apply(partial(self._nostride_dilate, dilate=2))
            model.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            model.layer4.apply(partial(self._nostride_dilate, dilate=2))

        self.conv1 = model.conv1[0]
        self.bn1 = model.conv1[1]
        self.relu1 = model.conv1[2]
        self.conv2 = model.conv1[3]
        self.bn2 = model.conv1[4]
        self.relu2 = model.conv1[5]
        self.conv3 = model.conv1[6]
        self.bn3 = model.bn1
        self.relu3 = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)
        return x_aux, x
        # return x

class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        # --------------------------------------------------------------------------------------------#
        #   根据下采样因子修改卷积的步长与膨胀系数
        #   当downsample_factor=16的时候，我们最终获得两个特征层，shape分别是：30,30,320和30,30,96
        # --------------------------------------------------------------------------------------------#
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x_aux = self.features[:14](x)
        x = self.features[14:](x_aux)
        return x_aux, x
        # return x

class _PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, norm_layer, trans_branch=None, attention=None):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        # -----------------------------------------------------#
        #   分区域进行平均池化
        #   30, 30, 320 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 = 30, 30, 640
        # -----------------------------------------------------#
        self.stages = nn.ModuleList(
            [self._make_stages(in_channels, out_channels, pool_size, norm_layer) for pool_size in pool_sizes])

        # 30, 30, 640 -> 30, 30, 80
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channels, kernel_size=3, padding=1,
                      bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

        self.Att = attention
        if self.Att:
           self.Attention = CBAM_block(in_channels // 4)

        self.trans = trans_branch


    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        if bin_sz == 1:
            conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, dilation=bin_sz, bias=False)
        else:
            conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=bin_sz, dilation=bin_sz, bias=False)
        # prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv1, conv2, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        for index in range(4):
            x = F.interpolate(self.stages[index](features), size=(h, w), mode='bilinear', align_corners=True)
            pyramids.extend([x])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        if self.Att:
            output = self.Attention(output)
        return output


class PSPNet(nn.Module):
    def __init__(self, num_classes, downsample_factor, backbone=1, pretrained=True, aux_branch=True,
                 trans_branch=0, attention=0):
        super(PSPNet, self).__init__()
        norm_layer = nn.BatchNorm2d

        #   1 resnet50; 2 MobileNetV2; 3 MobileNetV3_large; 4 MobileNetV3_small;
        #   5 EfficientNetV1; 6 EfficientNetV2;
        #   7 ShuffleNetV1; 8 ShuffleNetV2


        if backbone == 1:
            self.backbone = Resnet(downsample_factor, pretrained)
            out_channel = 2048
            aux_channel = 1024
        if backbone == 2:
            self.backbone = MobileNetV2(downsample_factor, pretrained)
            aux_channel = 96
            out_channel = 320

        # --------------------------------------------------------------#
        #	PSP模块，分区域进行池化
        #   分别分割成1x1的区域，2x2的区域，3x3的区域，6x6的区域
        #   30,30,320 -> 30,30,80 -> 30,30,21
        # --------------------------------------------------------------#
        # self.poolsize = [1, 2, 3, 5]
        self.poolsize = [1, 6, 12, 18]
        # self.pspmodule = _PSPModule(out_channel, pool_sizes=self.poolsize, norm_layer=norm_layer, \
        #                             trans_branch=trans_branch, attention=attention)
        #
        # self.initialize_weights(self.pspmodule)
        self.master_branch = nn.Sequential(
            _PSPModule(out_channel, pool_sizes=self.poolsize, norm_layer=norm_layer, trans_branch=trans_branch, attention=attention),
            nn.Conv2d(out_channel // 4, num_classes, kernel_size=1)
        )
        self.initialize_weights(self.master_branch)

        self.aux_branch = aux_branch

        if self.aux_branch:
            # ---------------------------------------------------#
            #	利用特征获得预测结果
            #   30, 30, 96 -> 30, 30, 40 -> 30, 30, 21
            # ---------------------------------------------------#
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(aux_channel, out_channel // 8, kernel_size=3, padding=1, bias=False),
                norm_layer(out_channel // 8),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_channel // 8, num_classes, kernel_size=1)
            )

        self.Att = attention
        if self.Att:
            self.Attention = SE_Block(out_channel)


    def forward(self, x):  # x为input
        input_size = (x.size()[2], x.size()[3])
        x_aux, x = self.backbone(x)
        # print("after backbone:\n", x.shape)
        if self.Att:
            x = self.Attention(x)
        output = self.master_branch(x)
        # output = self.pspmodule(x)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        # print(output.shape)
        if self.aux_branch:
            output_aux = self.auxiliary_branch(x_aux)
            output_aux = F.interpolate(output_aux, size=input_size, mode='bilinear', align_corners=True)
            return output_aux, output
        return output

    def initialize_weights(self, *models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.)
                    m.bias.data.fill_(1e-4)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.0001)
                    if m.bias is not None:
                        m.bias.data.zero_()


if __name__ == '__main__':
    x = torch.rand(8, 3, 512, 512)
    feature = torch.rand(8, 2048, 64, 64)
    cls_weights = np.ones([6], np.float32)
    weight = torch.from_numpy(cls_weights)
    png = torch.rand(8, 512, 512)
    # x = torch.rand(8,320,64,64)
    # self.poolsize = [1, 6, 12, 18]
    # conv1 = nn.Conv2d(320, 320, kernel_size=3, dilation=18, bias=False)
    # # prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
    # conv2 = nn.Conv2d(320, 80, kernel_size=1, bias=False)

    # y = conv2(conv1(x))
    # print(y.shape)
    pspnet = PSPNet(num_classes=6, downsample_factor=8,
                    backbone=2, pretrained=False, aux_branch=False,
                    trans_branch=0, attention=1)
    # print(pspnet)
    # output = pspnet(x)
    # print(output.shape)
    # loss= CE_Loss(output, png.long(), weight, num_classes=6)
    # print(loss)
    print(stat(pspnet, (3, 512, 512)))
