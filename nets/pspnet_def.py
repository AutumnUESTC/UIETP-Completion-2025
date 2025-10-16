from copy import deepcopy

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchstat import stat

from nets.Attention import SE_Block, ECA_block, CBAM_block, GAM_Attention, CoordAtt, SpatialAttention, ChannelAttention
from nets.ViT import Transformer
from nets.backbone.EfficientNetV1 import efficientnet_b6
from nets.backbone.EfficientNetV2 import efficientnetv2_m
from nets.backbone.MobileNetV3 import mobilenet_v3_large, mobilenet_v3_small
from nets.backbone.ShuffleNetV1 import shufflenet_g8
from nets.backbone.ShuffleNetV2 import shufflenet_v2_x2_0
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
        # return x_aux, x
        return x

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
        # return x_aux, x
        return x

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

        self.trans = trans_branch
        if self.trans:
            if self.trans == 1:
                self.trans_branch = Transformer(in_channels)
                self.bottleneck = nn.Sequential(
                    nn.Conv2d(in_channels + (out_channels * (len(pool_sizes) + 1)), out_channels, kernel_size=3,
                              padding=1,
                              bias=False),
                    norm_layer(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.1)
                )
            if self.trans == 2:
                self.trans_branch = Transformer(in_channels // 4)
                self.bottleneck = nn.Sequential(
                    nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channels, kernel_size=3,
                              padding=1,
                              bias=False),
                    norm_layer(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(0.1)
                )
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channels, kernel_size=3, padding=1,
                          bias=False),
                norm_layer(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1)
            )

        self.Att = attention
        if self.Att:
            if self.Att == 1:
                self.Attention = SE_Block(in_channels // 4)
            if self.Att == 2:
                self.Attention = ECA_block(in_channels // 4)
            if self.Att == 3:
                self.Attention = CBAM_block(in_channels // 4)
            if self.Att == 4:
                self.Attention = GAM_Attention(in_channels // 4, in_channels // 4)
            if self.Att == 5:
                self.Attention = CoordAtt(in_channels // 4, in_channels // 4)

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        for index in range(4):
            x = F.interpolate(self.stages[index](features), size=(h, w), mode='bilinear', align_corners=True)
            if self.trans == 2:
                x = self.trans_branch(x)
            pyramids.extend([x])
        if self.trans == 1:
            transmap = F.interpolate(self.trans_branch(features), size=(h, w), mode='bilinear', align_corners=True)
            pyramids.extend([transmap])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        if self.Att:
            output = self.Attention(output)
        return output


class PSPNet(nn.Module):
    def __init__(self, num_classes, downsample_factor, backbone, pretrained=True, aux_branch=True,
                 trans_branch=0, attention=0):
        super(PSPNet, self).__init__()
        norm_layer = nn.BatchNorm2d

        #   1 resnet50; 2 MobileNetV2; 3 MobileNetV3_large; 4 MobileNetV3_small;
        #   5 EfficientNetV1; 6 EfficientNetV2;
        #   7 ShuffleNetV1; 8 ShuffleNetV2

        if backbone == 1:
            self.backbone = Resnet(downsample_factor, pretrained)
            out_channel = 2048
        elif backbone == 2:
            self.backbone = MobileNetV2(downsample_factor, pretrained)
            aux_channel = 96
            out_channel = 320
        elif backbone == 3:
            self.backbone = mobilenet_v3_large(num_classes = num_classes)
            # aux_channel =
            out_channel = 960
        elif backbone == 4:
            self.backbone = mobilenet_v3_small(num_classes=num_classes)
            # aux_channel =
            out_channel = 576
        elif backbone == 5:
            # input image size 528x528
            self.backbone = efficientnet_b6(num_classes=num_classes)
            # aux_channel =
            out_channel = 2304
        elif backbone == 6:
            # train_size: 384, eval_size: 480
            self.backbone = efficientnetv2_m(num_classes=num_classes)
            # aux_channel =
            out_channel = 512
        elif backbone == 7:
            # Stage2，Stage3，Stage4输出的参数 [384, 768, 1536]
            self.backbone = shufflenet_g8()
            # aux_channel = 768
            out_channel = 1536
        elif backbone == 8:
            self.backbone = shufflenet_v2_x2_0(num_classes=num_classes)
            # aux_channel = 488
            out_channel = 2048


        # --------------------------------------------------------------#
        #	PSP模块，分区域进行池化
        #   分别分割成1x1的区域，2x2的区域，3x3的区域，6x6的区域
        #   30,30,320 -> 30,30,80 -> 30,30,21
        # --------------------------------------------------------------#
        self.poolsize = [1, 2, 3, 5]
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

        self.attention = attention
        if self.attention:
            self.Attention = SE_Block(out_channel)
            # if self.Att == 1:
            #     self.Attention = SE_Block(out_channel)
            # if self.Att == 2:
            #     self.Attention = ECA_block(out_channel)
            # if self.Att == 3:
            #     self.Attention = CBAM_block(out_channel)
            # if self.Att == 4:
            #     self.Attention = GAM_Attention(out_channel)
            # if self.Att == 5:
            #     self.Attention = CoordAtt(out_channel)


    def forward(self, x):  # x为input
        input_size = (x.size()[2], x.size()[3])
        x = self.backbone(x)  # cuda:0
        # print("after backbone:\n", x.shape)
        if self.attention:
            x = self.Attention(x)
        output = self.master_branch(x)
        # output = self.pspmodule(x, trans_branch = trans_branch)
        # output = self.unsample(output)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)

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
    # x = torch.rand(8, 3, 512, 512)
    # feature = torch.rand(8, 2048, 64, 64)
    # cls_weights = np.ones([6], np.float32)
    # weight = torch.from_numpy(cls_weights)
    # png = torch.rand(8, 512, 512)


      # trans_branch : 0 不使用ViT; 1——并行; 2——串行
      # attention : 0 不使用; 1——SeNet； 2——； 3——CBAM
    pspnet = PSPNet(num_classes=6, downsample_factor=8,
                    backbone=2, pretrained=False, aux_branch=False,
                    trans_branch=0, attention=3)
    # print(summary(pspnet,(3,512,512),batch_size=8))
    print(stat(pspnet,(3,512,512)))
    # output = pspnet(x)
    # loss= CE_Loss(output, png.long(), weight, num_classes=6)
    # loss_f = loss
    # print("output shape:\n", output.shape)
    # print(loss_f)
