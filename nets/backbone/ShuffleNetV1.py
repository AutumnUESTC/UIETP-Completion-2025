import torch
import torch.nn as nn
import torchvision

# 分类数
num_class = 6


# DW卷积
def Conv3x3BNReLU(in_channels, out_channels, stride, groups):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                  groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


# 普通的1x1卷积
def Conv1x1BNReLU(in_channels, out_channels, groups):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


# PW卷积(不使用激活函数)
def Conv1x1BN(in_channels, out_channels, groups):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, groups=groups),
        nn.BatchNorm2d(out_channels)
    )


# channel重组操作
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    # 进行维度的变换操作
    def forward(self, x):
        # Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


# ShuffleNetV1的单元结构
class ShuffleNetUnits(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super(ShuffleNetUnits, self).__init__()
        self.stride = stride

        # print("in_channels:", in_channels, "out_channels:", out_channels)
        # 当stride=2时，为了不因为 in_channels+out_channels 不与 out_channels相等，需要先减，这样拼接的时候数值才会不变
        out_channels = out_channels - in_channels if self.stride > 1 else out_channels

        # 结构中的前一个1x1组卷积与3x3组件是维度的最后一次1x1组卷积的1/4，与ResNet类似
        mid_channels = out_channels // 4
        # print("out_channels:",out_channels,"mid_channels:",mid_channels)

        # ShuffleNet基本单元: 1x1组卷积 -> ChannelShuffle -> 3x3组卷积 -> 1x1组卷积
        self.bottleneck = nn.Sequential(
            # 1x1组卷积升维
            Conv1x1BNReLU(in_channels, mid_channels, groups),
            # channelshuffle实现channel重组
            ChannelShuffle(groups),
            # 3x3组卷积改变尺寸
            Conv3x3BNReLU(mid_channels, mid_channels, stride, groups),
            # 1x1组卷积降维
            Conv1x1BN(mid_channels, out_channels, groups)
        )

        # 当stride=2时，需要进行池化操作然后拼接起来
        if self.stride > 1:
            # hw减半
            self.shortcut = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.bottleneck(x)
        # 如果是stride=2，则将池化后的结果与通过基本单元的结果拼接在一起, 否则直接将输入与通过基本单元的结果相加
        out = torch.cat([self.shortcut(x), out], dim=1) if self.stride > 1 else (out + x)

        # 假设当前想要输出的channel为240，但此时stride=2，需要将输出与池化后的输入作拼接，此时的channel为24,24+240=264
        # torch.Size([1, 264, 28, 28])， 但是想输出的是240， 所以在这里 out_channels 要先减去 in_channels
        # torch.Size([1, 240, 28, 28]),  这是先减去的结果
        # if self.stride > 1:
        #     out = torch.cat([self.shortcut(x),out],dim=1)
        # 当stride为1时，直接相加即可
        # if self.stride == 1:
        #     out = out+x

        return self.relu(out)


class ShuffleNetV1(nn.Module):
    def __init__(self, planes, layers, groups, num_classes=num_class):
        super(ShuffleNetV1, self).__init__()

        # Conv1的输入channel只有24， 不算大，所以可以不用使用组卷积
        self.stage1 = nn.Sequential(
            Conv3x3BNReLU(in_channels=3, out_channels=24, stride=2, groups=1),  # torch.Size([1, 24, 112, 112])
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # torch.Size([1, 24, 56, 56])
        )

        # 以Group = 3为例 4/8/4层堆叠结构
        # 24 -> 240， groups=3  4层  is_stage2=True，stage2第一层不需要使用组卷积,其余全部使用组卷积
        self.stage2 = self._make_layer(24, planes[0], groups, layers[0], True)
        # 240 -> 480， groups=3  8层  is_stage2=False，全部使用组卷积，减少计算量
        self.stage3 = self._make_layer(planes[0], planes[1], groups, layers[1], False)
        # 480 -> 960， groups=3  4层  is_stage2=False，全部使用组卷积，减少计算量
        self.stage4 = self._make_layer(planes[1], planes[2], groups, layers[2], False)

        # in: torch.Size([1, 960, 7, 7])
        self.global_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        # group=3时最后channel为960，所以in_features=960
        self.linear = nn.Linear(in_features=planes[2], out_features=num_classes)

        # 权重初始化操作
        self.init_params()

    def _make_layer(self, in_channels, out_channels, groups, block_num, is_stage2):
        layers = []
        # torch.Size([1, 240, 28, 28])
        # torch.Size([1, 480, 14, 14])
        # torch.Size([1, 960, 7, 7])
        # 每个Stage的第一个基本单元stride均为2，其他单元的stride为1。且stage2的第一个基本单元不使用组卷积，因为参数量不大。
        layers.append(ShuffleNetUnits(in_channels=in_channels, out_channels=out_channels, stride=2,
                                      groups=1 if is_stage2 else groups))

        # 每个Stage的非第一个基本单元stride均为1，且全部使用组卷积，来减少参数计算量, 再叠加block_num-1层
        for idx in range(1, block_num):
            layers.append(ShuffleNetUnits(in_channels=out_channels, out_channels=out_channels, stride=1, groups=groups))
        return nn.Sequential(*layers)

    # 初始化权重
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)  # torch.Size([1, 24, 56, 56])
        x = self.stage2(x)  # torch.Size([1, 240, 28, 28])
        x = self.stage3(x)  # torch.Size([1, 480, 14, 14])
        x = self.stage4(x)  # torch.Size([1, 960, 7, 7])

        # x = self.global_pool(x)  # torch.Size([1, 960, 1, 1])
        # x = x.view(x.size(0), -1)  # torch.Size([1, 960])
        # x = self.dropout(x)
        # x = self.linear(x)  # torch.Size([1, 5])
        return x


# planes 是Stage2，Stage3，Stage4输出的参数
# layers 是Stage2，Stage3，Stage4的层数
# g1/2/3/4/8 指的是组卷积操作时的分组数

def shufflenet_g8(**kwargs):
    planes = [384, 768, 1536]
    layers = [4, 8, 4]
    model = ShuffleNetV1(planes, layers, groups=8)
    return model


def shufflenet_g4(**kwargs):
    planes = [272, 544, 1088]
    layers = [4, 8, 4]
    model = ShuffleNetV1(planes, layers, groups=4)
    return model


def shufflenet_g3(**kwargs):
    planes = [240, 480, 960]
    layers = [4, 8, 4]
    model = ShuffleNetV1(planes, layers, groups=3)
    return model


def shufflenet_g2(**kwargs):
    planes = [200, 400, 800]
    layers = [4, 8, 4]
    model = ShuffleNetV1(planes, layers, groups=2)
    return model


def shufflenet_g1(**kwargs):
    planes = [144, 288, 576]
    layers = [4, 8, 4]
    model = ShuffleNetV1(planes, layers, groups=1)
    return model


if __name__ == '__main__':
    model = shufflenet_g3()   # 常用
    # model = shufflenet_g8()
    # print(model)

    input = torch.randn(8, 3, 512, 512)
    out = model(input)
    print(out.shape)