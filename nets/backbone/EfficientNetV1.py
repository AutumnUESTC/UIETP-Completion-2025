import torch.nn as nn
from typing import Optional, Callable
from torch import Tensor
from torch.nn import functional as F
from collections import OrderedDict
from functools import partial
import math
import copy
import torch


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    将ch调整到最近的8的倍数
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNActivation(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int = 3,
                 stride: int = 1, groups: int = 1,  # 正常卷积还是DW卷积
                 norm_layer: Optional[Callable[..., nn.Module]] = None,  # BN
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish (torch>=1.7)
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_channels=in_planes,
                      out_channels=out_planes,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer())


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, expand_c: int, squeeze_factor: int = 4):
        """
        :params input_c: MBConv中的输入feature map的channel
        :params expand_c: MBConv中DW卷积的输出feature map的channel=第一个1x1卷积升维后的channel
        :squeeze_factor: 第一个全连接层降维因子
        """
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor  # 第一个全连接层的节点个数
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)  # 使用卷积代替全连接层 效果一样   降维
        self.ac1 = nn.SiLU()  # Swish
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)  # 升维
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        # 对每个channel进行全局平均池化 注意力机制 得到每个channel对应的权重
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        # 再通过不断学习来优化权重
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


def drop_path(x, drop_prob: float = 0., traing: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    This function is taken from the rwightman.
    It can be seen here:    DropBlock, DropPath
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not traing:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MBConvConfig:
    def __init__(self, kernel: int, in_planes: int, out_planes: int, expanded_ratio: int,
                 stride: int, use_se: bool, drop_rate: float, index: str, width_coefficient: float):
        """
        params: kernel: MBConv中的DW卷积的kernel_size(对应图片中的k)
        params: in_planes: MBConv模块的输入feature map的channel
        params: out_planes: MBConv模块的输出feature map的channel
        params: expanded_ratio: MBConv模块的第一个1x1卷积层的expand_rate  升维
        params: stride: DW卷积的stride
        params: use_se: 是否使用se模块  全部是True
        params: drop_rate: MBConv模块的Dropout层的随机失活比率
        params: index: 记录当前MBConv模块的名称  1a 2a 2b
        params: width_coefficient: 网络宽度方向上的倍率因子 论文中的w
        """
        self.in_planes = self.adjust_channels(in_planes, width_coefficient)
        self.kernel = kernel
        self.expanded_planes = self.in_planes * expanded_ratio  # MBConv模块的第一个1x1卷积层的输出channel
        self.out_planes = self.adjust_channels(out_planes, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        # 将channel*宽度倍率因子，再调整到8的整数倍
        return _make_divisible(channels * width_coefficient, 8)


class MBConv(nn.Module):
    def __init__(self, cnf: MBConvConfig, norm_layer: Callable[..., nn.Module]):
        """
        params: cnf: MBConv层配置文件
        params: norm_layer: BN结构
        """
        super(MBConv, self).__init__()
        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        # 只有再DW卷积的stride=1 且 输入channel=输出channel才能进行shortcut连接
        self.use_shortcut = (cnf.stride == 1 and cnf.in_planes == cnf.out_planes)

        layers = OrderedDict()  # 依次存储MBConv中的结构
        activation_layer = nn.SiLU

        # 第一个1x1卷积层 升维
        # 只有当expanded_ratio=1时，expanded_planes=in_planes，没有升维，所以不需要这个1x1卷积层
        if cnf.expanded_planes != cnf.in_planes:
            layers.update({"expand_conv": ConvBNActivation(cnf.in_planes,
                                                           cnf.expanded_planes,
                                                           kernel_size=1,
                                                           norm_layer=norm_layer,  # BN
                                                           activation_layer=activation_layer)})  # Swish
        # DW卷积   groups=channel
        layers.update({"dwconv": ConvBNActivation(cnf.expanded_planes,
                                                  cnf.expanded_planes,
                                                  kernel_size=cnf.kernel,
                                                  stride=cnf.stride,
                                                  groups=cnf.expanded_planes,
                                                  norm_layer=norm_layer,  # BN
                                                  activation_layer=activation_layer)})  # Swish
        # SE模块
        if cnf.use_se:
            layers.update({"se": SqueezeExcitation(cnf.in_planes,
                                                   cnf.expanded_planes)})
        # 最后1x1卷积层
        layers.update({"project_conv": ConvBNActivation(cnf.expanded_planes,
                                                        cnf.out_planes,
                                                        kernel_size=1,
                                                        norm_layer=norm_layer,  # BN
                                                        activation_layer=nn.Identity)})  # Identity
        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_planes
        self.is_strided = cnf.stride > 1  # 似乎没什么用

        # 只有在使用shortcut连接时才使用dropout层
        if cnf.drop_rate > 0 and self.use_shortcut:
            # self.dropout = nn.Dropout2d(p=cnf.drop_rate, inplace=True)
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        result = self.dropout(result)
        if self.use_shortcut:
            result += x
        return result


class EfficientNetV1(nn.Module):
    def __init__(self, width_coefficient: float, depth_coefficient: float, num_classes: int = 1000,
                 dropout_rate: float = 0.2, drop_connect_rate: float = 0.2,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        """
        params: width_coefficient: 网络宽度上的倍率因子 对应论文中的 w
        params: depth_coefficient: 网络深度上的倍率因子 对应论文中的 d
        params: num_classes: 分类的类别个数
        params: dropout_rate: stage9的FC层前面的Dropout的随即失活比率
        params: drop_connect_rate: MBConv模块的Dropout层的随机失活比率 从0慢慢增长到0.2
        params: block: MBConv模块
        params: norm_layer: 普通的BN结构
        """
        super(EfficientNetV1, self).__init__()

        # 默认的B0网络配置文件 后面B1-B7都是在这个基础上乘以相应的深度、宽度、分辨率倍率因子
        # stage2 - stage8
        # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
        # kernel_size: MBConv后面写的knxn
        # in_channel/out_channel: 当前stage的第一个MBConv的输入/输出feature map的channel
        # exp_ratio: 第一个1x1卷积的膨胀率 对应当前MBConvn
        # strides: 当前stage的第一个
        # use_SE: 默认每个stage都使用SE模块
        # drop_connect_rate: MBConv模块的Dropout层的随机失活比率 选默认都是0.2 后面再调整
        # repeats: MBConv在当前stage中重复的次数
        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        def round_repeats(repeats):
            # depth_coefficient代表depth维度上的倍率因子（仅针对Stage2到Stage8）
            # 通过这个函数用depth_coefficient倍率因子动态的调整网络的深度（MBConv的重复次数）
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:
            block = MBConv

        if norm_layer is None:
            # patial方法搭建层结构，下次使用就不需要再传eps和momentum这两个参数了 会默认传入这两个值
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        # 通过这个函数用width_coefficient倍率因子动态的调整网络的宽度（channel）
        # 具体做法: 将channel*宽度倍率因子，再调整到8的整数倍
        adjust_channels = partial(MBConvConfig.adjust_channels, width_coefficient=width_coefficient)

        # 初始化单个MB_config
        MB_config = partial(MBConvConfig, width_coefficient=width_coefficient)

        # 得到stage2-stage8所有MB模块的配置信息
        b = 0  # 用于调整drop_connect_rate
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))  # 统计所以MB模块的重复次数
        MBConv_configs = []  # 存放所以MB模块的配置文件
        for stage, args in enumerate(default_cnf):  # 遍历每个stage
            cnf = copy.copy(args)
            for i in range(round_repeats(cnf.pop(-1))):  # 遍历每个stage中的MB模块
                if i > 0:
                    cnf[-3] = 1  # 当i>0时,stride=1
                    cnf[1] = cnf[2]  # 当i>0时,输入channel=输出channel=第一个MB模块的输出channel

                # cnf[-1] *= b / num_blocks  # update drop_connect_rate
                cnf[-1] = args[-2] * b / num_blocks
                index = str(stage + 1) + chr(i + 97)  # 记录当前MB是属于第几个stage中的第几个MB结构
                MBConv_configs.append(MB_config(*cnf, index))
                b += 1

        # 开始搭建整体网络结构
        layers = OrderedDict()

        # stage1
        layers.update({"stem_conv": ConvBNActivation(in_planes=3,
                                                     out_planes=adjust_channels(32),  # 通过width倍率因子调整
                                                     kernel_size=3,
                                                     stride=2,
                                                     norm_layer=norm_layer)})

        # stage2-stage8
        for cnf in MBConv_configs:
            layers.update({cnf.index: block(cnf, norm_layer)})

        # stage9
        last_conv_input_c = MBConv_configs[-1].out_planes
        last_conv_output_c = adjust_channels(1280)  # 通过width倍率因子调整
        layers.update({"top": ConvBNActivation(in_planes=last_conv_input_c,
                                               out_planes=last_conv_output_c,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(512 * 4, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # x = self.classifier(x)
        return x


def efficientnet_b0(num_classes=1000):
    # input image size 224x224
    return EfficientNetV1(width_coefficient=1.0,
                        depth_coefficient=1.0,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b1(num_classes=1000):
    # input image size 240x240
    return EfficientNetV1(width_coefficient=1.0,
                        depth_coefficient=1.1,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b2(num_classes=1000):
    # input image size 260x260
    return EfficientNetV1(width_coefficient=1.1,
                        depth_coefficient=1.2,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b3(num_classes=1000):
    # input image size 300x300
    return EfficientNetV1(width_coefficient=1.2,
                        depth_coefficient=1.4,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b4(num_classes=1000):
    # input image size 380x380
    return EfficientNetV1(width_coefficient=1.4,
                        depth_coefficient=1.8,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b5(num_classes=1000):
    # input image size 456x456
    return EfficientNetV1(width_coefficient=1.6,
                        depth_coefficient=2.2,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b6(num_classes=1000):
    # input image size 528x528
    return EfficientNetV1(width_coefficient=1.8,
                        depth_coefficient=2.6,
                        dropout_rate=0.5,
                        num_classes=num_classes)


def efficientnet_b7(num_classes=1000):
    # input image size 600x600
    return EfficientNetV1(width_coefficient=2.0,
                        depth_coefficient=3.1,
                        dropout_rate=0.5,
                        num_classes=num_classes)

if __name__ == '__main__':
    net = efficientnet_b6(num_classes=6)
    x =torch.randn(8,3,512,512)
    out = net(x)
    print(out.shape)