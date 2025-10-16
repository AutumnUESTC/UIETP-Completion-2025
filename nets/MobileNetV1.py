import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# 定义DSC结构：DW+PW操作
def BottleneckV1(in_channels, out_channels, stride):
    # 深度可分卷积操作模块: DSC卷积 = DW卷积 + PW卷积
    return nn.Sequential(
        # dw卷积,也是RexNeXt中的组卷积，当分组个数等于输入通道数时，输出矩阵的通道输也变成了输入通道数时，组卷积就是dw卷积
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1,
                  groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(inplace=True),
        # pw卷积，与普通的卷积一样，只是使用了1x1的卷积核
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


# 定义MobileNetV1结构
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=5):
        super(MobileNetV1, self).__init__()

        # torch.Size([1, 3, 224, 224])
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        # torch.Size([1, 32, 112, 112])

        # 叠加的基本结构是： DW+PW(DW用来减小尺寸stride=2实现,PW用来增加通道out_channels增加实现)
        self.bottleneck = nn.Sequential(
            BottleneckV1(32, 64, stride=1),  # torch.Size([1, 64, 112, 112]), stride=1
            BottleneckV1(64, 128, stride=2),  # torch.Size([1, 128, 56, 56]), stride=2
            BottleneckV1(128, 128, stride=1),  # torch.Size([1, 128, 56, 56]), stride=1
            BottleneckV1(128, 256, stride=2),  # torch.Size([1, 256, 28, 28]), stride=2
            BottleneckV1(256, 256, stride=1),  # torch.Size([1, 256, 28, 28]), stride=1
            BottleneckV1(256, 512, stride=2),  # torch.Size([1, 512, 14, 14]), stride=2
            BottleneckV1(512, 512, stride=1),  # torch.Size([1, 512, 14, 14]), stride=1
            BottleneckV1(512, 512, stride=1),  # torch.Size([1, 512, 14, 14]), stride=1
            BottleneckV1(512, 512, stride=1),  # torch.Size([1, 512, 14, 14]), stride=1
            BottleneckV1(512, 512, stride=1),  # torch.Size([1, 512, 14, 14]), stride=1
            BottleneckV1(512, 512, stride=1),  # torch.Size([1, 512, 14, 14]), stride=1
            BottleneckV1(512, 1024, stride=2),  # torch.Size([1, 1024, 7, 7]), stride=2
            BottleneckV1(1024, 1024, stride=1),  # torch.Size([1, 1024, 7, 7]), stride=1
        )

        # torch.Size([1, 1024, 7, 7])
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)  # torch.Size([1, 1024, 1, 1])
        self.linear = nn.Linear(in_features=1024, out_features=num_classes)
        self.dropout = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(dim=1)

        self.init_params()

    # 初始化操作
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)  # torch.Size([1, 32, 112, 112])
        x = self.bottleneck(x)  # torch.Size([1, 1024, 7, 7])
        x = self.avg_pool(x)  # torch.Size([1, 1024, 1, 1])
        x = x.view(x.size(0), -1)  # torch.Size([1, 1024])
        x = self.dropout(x)
        x = self.linear(x)  # torch.Size([1, 5])
        out = self.softmax(x)  # 概率化
        return x


# if __name__ == '__main__':
#     model = MobileNetV1()
#     # print(model)
#
#     input = torch.randn(1, 3, 224, 224)
#     out = model(input)
#     print(out.shape)
if __name__ == '__main__':
    net = MobileNetV1().cuda()
    summary(net, (3, 224, 224))