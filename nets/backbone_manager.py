import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from nets.pspnet import PSPNet

# 导入自定义主干网络
from nets.backbone.MobileNetV3 import mobilenet_v3_large
from nets.backbone.resnet import resnet50, resnet101

# 特征融合模块
class FeatureFusionModule(nn.Module):
    """特征融合模块，用于融合CNN特征和手工特征"""
    
    def __init__(self, cnn_channels, handcrafted_channels, output_channels):
        super(FeatureFusionModule, self).__init__()
        self.fusion_weight = 1.0
        
        # 调整手工特征的通道数
        self.handcrafted_conv = nn.Sequential(
            nn.Conv2d(handcrafted_channels, cnn_channels // 4, kernel_size=1),
            nn.BatchNorm2d(cnn_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(cnn_channels + cnn_channels // 4, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, cnn_features, handcrafted_features):
        # 调整手工特征
        handcrafted_adjusted = self.handcrafted_conv(handcrafted_features)
        
        # 调整手工特征的空间尺寸以匹配CNN特征
        if handcrafted_adjusted.size()[-2:] != cnn_features.size()[-2:]:
            handcrafted_adjusted = F.interpolate(
                handcrafted_adjusted, 
                size=cnn_features.size()[-2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        # 特征融合
        fused_features = torch.cat([cnn_features, handcrafted_adjusted], dim=1)
        fused_features = self.fusion_conv(fused_features)
        
        # 应用融合权重
        if self.fusion_weight < 1.0:
            fused_features = self.fusion_weight * fused_features + (1 - self.fusion_weight) * cnn_features
        
        return fused_features
    
    def set_fusion_weight(self, weight):
        """动态调整融合权重"""
        self.fusion_weight = weight

class BackboneManager(nn.Module):
    def __init__(self, backbone_type, pretrained=False, downsample_factor=16):
        super(BackboneManager, self).__init__()
        self.backbone_type = backbone_type
        self.pretrained = pretrained
        self.downsample_factor = downsample_factor
        
        # 创建主干网络
        self.backbone = self._create_backbone()
        self.backbone_name = self._get_backbone_name()
        
        # 获取主干网络的输出通道数
        self.output_channels = self._get_output_channels()

    def _get_backbone_name(self):
        names = {
            0: "MobileNetV3",
            1: "ResNet50", 
            2: "ResNet101",
            3: "EfficientNet"
        }
        return names.get(self.backbone_type, "Unknown")

    def _create_backbone(self):
        if self.backbone_type == 0:
            return self._create_mobilenet()
        elif self.backbone_type == 1:
            return self._create_resnet50()
        elif self.backbone_type == 2:
            return self._create_resnet101()
        elif self.backbone_type == 3:
            return self._create_efficientnet()
        else:
            raise ValueError(f"Unsupported backbone type: {self.backbone_type}")

    def _create_mobilenet(self):
        """创建MobileNetV3主干网络"""
        print("使用自定义MobileNetV3主干网络")
        model = mobilenet_v3_large(pretrained=self.pretrained)
        
        # 移除分类器，只保留特征提取部分
        class MobileNetV3Features(nn.Module):
            def __init__(self, original_model):
                super(MobileNetV3Features, self).__init__()
                self.features = original_model.features
                
            def forward(self, x):
                return self.features(x)
        
        return MobileNetV3Features(model)

    def _create_resnet50(self):
        """创建ResNet50主干网络"""
        print("使用自定义ResNet50主干网络")
        model = resnet50(pretrained=self.pretrained)
        
        # 移除最后的全连接层和平均池化层
        class ResNet50Features(nn.Module):
            def __init__(self, original_model):
                super(ResNet50Features, self).__init__()
                self.conv1 = original_model.conv1
                self.bn1 = original_model.bn1
                self.relu = original_model.relu
                self.maxpool = original_model.maxpool
                self.layer1 = original_model.layer1
                self.layer2 = original_model.layer2
                self.layer3 = original_model.layer3
                self.layer4 = original_model.layer4
                
            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                
                return x
                
        return ResNet50Features(model)

    def _create_resnet101(self):
        """创建ResNet101主干网络"""
        print("使用自定义ResNet101主干网络")
        model = resnet101(pretrained=self.pretrained)
        
        # 移除最后的全连接层和平均池化层
        class ResNet101Features(nn.Module):
            def __init__(self, original_model):
                super(ResNet101Features, self).__init__()
                self.conv1 = original_model.conv1
                self.bn1 = original_model.bn1
                self.relu = original_model.relu
                self.maxpool = original_model.maxpool
                self.layer1 = original_model.layer1
                self.layer2 = original_model.layer2
                self.layer3 = original_model.layer3
                self.layer4 = original_model.layer4
                
            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                
                return x
                
        return ResNet101Features(model)

    def _create_efficientnet(self):
        """创建EfficientNet主干网络"""
        print("使用自定义EfficientNet主干网络")
        # 使用torchvision的EfficientNet
        model = models.efficientnet_b0(pretrained=self.pretrained)
        
        # 移除分类器，只保留特征提取部分
        class EfficientNetFeatures(nn.Module):
            def __init__(self, original_model):
                super(EfficientNetFeatures, self).__init__()
                self.features = original_model.features
                
            def forward(self, x):
                return self.features(x)
        
        return EfficientNetFeatures(model)

    def _get_output_channels(self):
        """获取主干网络的输出通道数"""
        if self.backbone_type == 0:  # MobileNetV3
            return 960
        elif self.backbone_type in [1, 2]:  # ResNet50/101
            return 2048
        elif self.backbone_type == 3:  # EfficientNet
            return 1280
        else:
            return 2048  # 默认值

    def forward(self, x):
        return self.backbone(x)

    def load_pretrained(self, weight_path):
        """加载预训练权重"""
        try:
            if not os.path.exists(weight_path):
                print(f"警告: 预训练权重文件不存在: {weight_path}")
                return False
            
            # 加载权重
            checkpoint = torch.load(weight_path, map_location='cpu')
            
            # 处理不同的权重文件格式
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # 过滤不匹配的键
            model_state_dict = self.backbone.state_dict()
            filtered_state_dict = {}
            
            for k, v in state_dict.items():
                # 移除可能的模块前缀
                if k.startswith('module.'):
                    k = k[7:]
                
                # 检查键是否匹配
                if k in model_state_dict and v.shape == model_state_dict[k].shape:
                    filtered_state_dict[k] = v
                else:
                    # 尝试匹配不同的键命名约定
                    matched = False
                    for model_key in model_state_dict.keys():
                        if k.endswith(model_key) or model_key.endswith(k):
                            if v.shape == model_state_dict[model_key].shape:
                                filtered_state_dict[model_key] = v
                                matched = True
                                break
                    
                    if not matched:
                        print(f"跳过不匹配的权重: {k} (形状: {v.shape})")
            
            # 加载过滤后的权重
            if filtered_state_dict:
                self.backbone.load_state_dict(filtered_state_dict, strict=False)
                print(f"成功加载预训练权重: {weight_path}")
                print(f"加载了 {len(filtered_state_dict)}/{len(model_state_dict)} 个参数")
                return True
            else:
                print(f"警告: 没有找到匹配的权重参数")
                return False
                
        except Exception as e:
            print(f"加载预训练权重时出错: {e}")
            return False


class UnifiedPSPNet(nn.Module):
    def __init__(self, num_classes, backbone_type, pretrained=False, downsample_factor=16, use_feature_engineering=False):
        super(UnifiedPSPNet, self).__init__()
        
        self.use_feature_engineering = use_feature_engineering
        
        self.backbone_manager = BackboneManager(backbone_type, pretrained, downsample_factor)
        self.backbone_channels = self.backbone_manager.output_channels
        
        # 特征融合模块（如果启用特征工程）
        if self.use_feature_engineering:
            self.feature_fusion = FeatureFusionModule(
                self.backbone_channels, 16, self.backbone_channels  # 手工特征有18个通道
            )
        
        # PSP模块配置
        self.psp_pool_sizes = [1, 2, 3, 6]
        self.psp_output_channels = self.backbone_channels + len(self.psp_pool_sizes) * 512
        
        # 创建PSP模块
        self.psp = PSPModule(self.backbone_channels, pool_sizes=self.psp_pool_sizes)
        
        # 最终卷积层
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.psp_output_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, x, handcrafted_features=None):
        # 主干网络特征提取
        features = self.backbone_manager(x)
        
        # 特征融合（如果启用且提供了手工特征）
        if self.use_feature_engineering and handcrafted_features is not None:
            features = self.feature_fusion(features, handcrafted_features)
        
        # PSP模块处理
        psp_output = self.psp(features)
        
        # 最终卷积
        output = self.final_conv(psp_output)
        
        # 上采样到输入尺寸
        output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return output

    def set_fusion_weight(self, weight):
        """设置特征融合权重"""
        if hasattr(self, 'feature_fusion'):
            self.feature_fusion.set_fusion_weight(weight)


class PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super(PSPModule, self).__init__()
        self.pool_sizes = pool_sizes
        self.pool_layers = nn.ModuleList()
        
        for size in pool_sizes:
            self.pool_layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        input_size = x.size()[2:]
        
        # 主分支
        psp_outputs = [x]
        
        # 金字塔池化分支
        for pool_layer in self.pool_layers:
            pooled = pool_layer(x)
            upsampled = F.interpolate(pooled, size=input_size, mode='bilinear', align_corners=True)
            psp_outputs.append(upsampled)
        
        # 拼接所有特征
        return torch.cat(psp_outputs, dim=1)