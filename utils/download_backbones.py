import torch
import os
from torchvision.models import (
    resnet50, resnet101, mobilenet_v3_large, efficientnet_b0,
    ResNet50_Weights, ResNet101_Weights, MobileNet_V3_Large_Weights, EfficientNet_B0_Weights
)


class BackboneWeightDownloader:
    """主干网络权重下载器"""
    
    def __init__(self, save_dir="model_data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def download_all(self):
        """下载所有主干网络权重"""
        print("开始下载主干网络预训练权重...")
        
        # ResNet50
        print("下载ResNet50...")
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        torch.save(model.state_dict(), os.path.join(self.save_dir, "resnet50.pth"))
        
        # ResNet101  
        print("下载ResNet101...")
        model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        torch.save(model.state_dict(), os.path.join(self.save_dir, "resnet101.pth"))
        
        # MobileNetV3 Large
        print("下载MobileNetV3 Large...")
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        torch.save(model.state_dict(), os.path.join(self.save_dir, "mobilenet_v3_large.pth"))
        
        # EfficientNet-B0
        print("下载EfficientNet-B0...")
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        torch.save(model.state_dict(), os.path.join(self.save_dir, "efficientnet_b0.pth"))
        
        # 高级ResNet50（使用相同的base权重）
        print("创建高级ResNet50权重...")
        torch.save(model.state_dict(), os.path.join(self.save_dir, "resnet50_advanced.pth"))
        
        print("所有权重下载完成！")


if __name__ == "__main__":
    downloader = BackboneWeightDownloader()
    downloader.download_all()