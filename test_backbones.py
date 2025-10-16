import torch
import torch.nn as nn
from nets.backbone_manager import UnifiedPSPNet

def test_all_backbones():
    """测试所有主干网络"""
    input_tensor = torch.randn(2, 3, 512, 512)
    
    backbones = [0, 1, 2, 3]  # 所有支持的主干网络
    
    for backbone_type in backbones:
        print(f"\n=== 测试主干网络类型: {backbone_type} ===")
        
        try:
            # 创建模型
            model = UnifiedPSPNet(
                num_classes=6,
                backbone_type=backbone_type,
                pretrained=False  # 测试时不加载预训练权重
            )
            
            # 前向传播测试
            with torch.no_grad():
                output = model(input_tensor)
                print(f"输入尺寸: {input_tensor.shape}")
                print(f"输出尺寸: {output.shape}")
                print(f"主干网络: {model.backbone_manager.backbone_name}")
                print(f"参数数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
                print("测试成功!")
                
        except Exception as e:
            print(f"测试失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_all_backbones()
