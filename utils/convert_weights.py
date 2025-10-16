# utils/convert_weights.py
import torch
import os

def convert_pspnet_weights():
    """转换PSPNet权重格式以匹配新结构"""
    weight_path = "/home/wuyou/pspnet-pytorch/model_data/pspnet_resnet50.pth"
    
    if not os.path.exists(weight_path):
        print(f"权重文件不存在: {weight_path}")
        return
    
    try:
        # 加载原始权重
        original_weights = torch.load(weight_path, map_location='cpu')
        print(f"原始权重键数量: {len(original_weights)}")
        
        # 创建转换后的权重字典
        converted_weights = {}
        
        for key, value in original_weights.items():
            # 转换键名格式
            if key.startswith('backbone.'):
                # 直接保留backbone权重
                converted_weights[key] = value
            elif key.startswith('pspnet.'):
                # 移除pspnet前缀
                new_key = key.replace('pspnet.', '')
                converted_weights[new_key] = value
            else:
                # 其他权重保持不变
                converted_weights[key] = value
        
        # 保存转换后的权重
        converted_path = weight_path.replace('.pth', '_converted.pth')
        torch.save(converted_weights, converted_path)
        print(f"转换后的权重已保存: {converted_path}")
        print(f"转换后权重键数量: {len(converted_weights)}")
        
        return converted_path
        
    except Exception as e:
        print(f"权重转换失败: {e}")
        return None

if __name__ == "__main__":
    convert_pspnet_weights()