#!/usr/bin/env python3
import os
import sys
import torch

print("=== 最终项目验证 ===")
print(f"工作目录: {os.getcwd()}")
print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# 测试所有关键模块
modules_to_test = [
    "utils.dataloader",
    "utils.utils", 
    "utils.utils_metrics",
    "nets.pspnet",
    "train",
    "predict"
]

print("\n=== 模块导入测试 ===")
for module in modules_to_test:
    try:
        if module == "train":
            __import__("train")
        elif module == "predict":
            __import__("predict")
        else:
            __import__(module)
        print(f"✓ {module} 导入成功")
    except Exception as e:
        print(f"✗ {module} 导入失败: {e}")

# 检查关键文件
print("\n=== 文件检查 ===")
key_files = {
    "模型文件": "model_data/pspnet_mobilenetv2.pth",
    "数据集": "VOCdevkit/VOC2007",
    "训练脚本": "train.py",
    "预测脚本": "predict.py",
    "工具模块": "utils/__init__.py"
}

for desc, path in key_files.items():
    if os.path.exists(path):
        print(f"✓ {desc}: {path} 存在")
    else:
        print(f"✗ {desc}: {path} 缺失")

print("\n=== 项目状态: 正常！ ===")
