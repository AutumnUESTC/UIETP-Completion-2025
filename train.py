import os
import datetime
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import cv2
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import json
import shutil
import math

# 导入自定义模块
from nets.pspnet_CBAMi import PSPNet
from nets.pspnet_training import (get_lr_scheduler, set_optimizer_lr, weights_init, 
                                  CE_Loss, Dice_loss, Focal_Loss)
from nets.contrastive import ContrastiveLearner
from nets.dynamic_loss import DynamicWeightedLoss
from utils.EMA import EMA
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import PSPnetDataset, pspnet_dataset_collate, UnlabeledDataset, contrastive_transform
from utils.utils import download_weights, show_config
from utils.dataloader import preprocess_input

# 导入主干网络管理器
from nets.backbone_manager import BackboneManager, UnifiedPSPNet

# 导入对比学习分析工具
from utils.contrastive_analysis import ContrastiveAnalysisFixed


# ==================== 实验配置管理器 ====================
class ExperimentManager:
    """实验管理器，负责管理多组实验的配置和保存"""
    
    def __init__(self, base_log_dir="/home/wuyou/pspnet-pytorch/logs"):
        self.base_log_dir = base_log_dir
        self.experiment_id = self._generate_experiment_id()
        self.current_experiment_dir = None
        
    def _generate_experiment_id(self):
        """生成唯一的实验ID"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"exp_{timestamp}"
    
    def create_experiment(self, config_dict, experiment_name=None):
        """创建新的实验目录并保存配置"""
        if experiment_name:
            exp_dir_name = f"{self.experiment_id}_{experiment_name}"
        else:
            exp_dir_name = self.experiment_id
            
        self.current_experiment_dir = os.path.join(self.base_log_dir, exp_dir_name)
        
        # 创建实验目录
        os.makedirs(self.current_experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.current_experiment_dir, "weights"), exist_ok=True)
        os.makedirs(os.path.join(self.current_experiment_dir, "logs"), exist_ok=True)
        
        # 保存实验配置
        config_file = os.path.join(self.current_experiment_dir, "experiment_config.json")
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)
        
        # 保存一份当前代码的备份
        self._backup_current_code()
        
        print(f"实验目录已创建: {self.current_experiment_dir}")
        return self.current_experiment_dir
    
    def _backup_current_code(self):
        """备份当前代码到实验目录"""
        code_backup_dir = os.path.join(self.current_experiment_dir, "code_backup")
        os.makedirs(code_backup_dir, exist_ok=True)
        
        # 复制重要的代码文件
        files_to_backup = [
            "train.py",
            "nets/pspnet.py",
            "nets/backbone_manager.py",
            "nets/contrastive.py",
            "nets/dynamic_loss.py",
            "utils/callbacks.py",
            "utils/dataloader.py",
            "utils/contrastive_analysis.py"
        ]
        
        for file_path in files_to_backup:
            if os.path.exists(file_path):
                dest_path = os.path.join(code_backup_dir, file_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(file_path, dest_path)
    
    def get_experiment_log_dir(self):
        """获取当前实验的日志目录"""
        return os.path.join(self.current_experiment_dir, "logs")
    
    def get_experiment_weight_dir(self):
        """获取当前实验的权重保存目录"""
        return os.path.join(self.current_experiment_dir, "weights")


# ==================== 配置类 ====================
class TrainingConfig:
    """训练配置类"""
    def __init__(self):
        # 基础配置
        self.cuda = True
        self.distributed = False
        self.sync_bn = False
        self.fp16 = False
        self.local_rank = 0
        
        # 模型配置
        self.num_classes = 6
        self.backbone_type = 1
        self.backbone_pretrained = True
        self.downsample_factor = 8
        self.input_shape = [512, 512]
        
        # 预训练权重配置
        self.model_path = ""
        self.backbone_weights = {
            0: "/home/wuyou/pspnet-pytorch/model_data/mobilenet_v3_large.pth",
            1: "/home/wuyou/pspnet-pytorch/model_data/pspnet_resnet50.pth", 
            2: "/home/wuyou/pspnet-pytorch/model_data/pspnet_resnet50.pth",
            3: "/home/wuyou/pspnet-pytorch/model_data/efficientnet_b6.pth",
            4: "/home/wuyou/pspnet-pytorch/model_data/pspnet_resnet50.pth",
            5: "/home/wuyou/pspnet-pytorch/model_data/mobilenet_v3_small.pth",
            6: "/home/wuyou/pspnet-pytorch/model_data/pspnet_mobilenetv2.pth",
            7: "/home/wuyou/pspnet-pytorch/model_data/shufflenet_v1_g3.ckpt",
            8: "/home/wuyou/pspnet-pytorch/model_data/shufflenetv2_x2_0.pth",
            9: "/home/wuyou/pspnet-pytorch/model_data/efficientnet_v2.pth"
        }
        
        # 训练阶段配置
        self.init_epoch = 0
        self.freeze_epoch = 200
        self.freeze_batch_size = 8
        self.unfreeze_epoch = 1000
        self.unfreeze_batch_size = 32
        
        # 梯度累积配置
        self.accumulation_steps = 1
        
        # 训练策略配置
        self.freeze_train = False
        self.ema_train = False
        self.hl_train = False
        self.hl_c = 0
        self.hl_a = 0.1
        self.trans_branch = 0
        self.attention = 1
        
        # 实验变量配置
        self.use_feature_engineering = False
        self.use_contrastive_pretrain = True  # 对比学习预训练开关
        
        # 损失函数配置 - 简化为传统开关
        self.dice_loss = True      # 开启Dice Loss
        self.focal_loss = True     # 开启Focal Loss
        
        # 改进版参数（仅当dice_loss和focal_loss都为True时生效）
        self.improved_alpha = 0.3  # Dice vs Focal平衡参数
        self.improved_gamma = 2.0  # Focal Loss聚焦参数
        
        self.cls_weights = np.ones([self.num_classes], np.float32)
        self.aux_branch = False
        
        # 优化器配置
        self.init_lr = 1e-4
        self.min_lr = self.init_lr * 0.01
        self.optimizer_type = "adamw"
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.lr_decay_type = 'cos'
        
        # 学习率预热配置
        self.use_warmup = True
        self.warmup_epochs = 10
        self.warmup_lr = 1e-6
        
        # 对比学习配置
        self.contrastive_lr = 3e-5  
        self.contrastive_weight_decay = 1e-5
        self.contrastive_epochs = 500  
        self.contrastive_patience = 30
        
        # 训练过程配置
        self.save_period = 10
        self.save_dir = 'logs'
        self.eval_flag = True
        self.eval_period = 10
        self.vocdevkit_path = '/home/wuyou/pspnet-pytorch/VOCdevkit/VOC2007'
        
        # 数据加载配置
        self.num_workers = max(4, min(16, os.cpu_count()//2))
        self.pin_memory = True
        self.persistent_workers = True

    def validate_config(self):
        """验证配置是否正确设置"""
        print(f"\n配置验证结果:")
        print(f"  对比学习预训练: {'✅ 开启' if self.use_contrastive_pretrain else '❌ 关闭'}")
        print(f"  Dice损失: {'✅ 开启' if self.dice_loss else '❌ 关闭'}")
        print(f"  Focal损失: {'✅ 开启' if self.focal_loss else '❌ 关闭'}")
        
        if self.dice_loss and self.focal_loss:
            print(f"  损失函数组合: ✅ 改进版DiceFocal (alpha={self.improved_alpha}, gamma={self.improved_gamma})")
        elif self.dice_loss:
            print(f"  损失函数组合:  纯Dice损失")
        elif self.focal_loss:
            print(f"  损失函数组合:  纯Focal损失")  
        else:
            print(f"  损失函数组合:  交叉熵损失")

    def check_and_download_weights(self):
        """检查并下载缺失的权重文件"""
        import urllib.request
        
        for backbone_type, weight_path in self.backbone_weights.items():
            if not os.path.exists(weight_path):
                print(f"警告: 权重文件不存在: {weight_path}")
                
                if backbone_type == 2:  # resnet101
                    download_url = "https://download.pytorch.org/models/resnet101-63fe2227.pth"
                    print(f"正在下载ResNet101权重...")
                    try:
                        urllib.request.urlretrieve(download_url, weight_path)
                        print(f"已下载: {weight_path}")
                    except Exception as e:
                        print(f"下载失败: {e}")
                        self.backbone_weights[2] = self.backbone_weights[1]
                elif backbone_type == 3:  # efficientnet_b0
                    print(f"使用现有的efficientnet_b6.pth作为efficientnet_b0的替代")
                    self.backbone_weights[3] = "/home/wuyou/pspnet-pytorch/model_data/efficientnet_b6.pth"
                else:
                    print(f"请确保权重文件存在: {weight_path}")
                    if backbone_type != 1:
                        self.backbone_weights[backbone_type] = self.backbone_weights[1]

    def get_experiment_name(self):
        """根据配置生成实验名称"""
        backbone_map = {
            0: "mobilenet", 1: "resnet50", 2: "resnet101", 3: "efficientnet",
            4: "resnet50_adv", 5: "mobilenet_small", 6: "mobilenetv2",
            7: "shufflenet_v1", 8: "shufflenet_v2", 9: "efficientnet_v2"
        }
        
        backbone_name = backbone_map.get(self.backbone_type, f"backbone_{self.backbone_type}")
        
        if self.dice_loss and self.focal_loss:
            loss_name = f"improved_alpha{self.improved_alpha}_gamma{self.improved_gamma}"
        elif self.dice_loss:
            loss_name = "dice"
        elif self.focal_loss:
            loss_name = "focal"
        else:
            loss_name = "ce"
            
        contrastive_name = "contrastive" if self.use_contrastive_pretrain else "no_contrastive"
        feature_name = "feature_eng" if self.use_feature_engineering else "no_feature"
        
        experiment_name = f"{backbone_name}_{loss_name}_{contrastive_name}_{feature_name}"
        
        return experiment_name

    def to_dict(self):
        """将配置转换为字典"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_') and not callable(value):
                try:
                    json.dumps(value)
                    config_dict[key] = value
                except:
                    config_dict[key] = str(value)
        return config_dict


# ==================== 工具函数 ====================
def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def check_dataset_structure(dataset_path):
    """检查并修复数据集目录结构"""
    print("检查数据集目录结构...")
    
    required_dirs = ["JPEGImages", "SegmentationClass", "ImageSets/Segmentation"]
    
    for dir_path in required_dirs:
        full_path = os.path.join(dataset_path, dir_path)
        if not os.path.exists(full_path):
            print(f"创建缺失目录: {full_path}")
            os.makedirs(full_path, exist_ok=True)
    
    segmentation_class_png = os.path.join(dataset_path, "SegmentationClassPNG")
    segmentation_class = os.path.join(dataset_path, "SegmentationClass")
    
    if os.path.exists(segmentation_class_png) and not os.path.exists(segmentation_class):
        print(f"创建符号链接: {segmentation_class} -> {segmentation_class_png}")
        try:
            os.symlink(segmentation_class_png, segmentation_class)
        except:
            import shutil
            shutil.copytree(segmentation_class_png, segmentation_class)
    
    jpeg_dir = os.path.join(dataset_path, "JPEGImages")
    seg_dir = os.path.join(dataset_path, "SegmentationClass")
    
    jpeg_files = [f for f in os.listdir(jpeg_dir) if any(f.endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])] if os.path.exists(jpeg_dir) else []
    seg_files = [f for f in os.listdir(seg_dir) if f.endswith('.png')] if os.path.exists(seg_dir) else []
    
    print(f"图像文件数量: {len(jpeg_files)}")
    print(f"标签文件数量: {len(seg_files)}")
    
    if jpeg_files and seg_files:
        jpeg_names = set([os.path.splitext(f)[0] for f in jpeg_files])
        seg_names = set([os.path.splitext(f)[0] for f in seg_files])
        common_names = jpeg_names.intersection(seg_names)
        print(f"匹配的图像-标签对数量: {len(common_names)}")
        return len(common_names) > 0
    
    return False


def filter_existing_files(lines, dataset_path):
    """过滤掉不存在的文件"""
    valid_lines = []
    for line in lines:
        name = line.strip()
        jpeg_path = os.path.join(dataset_path, "JPEGImages", name + ".jpg")
        png_path = os.path.join(dataset_path, "SegmentationClass", name + ".png")
        
        if os.path.exists(jpeg_path) and os.path.exists(png_path):
            valid_lines.append(line)
        else:
            print(f"警告: 跳过不存在的文件对 - {name}")
    
    return valid_lines


def get_lr(optimizer):
    """获取当前学习率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_fusion_weight(epoch, start_epoch=50, warmup_epochs=20):
    """根据训练进度计算融合权重"""
    if epoch < start_epoch:
        return 0.0
    elif epoch < start_epoch + warmup_epochs:
        progress = (epoch - start_epoch) / warmup_epochs
        return progress * 0.8
    else:
        return 0.8


# ==================== 学习率调度函数 ====================
def yolox_warm_cos_lr(optimizer, epoch, warmup_epochs, warmup_lr, init_lr, min_lr, total_epochs):
    """YOLOX风格的热身+余弦退火学习率调度"""
    if epoch < warmup_epochs:
        lr = warmup_lr + (init_lr - warmup_lr) * epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (init_lr - min_lr) * (1 + math.cos(math.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def step_lr(optimizer, epoch, init_lr, min_lr, total_epochs):
    """步进学习率调度"""
    if epoch < total_epochs * 0.6:
        lr = init_lr
    elif epoch < total_epochs * 0.8:
        lr = init_lr * 0.1
    else:
        lr = init_lr * 0.01
    
    lr = max(lr, min_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr_scheduler(lr_decay_type, init_lr, min_lr, total_epochs, warmup_epochs=0, warmup_lr=0):
    """获取学习率调度函数"""
    if lr_decay_type == 'cos':
        def lr_scheduler_func(optimizer, epoch):
            yolox_warm_cos_lr(optimizer, epoch, warmup_epochs, warmup_lr, init_lr, min_lr, total_epochs)
        return lr_scheduler_func
    elif lr_decay_type == 'step':
        def lr_scheduler_func(optimizer, epoch):
            step_lr(optimizer, epoch, init_lr, min_lr, total_epochs)
        return lr_scheduler_func
    else:
        def lr_scheduler_func(optimizer, epoch):
            yolox_warm_cos_lr(optimizer, epoch, warmup_epochs, warmup_lr, init_lr, min_lr, total_epochs)
        return lr_scheduler_func


def set_optimizer_lr_with_warmup(optimizer, lr_scheduler_func, epoch, warmup_epochs, warmup_lr, init_lr):
    """带预热的学习率调度 - 修复版本"""
    if epoch < warmup_epochs:
        lr = warmup_lr + (init_lr - warmup_lr) * epoch / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr_scheduler_func(optimizer, epoch)


# ==================== 特征工程工具类 ====================
class FeatureEngineering:
    """特征工程工具类"""
    
    @staticmethod
    def extract_all_features(image, target_size=(512, 512)):
        """提取所有手工特征"""
        if image.shape[:2] != target_size:
            image = cv2.resize(image, target_size)
        
        features = []
        
        # 1. 原始RGB通道
        r_channel = image[:, :, 0].astype(np.float32) / 255.0
        g_channel = image[:, :, 1].astype(np.float32) / 255.0
        b_channel = image[:, :, 2].astype(np.float32) / 255.0
        features.extend([r_channel, g_channel, b_channel])
        
        # 2. HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h_channel = hsv[:, :, 0].astype(np.float32) / 180.0
        s_channel = hsv[:, :, 1].astype(np.float32) / 255.0
        v_channel = hsv[:, :, 2].astype(np.float32) / 255.0
        features.extend([h_channel, s_channel, v_channel])
        
        # 3. LAB颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(np.float32) / 255.0
        a_channel = (lab[:, :, 1].astype(np.float32) + 128) / 255.0
        b_channel_lab = (lab[:, :, 2].astype(np.float32) + 128) / 255.0
        features.extend([l_channel, a_channel, b_channel_lab])
        
        # 4. 灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        features.append(gray)
        
        # 5. 梯度特征 (Sobel)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sobel_magnitude = (sobel_magnitude - sobel_magnitude.min()) / (sobel_magnitude.max() - sobel_magnitude.min() + 1e-8)
        features.append(sobel_magnitude.astype(np.float32))
        
        # 6. 高斯模糊
        gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
        features.append(gaussian_blur.astype(np.float32))
        
        # 7. 边缘检测 (Canny)
        edges = cv2.Canny(image, 50, 150).astype(np.float32) / 255.0
        features.append(edges)
        
        # 8. 纹理特征 (LBP简化版)
        texture = FeatureEngineering._simple_lbp(gray)
        features.append(texture.astype(np.float32))
        
        # 9. 局部二值模式方差
        lbp_var = FeatureEngineering._local_variance(gray)
        features.append(lbp_var.astype(np.float32))
        
        # 10. 颜色直方图特征 (简化)
        hist_features = FeatureEngineering._color_histogram_features(image)
        features.append(hist_features.astype(np.float32))
        
        # 堆叠所有特征
        feature_stack = np.stack(features, axis=-1)
        
        return feature_stack
    
    @staticmethod
    def _simple_lbp(image, radius=1, neighbors=8):
        """简化的LBP特征"""
        h, w = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(radius, h-radius):
            for j in range(radius, w-radius):
                center = image[i, j]
                binary = 0
                for k in range(neighbors):
                    angle = 2 * np.pi * k / neighbors
                    x = i + int(radius * np.sin(angle))
                    y = j + int(radius * np.cos(angle))
                    if image[x, y] >= center:
                        binary |= (1 << k)
                lbp[i, j] = binary
        
        lbp = lbp / 255.0
        return lbp
    
    @staticmethod
    def _local_variance(image, window_size=3):
        """计算局部方差"""
        h, w = image.shape
        variance = np.zeros_like(image)
        pad = window_size // 2
        
        padded = np.pad(image, pad, mode='reflect')
        
        for i in range(h):
            for j in range(w):
                window = padded[i:i+window_size, j:j+window_size]
                variance[i, j] = np.var(window)
        
        if variance.max() > 0:
            variance = variance / variance.max()
        
        return variance
    
    @staticmethod
    def _color_histogram_features(image, bins=8):
        """颜色直方图特征"""
        h, w = image.shape[:2]
        features = np.zeros((h, w))
        
        block_size = 32
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = image[i:min(i+block_size, h), j:min(j+block_size, w)]
                if block.size > 0:
                    hist_r = cv2.calcHist([block], [0], None, [bins], [0, 256])
                    hist_g = cv2.calcHist([block], [1], None, [bins], [0, 256])
                    hist_b = cv2.calcHist([block], [2], None, [bins], [0, 256])
                    
                    hist_feature = np.concatenate([hist_r, hist_g, hist_b]).flatten()
                    hist_value = np.mean(hist_feature) / 255.0
                    
                    features[i:min(i+block_size, h), j:min(j+block_size, w)] = hist_value
        
        return features


# ==================== 数据集类 ====================
class EnhancedPSPnetDataset(torch.utils.data.Dataset):
    """增强的数据集类，支持特征工程"""
    
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path, use_feature_engineering=True):
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path
        self.use_feature_engineering = use_feature_engineering
        
        self.image_paths = []
        self.label_paths = []
        
        for line in annotation_lines:
            name = line.split()[0]
            
            jpeg_path = os.path.join(dataset_path, "JPEGImages", name + ".jpg")
            if not os.path.exists(jpeg_path):
                jpeg_path = os.path.join(dataset_path, "JPEGImages", name + ".png")
            
            png_path = os.path.join(dataset_path, "SegmentationClass", name + ".png")
            if not os.path.exists(png_path):
                png_path = os.path.join(dataset_path, "SegmentationClassPNG", name + ".png")
            
            self.image_paths.append(jpeg_path)
            self.label_paths.append(png_path)
        
    def __len__(self):
        return len(self.annotation_lines)
    
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        """数据增强函数"""
        image = image.convert('RGB')
        label = Image.fromarray(np.array(label))

        iw, ih = image.size
        h, w = input_shape

        if not random:
            scale = min(w/iw, h/ih)
            nw, nh = int(iw*scale), int(ih*scale)
            dx, dy = (w-nw)//2, (h-nh)//2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image = new_image

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', (w, h), (0))
            new_label.paste(label, (dx, dy))
            label = new_label
            return image, label

        new_ar = w/h * self.rand(1-jitter, 1+jitter) / self.rand(1-jitter, 1+jitter)
        scale = self.rand(.5, 2)

        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)

        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))

        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image, label = new_image, new_label

        if self.rand() < .5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1/self.rand(1, val)
        
        x = cv2.cvtColor(np.array(image, np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        return image_data, label

    def __getitem__(self, index):
        jpeg_path = self.image_paths[index]
        png_path = self.label_paths[index]
        
        image = Image.open(jpeg_path)
        png = Image.open(png_path)
        
        image, png = self.get_random_data(image, png, self.input_shape, random=self.train)
        
        image = np.transpose(preprocess_input(np.array(image, np.float32)), (2, 0, 1))
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))
        
        if self.use_feature_engineering:
            try:
                image_np = image.transpose(1, 2, 0)
                image_np = (image_np * 255).astype(np.uint8)
                
                handcrafted_features = FeatureEngineering.extract_all_features(
                    image_np, target_size=self.input_shape
                )
                
                handcrafted_tensor = torch.from_numpy(handcrafted_features.transpose(2, 0, 1).astype('float32'))
                
                return torch.from_numpy(image), torch.from_numpy(png), torch.from_numpy(seg_labels), handcrafted_tensor
            except Exception as e:
                print(f"特征工程处理失败: {e}")
                handcrafted_tensor = torch.zeros(16, self.input_shape[0], self.input_shape[1])
                return torch.from_numpy(image), torch.from_numpy(png), torch.from_numpy(seg_labels), handcrafted_tensor
        else:
            return torch.from_numpy(image), torch.from_numpy(png), torch.from_numpy(seg_labels)


def enhanced_pspnet_dataset_collate(batch):
    """数据加载collate函数"""
    if len(batch[0]) == 4:
        images, pngs, seg_labels, handcrafted_features = zip(*batch)
        return (torch.stack(images), torch.stack(pngs), 
                torch.stack(seg_labels), torch.stack(handcrafted_features))
    else:
        images, pngs, seg_labels = zip(*batch)
        return torch.stack(images), torch.stack(pngs), torch.stack(seg_labels)


# ==================== 损失函数 ====================
class ImprovedDiceFocalLoss(nn.Module):
    """改进的Dice Focal Loss - 简化版本"""
    def __init__(self, alpha=0.5, gamma=2.0, smooth=1e-6, class_weights=None):
        super().__init__()
        self.alpha = alpha      # Dice权重
        self.gamma = gamma      # Focal聚焦参数  
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, pred, target):
        # Dice Loss部分
        n, c, h, w = pred.size()
        nt, ht, wt = target.size()
        
        if h != ht and w != wt:
            pred = F.interpolate(pred, size=(ht, wt), mode="bilinear", align_corners=True)
            
        temp_inputs = torch.softmax(pred.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
        temp_target = F.one_hot(target, num_classes=c).view(n, -1, c)
        
        tp = torch.sum(temp_target * temp_inputs, axis=[0,1])
        fp = torch.sum(temp_inputs, axis=[0,1]) - tp
        fn = torch.sum(temp_target, axis=[0,1]) - tp
        
        score = ((1 + 1 ** 2) * tp + self.smooth) / ((1 + 1 ** 2) * tp + 1 ** 2 * fn + fp + self.smooth)
        dice_loss = 1 - torch.mean(score)
        
        # Focal Loss部分
        temp_inputs_flat = pred.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        temp_target_flat = target.view(-1)
        
        ce_loss = F.cross_entropy(temp_inputs_flat, temp_target_flat, 
                                weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        
        # 组合
        return self.alpha * dice_loss + (1 - self.alpha) * focal_loss


def calculate_loss(outputs, targets, seg_labels, weights, config):
    """统一的损失函数计算器 - 重构版本"""
    
    # 根据开关选择损失函数类型
    if config.dice_loss and config.focal_loss:
        # 使用改进版DiceFocal损失
        loss_fn = ImprovedDiceFocalLoss(
            alpha=config.improved_alpha, 
            gamma=config.improved_gamma, 
            class_weights=weights
        )
        loss = loss_fn(outputs, targets)
        loss_details = {
            "total": loss.item(), 
            "type": f"improved_dice_focal(α={config.improved_alpha},γ={config.improved_gamma})",
            "components": {
                "dice": (config.improved_alpha * loss.item()),
                "focal": ((1 - config.improved_alpha) * loss.item())
            }
        }
        
    elif config.dice_loss:
        # 纯Dice损失
        loss = Dice_loss(outputs, seg_labels)
        loss_details = {
            "total": loss.item(),
            "type": "dice_only",
            "components": {"dice": loss.item()}
        }
        
    elif config.focal_loss:
        # 纯Focal损失
        loss = Focal_Loss(outputs, targets, weights, num_classes=config.num_classes)
        loss_details = {
            "total": loss.item(), 
            "type": "focal_only",
            "components": {"focal": loss.item()}
        }
        
    else:
        # 交叉熵损失
        loss = CE_Loss(outputs, targets, weights, num_classes=config.num_classes)
        loss_details = {
            "total": loss.item(),
            "type": "cross_entropy", 
            "components": {"ce": loss.item()}
        }
    
    return loss, loss_details


# ==================== 训练函数 ====================
def fit_one_epoch_with_features(model_train, model, loss_history, eval_callback, optimizer, ema_train, ema, epoch, 
                              epoch_step, epoch_step_val, gen, gen_val, unfreeze_epoch, cuda, cls_weights, aux_branch, num_classes, fp16, scaler, save_period, save_dir, distributed, local_rank,
                              use_feature_engineering=False, fusion_weight=1.0, config=None):
    """支持特征工程的训练函数 - 修复设备问题"""
    
    total_loss = 0
    actual_model_train = model_train.module if hasattr(model_train, 'module') else model_train

    if hasattr(actual_model_train, 'set_fusion_weight'):
        actual_model_train.set_fusion_weight(fusion_weight)

    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{unfreeze_epoch}', postfix=dict, mininterval=0.3) as pbar:
        model_train.train()
        
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
                
            try:
                # 准备权重张量
                weights = torch.from_numpy(cls_weights).float()
                if cuda:
                    weights = weights.cuda(local_rank)
                
                # 修复：确保所有张量都在正确的设备上
                if use_feature_engineering and len(batch) == 4:
                    images, targets, seg_labels, handcrafted_features = batch
                    if cuda:
                        images = images.cuda(local_rank, non_blocking=True)
                        targets = targets.cuda(local_rank, non_blocking=True).long()
                        seg_labels = seg_labels.cuda(local_rank, non_blocking=True)
                        handcrafted_features = handcrafted_features.cuda(local_rank, non_blocking=True)
                    
                    if fp16:
                        with torch.amp.autocast('cuda'):
                            outputs = model_train(images, handcrafted_features)
                            loss, loss_details = calculate_loss(outputs, targets, seg_labels, weights, config)
                    else:
                        outputs = model_train(images, handcrafted_features)
                        loss, loss_details = calculate_loss(outputs, targets, seg_labels, weights, config)
                    
                else:
                    images, targets, seg_labels = batch
                    if cuda:
                        images = images.cuda(local_rank, non_blocking=True)
                        targets = targets.cuda(local_rank, non_blocking=True).long()
                        seg_labels = seg_labels.cuda(local_rank, non_blocking=True)
                    
                    if fp16:
                        with torch.amp.autocast('cuda'):
                            outputs = model_train(images)
                            loss, loss_details = calculate_loss(outputs, targets, seg_labels, weights, config)
                    else:
                        outputs = model_train(images)
                        loss, loss_details = calculate_loss(outputs, targets, seg_labels, weights, config)

                optimizer.zero_grad()
                
                if fp16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

                postfix_dict = {
                    'total_loss': total_loss / (iteration + 1), 
                    'lr': get_lr(optimizer),
                    'loss_type': loss_details['type']
                }
                
                if 'components' in loss_details:
                    for comp_name, comp_value in loss_details['components'].items():
                        postfix_dict[comp_name] = comp_value
                
                if fusion_weight > 0 and use_feature_engineering:
                    postfix_dict['fusion_weight'] = fusion_weight
                
                pbar.set_postfix(**postfix_dict)
                pbar.update(1)
                
            except Exception as e:
                print(f"训练过程中出错: {e}")
                import traceback
                traceback.print_exc()
                continue

    print('Finish Train')
    
    print('Start Validation')
    model_train.eval()
    total_val_loss, val_iterations = 0, 0
    
    with torch.no_grad():
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
                
            try:
                # 准备权重张量
                weights = torch.from_numpy(cls_weights).float()
                if cuda:
                    weights = weights.cuda(local_rank)
                
                # 修复：验证过程中确保所有张量都在正确的设备上
                if use_feature_engineering and len(batch) == 4:
                    images, targets, seg_labels, handcrafted_features = batch
                    if cuda:
                        images = images.cuda(local_rank, non_blocking=True)
                        targets = targets.cuda(local_rank, non_blocking=True).long()
                        seg_labels = seg_labels.cuda(local_rank, non_blocking=True)  # 修复：添加seg_labels设备转移
                        handcrafted_features = handcrafted_features.cuda(local_rank, non_blocking=True)
                    
                    if fp16:
                        with torch.amp.autocast('cuda'):
                            outputs = model_train(images, handcrafted_features)
                    else:
                        outputs = model_train(images, handcrafted_features)
                else:
                    images, targets, seg_labels = batch
                    if cuda:
                        images = images.cuda(local_rank, non_blocking=True)
                        targets = targets.cuda(local_rank, non_blocking=True).long()
                        seg_labels = seg_labels.cuda(local_rank, non_blocking=True)  # 修复：添加seg_labels设备转移
                    
                    if fp16:
                        with torch.amp.autocast('cuda'):
                            outputs = model_train(images)
                    else:
                        outputs = model_train(images)
                
                # 计算验证损失 - 确保所有输入都在相同设备上
                loss, loss_details = calculate_loss(outputs, targets, seg_labels, weights, config)
                total_val_loss += loss.item()
                val_iterations += 1
                
            except Exception as e:
                print(f"验证过程中出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    val_loss = total_val_loss / val_iterations if val_iterations > 0 else 0
    print('Finish Validation')
    
    train_loss = total_loss / epoch_step if epoch_step > 0 else 0
    
    if loss_history is not None:
        try:
            loss_history.append_loss(epoch + 1, train_loss, val_loss)
        except Exception as e:
            print(f"记录损失时出错: {e}")
        
    if (epoch + 1) % save_period == 0 or epoch + 1 == unfreeze_epoch:
        save_path = os.path.join(save_dir, f"ep{epoch+1}-loss{train_loss:.3f}-val{val_loss:.3f}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"模型已保存: {save_path}")


# ==================== 对比学习相关函数 ====================
def check_contrastive_pretrain_status(save_path):
    """检查对比学习预训练状态"""
    if not os.path.exists(save_path):
        return False, "未找到对比学习预训练权重文件"
    
    try:
        checkpoint = torch.load(save_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and len(checkpoint) > 0:
            return True, "对比学习预训练已完成"
        else:
            return False, "对比学习预训练权重文件无效"
    except Exception as e:
        return False, f"对比学习预训练权重文件损坏: {e}"


def find_best_contrastive_weight(experiment_weight_dir, backbone_type=1):
    """使用对比学习分析工具找到最佳权重"""
    print(f" 正在在 {experiment_weight_dir} 中寻找最佳的对比学习权重...")
    
    if not os.path.exists(experiment_weight_dir):
        print(f"❌ 目录不存在: {experiment_weight_dir}")
        return None
    
    try:
        analyzer = ContrastiveAnalysisFixed(
            output_dir=os.path.join(experiment_weight_dir, 'contrastive_analysis')
        )
        
        analyzer.run_comprehensive_analysis(experiment_weight_dir)
        
        if hasattr(analyzer, 'results') and analyzer.results:
            print(f"✅ 分析器找到 {len(analyzer.results)} 个有效的对比学习权重")
            
            if hasattr(analyzer, 'best_epoch') and analyzer.best_epoch is not None:
                best_file = analyzer.results[analyzer.best_epoch]['file_path']
                print(f"✅ 找到最佳对比学习权重: {os.path.basename(best_file)}")
                print(f"   选择理由: {getattr(analyzer, 'best_reason', '未知')}")
                print(f"   指标值: {getattr(analyzer, 'best_metric', 0):.6f}")
                return best_file
            else:
                last_epoch = max(analyzer.results.keys())
                best_file = analyzer.results[last_epoch]['file_path']
                print(f"✅ 使用最新的对比学习权重: {os.path.basename(best_file)} (epoch {last_epoch})")
                return best_file
        else:
            print(f"❌ 在 {experiment_weight_dir} 中没有找到可加载的对比学习权重")
            return None
        
    except Exception as e:
        print(f"❌ 对比学习权重分析失败: {e}")
        return find_latest_contrastive_weight(experiment_weight_dir)


def find_latest_contrastive_weight(experiment_weight_dir):
    """直接查找最新的对比学习权重文件"""
    print(" 使用备用方法查找对比学习权重...")
    
    contrastive_files = []
    for file in os.listdir(experiment_weight_dir):
        if file.startswith('contrastive_pretrained_backbone') and file.endswith('.pth'):
            file_path = os.path.join(experiment_weight_dir, file)
            contrastive_files.append(file_path)
    
    if not contrastive_files:
        print("❌ 没有找到任何对比学习权重文件")
        return None
    
    contrastive_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_file = contrastive_files[0]
    
    epoch_match = None
    if 'epoch' in latest_file:
        import re
        match = re.search(r'epoch(\d+)', latest_file)
        if match:
            epoch_match = int(match.group(1))
    
    if epoch_match:
        print(f"✅ 找到最新的对比学习权重: {os.path.basename(latest_file)} (epoch {epoch_match})")
    else:
        print(f"✅ 找到对比学习权重: {os.path.basename(latest_file)}")
    
    return latest_file


def contrastive_pretrain(model, dataloader, optimizer, device, epochs=500, save_path=None, fp16=False, patience=30):
    """对比学习预训练"""
    contrastive_learner = ContrastiveLearner(model.backbone_manager.backbone)
    contrastive_learner.to(device)
    contrastive_learner.train()

    scaler = torch.amp.GradScaler('cuda') if fp16 else None
    start_epoch, matched_keys = 0, 0
    
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"创建保存目录: {save_dir}")
    
    existing_weights = []
    if save_path and os.path.exists(os.path.dirname(save_path)):
        save_dir = os.path.dirname(save_path)
        for file in os.listdir(save_dir):
            if file.startswith('contrastive_pretrained_backbone') and file.endswith('.pth'):
                existing_weights.append(os.path.join(save_dir, file))
    
    if existing_weights:
        print(f"发现 {len(existing_weights)} 个现有的对比学习权重文件")
        best_weight = find_best_contrastive_weight(save_dir)
        if best_weight:
            print(f"加载最佳对比学习权重: {best_weight}")
            try:
                pretrained_dict = torch.load(best_weight, map_location=device, weights_only=False)
                model_dict = contrastive_learner.backbone.state_dict()
                
                matched_dict = {}
                for k, v in pretrained_dict.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        matched_dict[k] = v
                        matched_keys += 1
                
                if matched_keys > 0:
                    contrastive_learner.backbone.load_state_dict(matched_dict, strict=False)
                    print(f"成功加载 {matched_keys} 个对比学习预训练参数")
                    if 'epoch' in best_weight:
                        try:
                            epoch_str = best_weight.split('epoch')[-1].replace('.pth', '')
                            if epoch_str.isdigit():
                                start_epoch = int(epoch_str)
                                print(f"从第 {start_epoch} 个epoch继续训练")
                        except:
                            start_epoch = 0
                else:
                    print("没有匹配的预训练参数，将从零开始训练")
            except Exception as e:
                print(f"加载对比学习权重失败: {e}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7
    )
    
    best_loss = float('inf')
    patience_counter = 0
    
    print(f"开始对比学习预训练，从第 {start_epoch} 个epoch开始，总共 {epochs} 个epoch")
    print(f"早停机制: 连续 {patience} 个epoch没有改善将停止训练")
    
    for epoch in range(start_epoch, epochs):
        total_loss, num_batches = 0, 0
        
        for batch_idx, (x1, x2) in enumerate(dataloader):
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()

            if fp16:
                with torch.amp.autocast('cuda'):
                    z1, z2 = contrastive_learner(x1, x2)
                    loss = contrastive_learner.contrastive_loss(z1, z2)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                z1, z2 = contrastive_learner(x1, x2)
                loss = contrastive_learner.contrastive_loss(z1, z2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(contrastive_learner.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch: {epoch} [{batch_idx * len(x1)}/{len(dataloader.dataset)} '
                      f'({100. * batch_idx / len(dataloader):.0f}%)] Loss: {loss.item():.6f} LR: {current_lr:.6f}')

        scheduler.step()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch} Average Loss: {avg_loss:.6f} LR: {current_lr:.6f}')
        
        if avg_loss < best_loss * 0.999:
            best_loss = avg_loss
            patience_counter = 0
            if save_path:
                checkpoint_path = save_path.replace('.pth', f'_epoch{epoch+1}.pth')
                try:
                    torch.save(contrastive_learner.backbone.state_dict(), checkpoint_path)
                    print(f"Checkpoint saved at epoch {epoch + 1}")
                    
                    torch.save(contrastive_learner.backbone.state_dict(), save_path)
                    print(f"新的最佳模型已保存: {save_path} (Loss: {best_loss:.6f})")
                except Exception as e:
                    print(f"保存模型失败: {e}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停：在 epoch {epoch} 触发，连续 {patience} 个epoch没有显著改善")
                print(f"最佳损失: {best_loss:.6f}")
                break

    if save_path:
        try:
            torch.save(contrastive_learner.backbone.state_dict(), save_path)
            print(f"对比学习预训练权重已保存: {save_path} (最终Loss: {avg_loss:.6f})")
        except Exception as e:
            print(f"最终保存失败: {e}")
    
    return contrastive_learner


# ==================== 主训练类 ====================
class PSPNetTrainer:
    """PSPNet训练器类"""
    
    def __init__(self, config, experiment_manager):
        self.config = config
        self.experiment_manager = experiment_manager
        self.device = self._setup_device()
        self.model = None
        self.optimizer = None
        self.loss_history = None
        self.eval_callback = None
        
    def _setup_device(self):
        """设置训练设备"""
        if self.config.distributed:
            dist.init_process_group(backend="nccl")
            local_rank = int(os.environ["LOCAL_RANK"])
            device = torch.device("cuda", local_rank)
            if local_rank == 0:
                print(f"[{os.getpid()}] training...")
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.config.local_rank = 0

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        if self.config.local_rank == 0 and torch.cuda.is_available():
            print(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        
        return device

    def _create_dataset_splits(self):
        """创建数据集划分"""
        train_txt_path = os.path.join(self.config.vocdevkit_path, "ImageSets/Segmentation/train.txt")
        val_txt_path = os.path.join(self.config.vocdevkit_path, "ImageSets/Segmentation/val.txt")

        if not os.path.exists(train_txt_path) or not os.path.exists(val_txt_path):
            self._create_dataset_files(train_txt_path, val_txt_path)

        with open(train_txt_path, "r") as f:
            train_lines_original = f.readlines()
        with open(val_txt_path, "r") as f:
            val_lines_original = f.readlines()

        train_lines = filter_existing_files(train_lines_original, self.config.vocdevkit_path)
        val_lines = filter_existing_files(val_lines_original, self.config.vocdevkit_path)

        if len(train_lines) != len(train_lines_original) or len(val_lines) != len(val_lines_original):
            print(f"过滤后训练集: {len(train_lines)} 个样本 (原 {len(train_lines_original)})")
            print(f"过滤后验证集: {len(val_lines)} 个样本 (原 {len(val_lines_original)})")
            
            with open(train_txt_path, "w") as f:
                for line in train_lines:
                    f.write(line)
            with open(val_txt_path, "w") as f:
                for line in val_lines:
                    f.write(line)
            
            print("已更新数据集文件，移除了不存在的文件")

        return train_lines, val_lines

    def _create_dataset_files(self, train_txt_path, val_txt_path):
        """创建数据集文件"""
        print("检测到缺少训练集/验证集文件，正在自动创建...")
        
        jpeg_dir = os.path.join(self.config.vocdevkit_path, "JPEGImages")
        segmentation_dir = os.path.join(self.config.vocdevkit_path, "SegmentationClass")
        
        image_files = [f for f in os.listdir(jpeg_dir) if any(f.endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])]
        label_files = [f for f in os.listdir(segmentation_dir) if f.endswith('.png')]
        
        image_names = set([os.path.splitext(f)[0] for f in image_files])
        label_names = set([os.path.splitext(f)[0] for f in label_files])
        
        valid_names = list(image_names.intersection(label_names))
        
        if len(valid_names) == 0:
            print("错误: 未找到匹配的图像和标签文件！")
            exit(1)
        
        print(f"找到 {len(valid_names)} 个有效的图像-标签对")
        
        random.shuffle(valid_names)
        split_idx = int(0.8 * len(valid_names))
        
        train_names = valid_names[:split_idx]
        val_names = valid_names[split_idx:]
        
        os.makedirs(os.path.dirname(train_txt_path), exist_ok=True)
        
        with open(train_txt_path, "w") as f:
            for name in train_names:
                f.write(name + "\n")
        
        with open(val_txt_path, "w") as f:
            for name in val_names:
                f.write(name + "\n")
        
        print(f"已创建训练集文件: {train_txt_path} (包含 {len(train_names)} 个样本)")
        print(f"已创建验证集文件: {val_txt_path} (包含 {len(val_names)} 个样本)")

    def _setup_model(self):
        """设置模型"""
        self.model = UnifiedPSPNet(
            num_classes=self.config.num_classes,
            backbone_type=self.config.backbone_type,
            downsample_factor=self.config.downsample_factor,
            use_feature_engineering=self.config.use_feature_engineering,
            pretrained=False
        )
        
        if self.config.backbone_pretrained:
            weight_path = self.config.backbone_weights.get(self.config.backbone_type)
            if weight_path and os.path.exists(weight_path):
                print(f"加载主干网络预训练权重: {weight_path}")
                success = self.model.backbone_manager.load_pretrained(weight_path)
                if not success:
                    print("警告: 预训练权重加载失败，使用随机初始化")
            else:
                print(f"警告: 未找到主干网络权重: {weight_path}")
                print("将使用随机初始化的主干网络")
        
        return self.model

    def _setup_training(self):
        """设置训练环境"""
        if self.config.local_rank == 0:
            time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
            log_dir = os.path.join(self.experiment_manager.get_experiment_log_dir(), "loss_" + str(time_str))
            self.loss_history = LossHistory(log_dir, self.model, input_shape=self.config.input_shape)
        else:
            self.loss_history = None

        scaler = torch.amp.GradScaler('cuda') if self.config.fp16 else None

        model_train = self.model.train()
        
        if self.config.sync_bn and torch.cuda.device_count() > 1 and self.config.distributed:
            model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)

        if self.config.cuda:
            if self.config.distributed:
                model_train = model_train.cuda(self.config.local_rank)
                model_train = torch.nn.parallel.DistributedDataParallel(
                    model_train, device_ids=[self.config.local_rank], find_unused_parameters=True
                )
            else:
                model_train = torch.nn.DataParallel(self.model)
                cudnn.benchmark = True
                model_train = model_train.cuda()
        
        return model_train, scaler

    def _contrastive_pretraining(self, model_train):
        """对比学习预训练"""
        print(f"\n{'='*50}")
        print(f"对比学习预训练状态检查")
        print(f"{'='*50}")
        
        if not self.config.use_contrastive_pretrain:
            print("❌ 配置为不使用对比学习预训练，跳过此步骤")
            print(f"{'='*50}\n")
            return model_train
            
        print("✅ 配置为使用对比学习预训练，继续执行...")
        print(f"{'='*50}\n")
        
        contrastive_pretrain_path = os.path.join(self.experiment_manager.get_experiment_weight_dir(), 'contrastive_pretrained_backbone.pth')
        
        print(" 搜索现有的对比学习权重...")
        
        search_dirs = [
            self.experiment_manager.get_experiment_weight_dir(),
            "/home/wuyou/pspnet-pytorch/logs/exp_20251016_095016/weights",
            os.path.dirname(self.experiment_manager.get_experiment_weight_dir()),
        ]
        
        best_weight = None
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                print(f"在目录中搜索: {search_dir}")
                found_weight = find_best_contrastive_weight(search_dir, self.config.backbone_type)
                if found_weight:
                    best_weight = found_weight
                    print(f"✅ 在 {search_dir} 中找到对比学习权重")
                    break
        
        if best_weight:
            print(f"✅ 加载现有对比学习权重: {os.path.basename(best_weight)}")
            try:
                pretrained_dict = torch.load(best_weight, map_location=self.device, weights_only=False)
                model_dict = self.model.backbone_manager.backbone.state_dict()
                
                matched_dict = {}
                matched_count = 0
                for k, v in pretrained_dict.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        matched_dict[k] = v
                        matched_count += 1
                
                if matched_dict:
                    self.model.backbone_manager.backbone.load_state_dict(matched_dict, strict=False)
                    print(f"✅ 成功加载 {matched_count} 个对比学习预训练参数")
                    return model_train
                else:
                    print("❌ 没有匹配的预训练参数，将进行新的对比学习预训练")
            except Exception as e:
                print(f"❌ 加载对比学习权重失败: {e}")
                print("将进行新的对比学习预训练")
        else:
            print("ℹ️  没有找到现有的对比学习权重，开始新的对比学习预训练")
        
        print("\n 开始新的对比学习预训练...")
        
        contrastive_dataset_path = os.path.join(self.config.vocdevkit_path, "JPEGImages")
        if os.path.exists(contrastive_dataset_path):
            print(f"使用对比学习数据集路径: {contrastive_dataset_path}")
            unlabeled_dataset = UnlabeledDataset(contrastive_dataset_path, transform=contrastive_transform)
            unlabeled_loader = DataLoader(
                unlabeled_dataset, batch_size=8, shuffle=True, num_workers=4, 
                pin_memory=True, persistent_workers=True, prefetch_factor=2
            )

            print(f"创建对比学习优化器 (lr={self.config.contrastive_lr})")
            contrastive_optimizer = torch.optim.AdamW(
                self.model.backbone_manager.backbone.parameters(), 
                lr=self.config.contrastive_lr, 
                weight_decay=self.config.contrastive_weight_decay, 
                betas=(0.9, 0.98)
            )
            
            print(f"开始对比学习预训练，共 {self.config.contrastive_epochs} 个epoch")
            contrastive_learner = contrastive_pretrain(
                self.model, unlabeled_loader, contrastive_optimizer, self.device, 
                epochs=self.config.contrastive_epochs, 
                save_path=contrastive_pretrain_path, 
                fp16=self.config.fp16,
                patience=self.config.contrastive_patience
            )
            print("✅ 对比学习预训练完成")
            
            self.model.backbone_manager.backbone.load_state_dict(contrastive_learner.backbone.state_dict())
            
            print("分析新训练的对比学习权重...")
            best_weight = find_best_contrastive_weight(self.experiment_manager.get_experiment_weight_dir(), self.config.backbone_type)
            if best_weight:
                print(f"✅ 使用新训练的最佳权重: {os.path.basename(best_weight)}")
        else:
            print(f"❌ 对比学习数据集路径不存在: {contrastive_dataset_path}")
            print("跳过对比学习预训练")
        
        return model_train

    def _setup_data_loaders(self, train_lines, val_lines):
        """设置数据加载器"""
        nbs = 16
        effective_batch_size = self.config.unfreeze_batch_size
        lr_limit_max = 1e-3 if self.config.optimizer_type == 'adam' else 1e-1
        lr_limit_min = 1e-4 if self.config.optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(effective_batch_size / nbs * self.config.init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(effective_batch_size / nbs * self.config.min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        print(f"实际批次大小={self.config.unfreeze_batch_size}")
        print(f"学习率调整: 初始学习率={Init_lr_fit:.6f}, 最小学习率={Min_lr_fit:.6f}")

        if self.config.optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=Init_lr_fit, 
                betas=(self.config.momentum, 0.999), 
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = {
                'adam': optim.Adam(self.model.parameters(), Init_lr_fit, betas=(self.config.momentum, 0.999), weight_decay=self.config.weight_decay),
                'sgd': optim.SGD(self.model.parameters(), Init_lr_fit, momentum=self.config.momentum, nesterov=True, weight_decay=self.config.weight_decay)
            }[self.config.optimizer_type]

        lr_scheduler_func = get_lr_scheduler(
            self.config.lr_decay_type, 
            Init_lr_fit, 
            Min_lr_fit, 
            self.config.unfreeze_epoch,
            self.config.warmup_epochs,
            self.config.warmup_lr
        )

        batch_size = self.config.freeze_batch_size if self.config.freeze_train else self.config.unfreeze_batch_size
        epoch_step = len(train_lines) // batch_size
        epoch_step_val = len(val_lines) // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset = EnhancedPSPnetDataset(
            train_lines, self.config.input_shape, self.config.num_classes, True, 
            self.config.vocdevkit_path, self.config.use_feature_engineering
        )
        val_dataset = EnhancedPSPnetDataset(
            val_lines, self.config.input_shape, self.config.num_classes, False, 
            self.config.vocdevkit_path, self.config.use_feature_engineering
        )

        gen = DataLoader(
            train_dataset, 
            shuffle=True, 
            batch_size=batch_size, 
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True, 
            collate_fn=enhanced_pspnet_dataset_collate,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=4,
            multiprocessing_context='spawn' if os.name != 'nt' else None
        )
        gen_val = DataLoader(
            val_dataset, 
            shuffle=True, 
            batch_size=batch_size, 
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True, 
            collate_fn=enhanced_pspnet_dataset_collate,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=4,
            multiprocessing_context='spawn' if os.name != 'nt' else None
        )

        return gen, gen_val, epoch_step, epoch_step_val, lr_scheduler_func, batch_size, Init_lr_fit, Min_lr_fit

    def _training_loop(self, model_train, gen, gen_val, epoch_step, epoch_step_val, 
                      lr_scheduler_func, batch_size, scaler, Init_lr_fit, Min_lr_fit):
        """训练循环"""
        unfreeze_flag = False

        if self.config.freeze_train:
            for param in self.model.backbone_manager.backbone.parameters():
                param.requires_grad = False

        for epoch in range(self.config.init_epoch, self.config.unfreeze_epoch):
            if epoch >= self.config.freeze_epoch and not unfreeze_flag and self.config.freeze_train:
                batch_size = self.config.unfreeze_batch_size
                epoch_step = len(gen.dataset) // batch_size
                epoch_step_val = len(gen_val.dataset) // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                for param in self.model.backbone_manager.backbone.parameters():
                    param.requires_grad = True

                unfreeze_flag = True

            if self.config.use_warmup:
                set_optimizer_lr_with_warmup(
                    self.optimizer, lr_scheduler_func, epoch,
                    self.config.warmup_epochs, self.config.warmup_lr, Init_lr_fit
                )
            else:
                lr_scheduler_func(self.optimizer, epoch)
            
            fusion_weight = get_fusion_weight(
                epoch, 
                self.config.feature_fusion_start_epoch, 
                self.config.feature_fusion_warmup
            ) if self.config.use_feature_engineering else 0.0

            fit_one_epoch_with_features(
                model_train, self.model, self.loss_history, self.eval_callback, self.optimizer, 
                self.config.ema_train, None, epoch, epoch_step, epoch_step_val, gen, gen_val, 
                self.config.unfreeze_epoch, self.config.cuda, self.config.cls_weights, self.config.aux_branch, self.config.num_classes, 
                self.config.fp16, scaler, self.config.save_period, self.experiment_manager.get_experiment_weight_dir(), 
                self.config.distributed, self.config.local_rank,
                use_feature_engineering=self.config.use_feature_engineering,
                fusion_weight=fusion_weight,
                config=self.config
            )
            
            current_epoch = epoch + 1
            if self.eval_callback is not None and current_epoch % self.config.eval_period == 0 and current_epoch > 1:
                print(f"\n正在进行第 {current_epoch} 个epoch的评估...")
                try:
                    self.eval_callback.on_epoch_end(current_epoch, model_train)
                    print("评估完成")
                except Exception as e:
                    print(f"评估过程中出现错误: {e}")
                    import traceback
                    traceback.print_exc()

    def train(self):
        """主训练函数"""
        self.config.validate_config()
        
        print(f"\n{'='*80}")
        print(f"配置调试信息")
        print(f"{'='*80}")
        print(f"use_contrastive_pretrain: {self.config.use_contrastive_pretrain}")
        print(f"dice_loss: {self.config.dice_loss}")
        print(f"focal_loss: {self.config.focal_loss}")

        if self.config.dice_loss and self.config.focal_loss:
            print(f"改进版参数: alpha={self.config.improved_alpha}, gamma={self.config.improved_gamma}")
        print(f"{'='*80}\n")
        
        if self.config.local_rank == 0:
            self.config.check_and_download_weights()
        
        set_seed(42)
        
        print("正在检查数据集结构...")
        if not check_dataset_structure(self.config.vocdevkit_path):
            print("错误: 数据集结构不完整或没有匹配的图像-标签对！")
            exit(1)
        
        self._setup_model()
        
        model_train, scaler = self._setup_training()
        
        model_train = self._contrastive_pretraining(model_train)
        
        train_lines, val_lines = self._create_dataset_splits()
        
        num_train, num_val = len(train_lines), len(val_lines)
        print(f"最终训练集样本数: {num_train}")
        print(f"最终验证集样本数: {num_val}")

        if num_train == 0 or num_val == 0:
            print("错误: 数据集为空，请检查数据文件！")
            exit(1)

        if self.config.local_rank == 0:
            backbone_map = {
                0: "mobilenet", 1: "resnet50", 2: "resnet101", 3: "efficientnet",
                4: "resnet50_adv", 5: "mobilenet_small", 6: "mobilenetv2",
                7: "shufflenet_v1", 8: "shufflenet_v2", 9: "efficientnet_v2"
            }
            
            if self.config.dice_loss and self.config.focal_loss:
                loss_name = f"改进版DiceFocal(α={self.config.improved_alpha},γ={self.config.improved_gamma})"
            elif self.config.dice_loss:
                loss_name = "纯Dice损失"
            elif self.config.focal_loss:
                loss_name = "纯Focal损失"
            else:
                loss_name = "交叉熵损失"
                
            print(f"\n{'='*80}")
            print(f"实验配置")
            print(f"{'='*80}")
            print(f"  实验ID: {self.experiment_manager.experiment_id}")
            print(f"  实验目录: {self.experiment_manager.current_experiment_dir}")
            print(f"  主干网络: {backbone_map.get(self.config.backbone_type, '未知')}")
            print(f"  损失函数: {loss_name}")
            print(f"  对比学习: {'✅ 使用' if self.config.use_contrastive_pretrain else '❌ 未使用'}")
            print(f"  特征工程: {'✅ 使用' if self.config.use_feature_engineering else '❌ 未使用'}")
            
            if self.config.use_contrastive_pretrain:
                print(f"对比学习配置:")
                print(f"  对比学习学习率: {self.config.contrastive_lr}")
                print(f"  对比学习权重衰减: {self.config.contrastive_weight_decay}")
                print(f"  对比学习训练轮数: {self.config.contrastive_epochs}")
                print(f"  早停机制: {self.config.contrastive_patience} 个epoch")
            
            print(f"特征工程配置:")
            print(f"  使用特征工程: {self.config.use_feature_engineering}")
            if self.config.use_feature_engineering:
                print(f"  特征融合开始轮次: {self.config.feature_fusion_start_epoch}")
                print(f"  特征融合预热轮次: {self.config.feature_fusion_warmup}")
            
            backbone_name = self.model.backbone_manager.backbone_name
            print(f"主干网络: {backbone_name}")
            print(f"输出通道数: {self.model.backbone_manager.output_channels}")
            print(f"{'='*80}\n")
        
        if self.config.local_rank == 0:
            self.eval_callback = EvalCallback(
                self.model, self.config.input_shape, self.config.num_classes, val_lines, 
                self.config.vocdevkit_path, self.experiment_manager.get_experiment_log_dir(), self.config.cuda,
                eval_flag=self.config.eval_flag, period=self.config.eval_period
            )
        else:
            self.eval_callback = None
        
        gen, gen_val, epoch_step, epoch_step_val, lr_scheduler_func, batch_size, Init_lr_fit, Min_lr_fit = self._setup_data_loaders(train_lines, val_lines)
        
        self._training_loop(
            model_train, gen, gen_val, epoch_step, epoch_step_val, 
            lr_scheduler_func, batch_size, scaler, Init_lr_fit, Min_lr_fit
        )


# ==================== 实验运行器 ====================
def run_experiment(config_dict=None, experiment_name=None):
    """运行单个实验"""
    config = TrainingConfig()
    
    if config_dict:
        print(f"正在更新配置...")
        for key, value in config_dict.items():
            if hasattr(config, key):
                old_value = getattr(config, key)
                setattr(config, key, value)
                print(f"  更新配置: {key} = {value} (原值: {old_value})")
            else:
                print(f"  警告: 配置项 {key} 不存在，跳过")
    
    exp_manager = ExperimentManager()
    
    exp_dir = exp_manager.create_experiment(config.to_dict(), experiment_name)
    
    config.save_dir = exp_dir
    
    trainer = PSPNetTrainer(config, exp_manager)
    trainer.train()
    
    return exp_dir


# ==================== 主函数 ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PSPNet训练脚本')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch'], 
                       help='运行模式: single-单个实验, batch-批量实验')
    parser.add_argument('--backbone', type=int, default=None, help='主干网络类型 (0-9)')
    parser.add_argument('--contrastive', action='store_true', default=None, help='使用对比学习')
    parser.add_argument('--no-contrastive', action='store_true', default=None, help='不使用对比学习')
    parser.add_argument('--feature', action='store_true', default=None, help='使用特征工程')
    parser.add_argument('--no-feature', action='store_true', default=None, help='不使用特征工程')
    parser.add_argument('--dice_loss', action='store_true', default=None, help='使用Dice损失')
    parser.add_argument('--no-dice_loss', action='store_true', default=None, help='不使用Dice损失')
    parser.add_argument('--focal_loss', action='store_true', default=None, help='使用Focal损失')
    parser.add_argument('--no-focal_loss', action='store_true', default=None, help='不使用Focal损失')
    parser.add_argument('--improved_alpha', type=float, default=None, help='改进版alpha参数')
    parser.add_argument('--improved_gamma', type=float, default=None, help='改进版gamma参数')
    
    args = parser.parse_args()
    
    config_dict = {}
    
    if args.backbone is not None:
        config_dict["backbone_type"] = args.backbone
    
    if args.contrastive is not None:
        config_dict["use_contrastive_pretrain"] = True
    elif args.no_contrastive is not None:
        config_dict["use_contrastive_pretrain"] = False
    
    if args.feature is not None:
        config_dict["use_feature_engineering"] = True
    elif args.no_feature is not None:
        config_dict["use_feature_engineering"] = False
    
    if args.dice_loss is not None:
        config_dict["dice_loss"] = True
    elif args.no_dice_loss is not None:
        config_dict["dice_loss"] = False
    
    if args.focal_loss is not None:
        config_dict["focal_loss"] = True
    elif args.no_focal_loss is not None:
        config_dict["focal_loss"] = False
    
    if args.improved_alpha is not None:
        config_dict["improved_alpha"] = args.improved_alpha
    
    if args.improved_gamma is not None:
        config_dict["improved_gamma"] = args.improved_gamma
    
    print(f"\n最终配置:")
    for key, value in config_dict.items():
        print(f"  {key}: {value}")
    
    if args.mode == 'single':
        run_experiment(config_dict)
        
    else:
        experiments = create_experiment_combinations()
        print(f"即将运行 {len(experiments)} 个实验...")
        
        for i, (exp_config, exp_name) in enumerate(experiments):
            print(f"\n{'='*60}")
            print(f"开始第 {i+1}/{len(experiments)} 个实验: {exp_name}")
            print(f"{'='*60}")
            
            try:
                exp_dir = run_experiment(exp_config, exp_name)
                print(f"实验完成: {exp_name}")
                print(f"结果保存在: {exp_dir}")
            except Exception as e:
                print(f"实验失败: {exp_name}, 错误: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"{'='*60}\n")


def create_experiment_combinations():
    """创建实验组合"""
    experiments = []
    
    backbones = [0, 1, 2, 3]
    
    experiment_variables = [
        {"dice_loss": True, "focal_loss": True, "improved_alpha": 0.5, "improved_gamma": 2.0},
        {"dice_loss": True, "focal_loss": False},
        {"dice_loss": False, "focal_loss": True},
        {"dice_loss": True, "focal_loss": True, "improved_alpha": 0.3, "improved_gamma": 2.5},
        {"dice_loss": True, "focal_loss": True, "improved_alpha": 0.7, "improved_gamma": 1.5},
    ]
    
    for backbone in backbones:
        for exp_vars in experiment_variables:
            exp_config = {
                "backbone_type": backbone,
            }
            exp_config.update(exp_vars)
            
            backbone_names = {0: "mobilenet", 1: "resnet50", 2: "resnet101", 3: "efficientnet"}
            
            if exp_vars["dice_loss"] and exp_vars["focal_loss"]:
                if "improved_alpha" in exp_vars:
                    loss_name = f"improved_a{exp_vars['improved_alpha']}_g{exp_vars['improved_gamma']}"
                else:
                    loss_name = "dice_focal"
            elif exp_vars["dice_loss"]:
                loss_name = "dice"
            elif exp_vars["focal_loss"]:
                loss_name = "focal"
            else:
                loss_name = "ce"
            
            exp_name = f"{backbone_names[backbone]}_{loss_name}"
            
            experiments.append((exp_config, exp_name))
    
    return experiments