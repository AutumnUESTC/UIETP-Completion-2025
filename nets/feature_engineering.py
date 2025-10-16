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

# 导入自定义模块
from nets.pspnet import PSPNet
from nets.pspnet_training import (get_lr_scheduler, set_optimizer_lr, weights_init)
from nets.feature_engineering import (FeatureEngineering, diagnose_feature_fusion)
from nets.contrastive import ContrastiveLearner
from nets.dynamic_loss import DynamicWeightedLoss
from utils.EMA import EMA
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import PSPnetDataset, pspnet_dataset_collate, UnlabeledDataset, contrastive_transform
from utils.utils import download_weights, show_config
from utils.dataloader import preprocess_input

# 导入主干网络管理器
from nets.backbone_manager import BackboneManager, UnifiedPSPNet


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
        self.backbone_type = 1  # 0:mobilenet, 1:resnet50, 2:resnet101, 3:efficientnet, 4:resnet50_adv
        self.backbone_pretrained = True
        self.downsample_factor = 8
        self.input_shape = [512, 512]
        
        # 预训练权重配置
        self.model_path = ""  # 不再使用单个模型路径
        self.backbone_weights = {
            0: "/home/wuyou/pspnet-pytorch/model_data/mobilenet_v3_large.pth",
            1: "/home/wuyou/pspnet-pytorch/model_data/pspnet_resnet50.pth", 
            2: "/home/wuyou/pspnet-pytorch/model_data/resnet101.pth",  # 需要下载
            3: "/home/wuyou/pspnet-pytorch/model_data/efficientnet_b0.pth",  # 需要下载
            4: "/home/wuyou/pspnet-pytorch/model_data/pspnet_resnet50.pth"  # 使用相同的resnet50权重
        }
        
        # 训练阶段配置
        self.init_epoch = 0
        self.freeze_epoch = 200
        self.freeze_batch_size = 4
        self.unfreeze_epoch = 1000
        self.unfreeze_batch_size = 8
        
        # 训练策略配置
        self.freeze_train = False
        self.ema_train = False
        self.hl_train = False
        self.hl_c = 0
        self.hl_a = 0.1
        self.trans_branch = 0
        self.attention = 1
        
        # 新增功能配置
        self.use_feature_engineering = True
        self.use_improved_loss = False
        self.use_contrastive_pretrain = False
        self.feature_fusion_start_epoch = 50  # 从第50轮开始特征融合
        self.feature_fusion_warmup = 20  # 20轮warmup
        
        # 优化器配置
        self.init_lr = 5e-5  # 降低学习率
        self.min_lr = self.init_lr * 0.01
        self.optimizer_type = "adam"
        self.momentum = 0.9
        self.weight_decay = 0
        self.lr_decay_type = 'cos'
        
        # 训练过程配置
        self.save_period = 10
        self.save_dir = 'logs'
        self.eval_flag = True
        self.eval_period = 10
        self.vocdevkit_path = '/home/wuyou/pspnet-pytorch/VOCdevkit/VOC2007'
        
        # 损失函数配置
        self.dice_loss = True
        self.focal_loss = True
        self.cls_weights = np.ones([self.num_classes], np.float32)
        self.aux_branch = False
        
        # 数据加载配置
        self.num_workers = 4


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
        return 0.0  # 完全不使用特征融合
    elif epoch < start_epoch + warmup_epochs:
        # 线性增加权重
        progress = (epoch - start_epoch) / warmup_epochs
        return progress * 0.8  # 最大0.8权重
    else:
        return 0.8  # 固定权重


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

        # 随机缩放和裁剪
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

        # 随机翻转
        if self.rand() < .5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 颜色增强
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
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]
        
        # 查找图像文件
        jpeg_path = os.path.join(self.dataset_path, "JPEGImages", name + ".jpg")
        if not os.path.exists(jpeg_path):
            jpeg_path = os.path.join(self.dataset_path, "JPEGImages", name + ".png")
        
        # 查找标签文件
        png_path = os.path.join(self.dataset_path, "SegmentationClass", name + ".png")
        if not os.path.exists(png_path):
            png_path = os.path.join(self.dataset_path, "SegmentationClassPNG", name + ".png")
        
        image = Image.open(jpeg_path)
        png = Image.open(png_path)
        
        # 数据增强
        image, png = self.get_random_data(image, png, self.input_shape, random=self.train)
        
        # 预处理
        image = np.transpose(preprocess_input(np.array(image, np.float32)), (2, 0, 1))
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))
        
        # 特征工程
        if self.use_feature_engineering:
            try:
                image_np = image.transpose(1, 2, 0)
                image_np = (image_np * 255).astype(np.uint8)
                
                # 使用完整的18个特征
                handcrafted_features = FeatureEngineering.extract_all_features(
                    image_np, target_size=self.input_shape
                )
                
                handcrafted_tensor = torch.from_numpy(handcrafted_features.transpose(2, 0, 1).astype('float32'))
                
                return torch.from_numpy(image), torch.from_numpy(png), torch.from_numpy(seg_labels), handcrafted_tensor
            except Exception as e:
                print(f"特征工程处理失败: {e}")
                handcrafted_tensor = torch.zeros(18, self.input_shape[0], self.input_shape[1])
                return torch.from_numpy(image), torch.from_numpy(png), torch.from_numpy(seg_labels), handcrafted_tensor
        else:
            return torch.from_numpy(image), torch.from_numpy(png), torch.from_numpy(seg_labels)


def enhanced_pspnet_dataset_collate(batch):
    """数据加载collate函数"""
    if len(batch[0]) == 4:  # 包含特征工程
        images, pngs, seg_labels, handcrafted_features = zip(*batch)
        return (torch.stack(images), torch.stack(pngs), 
                torch.stack(seg_labels), torch.stack(handcrafted_features))
    else:  # 不包含特征工程
        images, pngs, seg_labels = zip(*batch)
        return torch.stack(images), torch.stack(pngs), torch.stack(seg_labels)


# ==================== 损失函数 ====================
class ImprovedDiceFocalLoss(nn.Module):
    """改进的Dice Focal Loss"""
    def __init__(self, alpha=0.65, gamma=2.5, smooth=1e-6, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, pred, target):
        if pred.shape[1] != target.max() + 1:
            pred = F.softmax(pred, dim=1)
        
        num_classes = pred.shape[1]
        dice_loss = 0.0
        valid_classes = 0
        
        for cls in range(num_classes):
            pred_cls = pred[:, cls, ...]
            target_cls = (target == cls).float()
            
            if target_cls.sum() == 0:
                if pred_cls.sum() == 0:
                    continue
                else:
                    dice_loss += (pred_cls ** 2).mean()
                valid_classes += 1
                continue
                
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()
            
            dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss += 1 - dice_score
            valid_classes += 1
        
        if valid_classes > 0:
            dice_loss /= valid_classes
        else:
            dice_loss = torch.tensor(0.0).to(pred.device)

        ce_loss = F.cross_entropy(pred, target, reduction='none', weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        return self.alpha * dice_loss + (1 - self.alpha) * focal_loss


# ==================== 训练函数 ====================
def fit_one_epoch_with_features(model_train, model, loss_history, eval_callback, optimizer, ema_train, ema, epoch, 
                              epoch_step, epoch_step_val, gen, gen_val, unfreeze_epoch, cuda, dice_loss, focal_loss, 
                              cls_weights, aux_branch, num_classes, fp16, scaler, save_period, save_dir, distributed, local_rank,
                              use_feature_engineering=False, fusion_weight=1.0, config=None):
    """支持特征工程的训练函数"""
    
    total_loss = 0
    # 修复：正确处理DataParallel包装
    actual_model_train = model_train.module if hasattr(model_train, 'module') else model_train

    # 设置特征融合权重
    if hasattr(actual_model_train, 'set_fusion_weight'):
        actual_model_train.set_fusion_weight(fusion_weight)

    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{unfreeze_epoch}', postfix=dict, mininterval=0.3) as pbar:
        model_train.train()
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
                
            try:
                if use_feature_engineering and len(batch) == 4:
                    images, targets, seg_labels, handcrafted_features = batch
                    if cuda:
                        images = images.cuda(local_rank, non_blocking=True)
                        targets = targets.cuda(local_rank, non_blocking=True).long()
                        seg_labels = seg_labels.cuda(local_rank, non_blocking=True)
                        handcrafted_features = handcrafted_features.cuda(local_rank, non_blocking=True)
                    
                    # 暂时禁用诊断，直到特征融合完全正常工作
                    # if epoch == config.init_epoch and iteration == 0 and local_rank == 0:
                    #     diagnose_feature_fusion(model_train, batch, torch.device('cuda' if cuda else 'cpu'))
                    
                    outputs = model_train(images, handcrafted_features)
                else:
                    images, targets, seg_labels = batch
                    if cuda:
                        images = images.cuda(local_rank, non_blocking=True)
                        targets = targets.cuda(local_rank, non_blocking=True).long()
                        seg_labels = seg_labels.cuda(local_rank, non_blocking=True)
                    
                    outputs = model_train(images)

                # 清零梯度并计算损失
                optimizer.zero_grad()
                
                if not fp16:
                    loss = F.cross_entropy(outputs, targets)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.amp.autocast('cuda'):
                        loss = F.cross_entropy(outputs, targets)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                total_loss += loss.item()
                
                # 监控特征融合情况
                postfix_dict = {'total_loss': total_loss / (iteration + 1), 'lr': get_lr(optimizer)}
                if fusion_weight > 0 and use_feature_engineering:
                    postfix_dict['fusion_weight'] = fusion_weight
                
                pbar.set_postfix(**postfix_dict)
                pbar.update(1)
                
                # 每50个iteration打印一次特征统计
                if use_feature_engineering and iteration % 50 == 0 and len(batch) == 4:
                    images, targets, seg_labels, handcrafted_features = batch
                    if cuda:
                        handcrafted_features = handcrafted_features.cuda(local_rank, non_blocking=True)
                    
                    feat_mean = handcrafted_features.mean().item()
                    feat_std = handcrafted_features.std().item()
                    print(f"  手工特征 - 均值: {feat_mean:.4f}, 标准差: {feat_std:.4f}")
                
            except Exception as e:
                print(f"训练过程中出错: {e}")
                import traceback
                traceback.print_exc()
                continue

    print('Finish Train')
    
    # 验证阶段
    print('Start Validation')
    model_train.eval()
    total_val_loss, val_iterations = 0, 0
    
    with torch.no_grad():
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
                
            try:
                if use_feature_engineering and len(batch) == 4:
                    images, targets, seg_labels, handcrafted_features = batch
                    if cuda:
                        images = images.cuda(local_rank, non_blocking=True)
                        targets = targets.cuda(local_rank, non_blocking=True).long()
                        handcrafted_features = handcrafted_features.cuda(local_rank, non_blocking=True)
                    
                    outputs = model_train(images, handcrafted_features)
                else:
                    images, targets, seg_labels = batch
                    if cuda:
                        images = images.cuda(local_rank, non_blocking=True)
                        targets = targets.cuda(local_rank, non_blocking=True).long()
                    
                    outputs = model_train(images)
                
                loss = F.cross_entropy(outputs, targets)
                total_val_loss += loss.item()
                val_iterations += 1
                
            except Exception as e:
                print(f"验证过程中出错: {e}")
                continue
    
    val_loss = total_val_loss / val_iterations if val_iterations > 0 else 0
    print('Finish Validation')
    
    # 记录损失
    train_loss = total_loss / epoch_step if epoch_step > 0 else 0
    
    if loss_history is not None:
        try:
            loss_history.append_loss(epoch + 1, train_loss, val_loss)
        except Exception as e:
            print(f"记录损失时出错: {e}")
        
    # 保存模型
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


def contrastive_pretrain(model, dataloader, optimizer, device, epochs=200, save_path=None, fp16=False):
    """对比学习预训练"""
    contrastive_learner = ContrastiveLearner(model.backbone)
    contrastive_learner.to(device)
    contrastive_learner.train()

    scaler = torch.amp.GradScaler('cuda') if fp16 else None
    start_epoch, matched_keys = 0, 0
    
    if save_path and os.path.exists(save_path):
        print(f"加载对比学习预训练权重: {save_path}")
        try:
            pretrained_dict = torch.load(save_path, map_location=device, weights_only=False)
            model_dict = contrastive_learner.backbone.state_dict()
            
            matched_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    matched_dict[k] = v
                    matched_keys += 1
            
            if matched_keys > 0:
                contrastive_learner.backbone.load_state_dict(matched_dict, strict=False)
                print(f"成功加载 {matched_keys} 个对比学习预训练参数")
                start_epoch = int(save_path.split('_')[-1].split('.')[0]) if '_epoch' in save_path else 0
            else:
                print("没有匹配的预训练参数，将从零开始训练")
        except Exception as e:
            print(f"加载对比学习权重失败: {e}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=1e-6
    )
    
    best_loss = float('inf')
    print(f"开始对比学习预训练，从第 {start_epoch} 个epoch开始，总共 {epochs} 个epoch")
    
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
        
        if save_path and (epoch + 1) % 20 == 0:
            checkpoint_path = save_path.replace('.pth', f'_epoch{epoch+1}.pth')
            torch.save(contrastive_learner.backbone.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(contrastive_learner.backbone.state_dict(), save_path)
                print(f"新的最佳模型已保存: {save_path} (Loss: {best_loss:.6f})")

    if save_path:
        torch.save(contrastive_learner.backbone.state_dict(), save_path)
        print(f"对比学习预训练权重已保存: {save_path} (最终Loss: {avg_loss:.6f})")
    
    return contrastive_learner


# ==================== 主训练类 ====================
class PSPNetTrainer:
    """PSPNet训练器类"""
    
    def __init__(self, config):
        self.config = config
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

        # 打印GPU信息
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
        """设置模型 - 使用统一的主干网络"""
        # 创建统一的PSPNet模型
        self.model = UnifiedPSPNet(
            num_classes=self.config.num_classes,
            backbone_type=self.config.backbone_type,
            downsample_factor=self.config.downsample_factor,
            use_feature_engineering=self.config.use_feature_engineering,
            pretrained=False  # 设为False，因为我们手动加载权重
        )
        
        # 加载主干网络预训练权重
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
        # 记录Loss
        if self.config.local_rank == 0:
            time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
            log_dir = os.path.join(self.config.save_dir, "loss_" + str(time_str))
            self.loss_history = LossHistory(log_dir, self.model, input_shape=self.config.input_shape)
        else:
            self.loss_history = None

        # 混合精度训练
        scaler = torch.amp.GradScaler('cuda') if self.config.fp16 else None

        model_train = self.model.train()
        
        # 多卡同步Bn
        if self.config.sync_bn and torch.cuda.device_count() > 1 and self.config.distributed:
            model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)

        # GPU设置
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
        contrastive_pretrain_path = os.path.join(self.config.save_dir, 'contrastive_pretrained_backbone.pth')
        
        if not self.config.use_contrastive_pretrain:
            print("配置为不使用对比学习预训练，跳过此步骤")
            return model_train

        is_pretrained, status_message = check_contrastive_pretrain_status(contrastive_pretrain_path)
        print(f"对比学习预训练状态: {status_message}")
        
        if is_pretrained:
            print("检测到已完成的对比学习预训练，直接加载权重...")
            try:
                pretrained_dict = torch.load(contrastive_pretrain_path, map_location=self.device)
                model_dict = self.model.backbone_manager.backbone.state_dict()
                
                matched_dict = {}
                for k, v in pretrained_dict.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        matched_dict[k] = v
                
                if matched_dict:
                    self.model.backbone_manager.backbone.load_state_dict(matched_dict, strict=False)
                    print(f"成功加载 {len(matched_dict)} 个对比学习预训练参数")
                else:
                    print("没有匹配的预训练参数，将使用当前权重")
            except Exception as e:
                print(f"加载对比学习权重失败: {e}")
        else:
            print("开始对比学习预训练...")
            
            contrastive_dataset_path = os.path.join(self.config.vocdevkit_path, "JPEGImages")
            if os.path.exists(contrastive_dataset_path):
                unlabeled_dataset = UnlabeledDataset(contrastive_dataset_path, transform=contrastive_transform)
                unlabeled_loader = DataLoader(
                    unlabeled_dataset, batch_size=8, shuffle=True, num_workers=4, 
                    pin_memory=True, persistent_workers=True, prefetch_factor=2
                )

                contrastive_optimizer = torch.optim.AdamW(
                    self.model.backbone_manager.backbone.parameters(), 
                    lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.98)
                )
                
                contrastive_learner = contrastive_pretrain(
                    self.model, unlabeled_loader, contrastive_optimizer, self.device, 
                    epochs=200, save_path=contrastive_pretrain_path, fp16=self.config.fp16
                )
                print("对比学习预训练完成")
                
                self.model.backbone_manager.backbone.load_state_dict(contrastive_learner.backbone.state_dict())
            else:
                print(f"对比学习数据集路径不存在: {contrastive_dataset_path}")
                print("跳过对比学习预训练")
        
        return model_train

    def _setup_data_loaders(self, train_lines, val_lines):
        """设置数据加载器"""
        # 自适应调整学习率
        nbs = 16
        lr_limit_max = 5e-4 if self.config.optimizer_type == 'adam' else 1e-1
        lr_limit_min = 3e-4 if self.config.optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(self.config.unfreeze_batch_size / nbs * self.config.init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(self.config.unfreeze_batch_size / nbs * self.config.min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # 优化器选择
        self.optimizer = {
            'adam': optim.Adam(self.model.parameters(), Init_lr_fit, betas=(self.config.momentum, 0.999), weight_decay=self.config.weight_decay),
            'sgd': optim.SGD(self.model.parameters(), Init_lr_fit, momentum=self.config.momentum, nesterov=True, weight_decay=self.config.weight_decay)
        }[self.config.optimizer_type]

        # 学习率调度
        lr_scheduler_func = get_lr_scheduler(self.config.lr_decay_type, Init_lr_fit, Min_lr_fit, self.config.unfreeze_epoch)

        # 数据加载器
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
            train_dataset, shuffle=True, batch_size=batch_size, num_workers=self.config.num_workers, 
            pin_memory=True, drop_last=True, collate_fn=enhanced_pspnet_dataset_collate,
            persistent_workers=True, prefetch_factor=2
        )
        gen_val = DataLoader(
            val_dataset, shuffle=True, batch_size=batch_size, num_workers=self.config.num_workers, 
            pin_memory=True, drop_last=True, collate_fn=enhanced_pspnet_dataset_collate,
            persistent_workers=True, prefetch_factor=2
        )

        return gen, gen_val, epoch_step, epoch_step_val, lr_scheduler_func, batch_size

    def _training_loop(self, model_train, gen, gen_val, epoch_step, epoch_step_val, 
                      lr_scheduler_func, batch_size, scaler):
        """训练循环"""
        unfreeze_flag = False

        # 冻结训练
        if self.config.freeze_train:
            for param in self.model.backbone_manager.backbone.parameters():
                param.requires_grad = False

        for epoch in range(self.config.init_epoch, self.config.unfreeze_epoch):
            # 解冻训练
            if epoch >= self.config.freeze_epoch and not unfreeze_flag and self.config.freeze_train:
                batch_size = self.config.unfreeze_batch_size
                epoch_step = len(gen.dataset) // batch_size
                epoch_step_val = len(gen_val.dataset) // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                for param in self.model.backbone_manager.backbone.parameters():
                    param.requires_grad = True

                unfreeze_flag = True

            # 设置学习率
            set_optimizer_lr(self.optimizer, lr_scheduler_func, epoch)
            
            # 计算特征融合权重
            fusion_weight = get_fusion_weight(
                epoch, 
                self.config.feature_fusion_start_epoch, 
                self.config.feature_fusion_warmup
            ) if self.config.use_feature_engineering else 0.0

            # 训练一个epoch
            fit_one_epoch_with_features(
                model_train, self.model, self.loss_history, self.eval_callback, self.optimizer, 
                self.config.ema_train, None, epoch, epoch_step, epoch_step_val, gen, gen_val, 
                self.config.unfreeze_epoch, self.config.cuda, self.config.dice_loss, self.config.focal_loss,
                self.config.cls_weights, self.config.aux_branch, self.config.num_classes, 
                self.config.fp16, scaler, self.config.save_period, self.config.save_dir, 
                self.config.distributed, self.config.local_rank,
                use_feature_engineering=self.config.use_feature_engineering,
                fusion_weight=fusion_weight,
                config=self.config
            )
            
            # 评估
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
        # 设置随机种子
        set_seed(42)
        
        # 检查数据集
        print("正在检查数据集结构...")
        if not check_dataset_structure(self.config.vocdevkit_path):
            print("错误: 数据集结构不完整或没有匹配的图像-标签对！")
            exit(1)
        
        # 设置模型
        self._setup_model()
        
        # 设置训练环境
        model_train, scaler = self._setup_training()
        
        # 对比学习预训练
        model_train = self._contrastive_pretraining(model_train)
        
        # 创建数据集
        train_lines, val_lines = self._create_dataset_splits()
        
        num_train, num_val = len(train_lines), len(val_lines)
        print(f"最终训练集样本数: {num_train}")
        print(f"最终验证集样本数: {num_val}")

        if num_train == 0 or num_val == 0:
            print("错误: 数据集为空，请检查数据文件！")
            exit(1)

        # 显示配置
        if self.config.local_rank == 0:
            show_config(
                num_classes=self.config.num_classes, backbone=self.config.backbone_type, model_path="使用统一主干网络", 
                input_shape=self.config.input_shape, Init_Epoch=self.config.init_epoch, Freeze_Epoch=self.config.freeze_epoch, 
                UnFreeze_Epoch=self.config.unfreeze_epoch, Freeze_batch_size=self.config.freeze_batch_size, 
                Unfreeze_batch_size=self.config.unfreeze_batch_size, Freeze_Train=self.config.freeze_train, 
                Init_lr=self.config.init_lr, Min_lr=self.config.min_lr, optimizer_type=self.config.optimizer_type, 
                momentum=self.config.momentum, lr_decay_type=self.config.lr_decay_type, HL_Train=self.config.hl_train, 
                HL_a=self.config.hl_a, HL_c=self.config.hl_c, EMA_Train=self.config.ema_train, 
                Trans_branch=self.config.trans_branch, attention=self.config.attention, save_period=self.config.save_period, 
                save_dir=self.config.save_dir, num_workers=self.config.num_workers, num_train=num_train, num_val=num_val
            )
            
            # 显示特征工程配置
            print(f"特征工程配置:")
            print(f"  使用特征工程: {self.config.use_feature_engineering}")
            print(f"  特征融合开始轮次: {self.config.feature_fusion_start_epoch}")
            print(f"  特征融合预热轮次: {self.config.feature_fusion_warmup}")
            print(f"  使用完整18个特征")
            
            # 显示主干网络信息
            backbone_name = self.model.backbone_manager.backbone_name
            print(f"主干网络: {backbone_name}")
            print(f"输出通道数: {self.model.backbone_manager.output_channels}")
        
        # 设置评估回调
        if self.config.local_rank == 0:
            self.eval_callback = EvalCallback(
                self.model, self.config.input_shape, self.config.num_classes, val_lines, 
                self.config.vocdevkit_path, self.config.save_dir, self.config.cuda,
                eval_flag=self.config.eval_flag, period=self.config.eval_period
            )
        else:
            self.eval_callback = None
        
        # 设置数据加载器
        gen, gen_val, epoch_step, epoch_step_val, lr_scheduler_func, batch_size = self._setup_data_loaders(train_lines, val_lines)
        
        # 训练循环
        self._training_loop(
            model_train, gen, gen_val, epoch_step, epoch_step_val, 
            lr_scheduler_func, batch_size, scaler
        )


# ==================== 主函数 ====================
if __name__ == "__main__":
    # 创建配置并开始训练
    config = TrainingConfig()
    trainer = PSPNetTrainer(config)
    trainer.train()