import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input

import random
from torchvision import transforms
from PIL import ImageFilter

# ==================== 对比学习相关功能 ====================

class GaussianBlur(object):
    """高斯模糊变换"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

# 对比学习数据增强变换
contrastive_transform = transforms.Compose([
    transforms.RandomResizedCrop(512, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class UnlabeledDataset(Dataset):
    """未标注数据集，用于对比学习预训练"""
    
    def __init__(self, image_dir, transform=None, target_size=(512, 512)):
        """
        参数:
            image_dir: 未标注图像的目录路径
            transform: 用于数据增强的变换
            target_size: 目标图像尺寸
        """
        self.image_paths = []
        self.target_size = target_size
        
        # 检查目录是否存在
        if os.path.exists(image_dir):
            for f in os.listdir(image_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(image_dir, f))
        
        self.transform = transform
        print(f"找到 {len(self.image_paths)} 张未标注图像")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            
            # 调整图像尺寸
            if image.size != self.target_size:
                image = image.resize(self.target_size, Image.BICUBIC)
            
            if self.transform:
                # 对比学习：对同一图像生成两个增强视图
                view1 = self.transform(image)
                view2 = self.transform(image)
                return view1, view2
            else:
                # 如果没有变换，直接返回原图（转换为tensor）
                transform_base = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                image_tensor = transform_base(image)
                return image_tensor, image_tensor
                
        except Exception as e:
            print(f"加载图像 {img_path} 失败: {e}")
            # 返回空白图像作为备用
            blank = torch.zeros(3, self.target_size[0], self.target_size[1])
            return blank, blank

# ==================== 特征工程相关功能 ====================

class FeatureEngineeredDataset(Dataset):
    """特征工程数据集，集成手工特征"""
    
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(FeatureEngineeredDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path
        self.pspnet_dataset = PSPnetDataset(annotation_lines, input_shape, num_classes, train, dataset_path)
        
    def __len__(self):
        return self.length
        
    def extract_handcrafted_features(self, image):
        """提取手工特征"""
        try:
            # 将PIL图像转换为numpy数组
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image
                
            # 确保是3通道
            if len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                
            # 调整尺寸
            if image_np.shape[0] != self.input_shape[0] or image_np.shape[1] != self.input_shape[1]:
                image_np = cv2.resize(image_np, (self.input_shape[1], self.input_shape[0]))
                
            # 简单的特征提取示例
            features = []
            
            # 1. 灰度图
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            features.append(gray)
            
            # 2. 边缘特征 (Sobel)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
            sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 1, cv2.NORM_MINMAX)
            features.append(sobel_magnitude)
            
            # 3. HSV颜色空间
            hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            features.extend([h.astype(np.float32)/179.0, s.astype(np.float32)/255.0, v.astype(np.float32)/255.0])
            
            # 堆叠所有特征
            handcrafted_features = np.stack(features, axis=2)  # [H, W, 5]
            
            return handcrafted_features
            
        except Exception as e:
            print(f"特征提取失败: {e}")
            # 返回零特征
            return np.zeros((self.input_shape[0], self.input_shape[1], 5), dtype=np.float32)
    
    def __getitem__(self, index):
        # 获取基础数据
        jpg, png, seg_labels = self.pspnet_dataset[index]
        
        # 提取手工特征
        # 注意：这里需要将jpg转换回PIL图像进行特征提取
        jpg_np = np.transpose(jpg, [1, 2, 0])  # CHW -> HWC
        jpg_np = (jpg_np * 255).astype(np.uint8)
        jpg_pil = Image.fromarray(jpg_np)
        
        handcrafted_features = self.extract_handcrafted_features(jpg_pil)
        handcrafted_features = np.transpose(handcrafted_features, [2, 0, 1])  # HWC -> CHW
        
        return jpg, torch.from_numpy(handcrafted_features).float(), png, seg_labels

# 特征工程数据集的collate函数
def feature_engineered_collate(batch):
    images = []
    handcrafted_features = []
    pngs = []
    seg_labels = []
    
    for img, hf, png, labels in batch:
        images.append(img)
        handcrafted_features.append(hf)
        pngs.append(png)
        seg_labels.append(labels)
        
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    handcrafted_features = torch.stack(handcrafted_features)
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    
    return images, handcrafted_features, pngs, seg_labels

class PSPnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(PSPnetDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, "JPEGImages"), name + ".jpg"))
        png         = Image.open(os.path.join(os.path.join(self.dataset_path, "SegmentationClassPNG"), name + ".png"))
        #-------------------------------#
        #   数据增强
        #-------------------------------#
        jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)

        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        png         = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        #-------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        #-------------------------------------------------------#
        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_shape[1]), int(self.input_shape[0]), self.num_classes+1))

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image   = cvtColor(image)
        label   = Image.fromarray(np.array(label))
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            iw, ih  = image.size
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data      = np.array(image, np.uint8)

        #------------------------------------------#
        #   高斯模糊
        #------------------------------------------#
        blur = self.rand() < 0.25
        if blur: 
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        #------------------------------------------#
        #   旋转
        #------------------------------------------#
        rotate = self.rand() < 0.25
        if rotate: 
            center      = (w // 2, h // 2)
            rotation    = np.random.randint(-10, 11)
            M           = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data  = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128,128,128))
            label       = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label

# DataLoader中collate_fn使用
def pspnet_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels
