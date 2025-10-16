import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加设备定义
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ContrastiveHead(nn.Module):
    """对比学习投影头"""
    def __init__(self, in_features, out_features=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, out_features)
        )
    
    def forward(self, x):
        return F.normalize(self.mlp(x), dim=1)

class ContrastiveLearner(nn.Module):
    """对比学习模块"""
    def __init__(self, backbone, feature_dim=128, temperature=0.5, device=None):
        super().__init__()
        self.backbone = backbone
        self.temperature = temperature
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 获取骨干网络的特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            features = self.backbone(dummy_input)
            
            # 处理backbone返回的可能是元组或单个张量的情况
            if isinstance(features, tuple):
                # 如果是元组，取最后一个特征图（通常是最高层的特征）
                features = features[-1]
            
            # 获取特征维度
            if len(features.shape) == 4:  # [batch, channels, height, width]
                # 使用全局平均池化获取特征向量
                features = torch.mean(features, dim=[2, 3])
                in_features = features.shape[1]
            else:
                in_features = features.shape[-1]  # 如果已经是展平的特征
        
        # 对比学习投影头
        self.projector = ContrastiveHead(in_features, feature_dim)
    
    def forward(self, x1, x2):
        # 提取特征
        features1 = self.backbone(x1)
        features2 = self.backbone(x2)
        
        # 处理可能的元组输出
        if isinstance(features1, tuple):
            features1 = features1[-1]  # 取最后一个特征图
        if isinstance(features2, tuple):
            features2 = features2[-1]  # 取最后一个特征图
            
        # 全局平均池化 [B, C, H, W] -> [B, C]
        if len(features1.shape) == 4:
            features1 = torch.mean(features1, dim=[2, 3])
        if len(features2.shape) == 4:
            features2 = torch.mean(features2, dim=[2, 3])
        
        # 投影到对比学习空间
        z1 = self.projector(features1)
        z2 = self.projector(features2)
        
        return z1, z2
    
    def contrastive_loss(self, z1, z2):
        batch_size = z1.size(0)
        
        # 合并特征
        features = torch.cat([z1, z2], dim=0)
        
        # 计算相似度矩阵
        similarity_matrix = F.cosine_similarity(
            features.unsqueeze(1), features.unsqueeze(0), dim=2)
        
        # 创建标签
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(z1.device)
        
        # 排除对角线
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z1.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        # 选择正负样本
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z1.device)
        
        logits = logits / self.temperature
        return F.cross_entropy(logits, labels)