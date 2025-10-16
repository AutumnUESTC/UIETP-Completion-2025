import torch
import torch.nn as nn

class DynamicWeightedLoss(nn.Module):
    """动态类别权重损失函数"""
    def __init__(self, base_loss_func, num_classes, alpha=0.9):
        super().__init__()
        self.base_loss_func = base_loss_func
        self.num_classes = num_classes
        self.alpha = alpha
        self.register_buffer('class_weights', torch.ones(num_classes))
        self.register_buffer('class_performance', torch.zeros(num_classes))
    
    def update_weights(self, class_iou):
        """根据各类别IoU更新权重"""
        # IoU越低，权重越高（给予更多关注）
        new_weights = 1.0 - torch.tensor(class_iou, device=self.class_weights.device)
        # 平滑更新
        self.class_weights = self.alpha * self.class_weights + (1 - self.alpha) * new_weights
        # 归一化
        self.class_weights = self.class_weights / self.class_weights.mean()
        
        print(f"更新类别权重: {self.class_weights}")
    
    def forward(self, inputs, targets):
        # 为每个样本分配权重
        weights = self.class_weights[targets]
        
        # 计算基础损失
        loss = self.base_loss_func(inputs, targets)
        
        # 应用权重
        weighted_loss = loss * weights
        
        return weighted_loss.mean()