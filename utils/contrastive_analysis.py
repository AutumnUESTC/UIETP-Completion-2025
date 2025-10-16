# utils/contrastive_analysis.py
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免GUI问题
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('/home/wuyou/pspnet-pytorch')

class ContrastiveAnalysisFixed:
    """修复版的对比学习分析工具"""
    
    def __init__(self, device='cuda', output_dir='contrastive_analysis_results'):
        self.device = device
        self.output_dir = output_dir
        self.results = {}
        self.best_epoch = None
        self.best_metric = None
        self.best_reason = ""
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        print(f"结果将保存到: {os.path.abspath(output_dir)}")
    
    def collect_contrastive_weights(self, logs_dir):
        """收集所有对比学习权重文件"""
        print("收集对比学习权重文件...")
        
        contrastive_files = []
        for root, dirs, files in os.walk(logs_dir):
            for file in files:
                if file.startswith('contrastive_pretrained_backbone') and file.endswith('.pth'):
                    # 提取epoch数字
                    file_path = os.path.join(root, file)
                    if 'epoch' in file:
                        epoch_str = file.split('epoch')[-1].replace('.pth', '')
                        if epoch_str.isdigit():
                            epoch = int(epoch_str)
                        else:
                            epoch = 0
                    else:
                        epoch = 0
                    
                    contrastive_files.append((epoch, file_path))
        
        # 按epoch排序
        contrastive_files.sort()
        print(f"找到 {len(contrastive_files)} 个对比学习权重文件")
        
        return contrastive_files
    
    def analyze_weight_statistics(self, weight_path):
        """分析权重统计信息"""
        try:
            checkpoint = torch.load(weight_path, map_location='cpu')
            
            if not isinstance(checkpoint, dict):
                print(f"  警告: {os.path.basename(weight_path)} 不是字典格式")
                return None
            
            # 收集所有权重值
            all_weights = []
            layer_stats = {}
            
            for key, tensor in checkpoint.items():
                if tensor.dtype in [torch.float32, torch.float16]:
                    weights_flat = tensor.flatten().cpu().numpy()
                    all_weights.append(weights_flat)
                    
                    # 记录每层统计
                    layer_stats[key] = {
                        'mean': float(weights_flat.mean()),
                        'std': float(weights_flat.std()),
                        'shape': list(tensor.shape)
                    }
            
            if all_weights:
                all_weights = np.concatenate(all_weights)
                stats = {
                    'mean': float(all_weights.mean()),
                    'std': float(all_weights.std()),
                    'min': float(all_weights.min()),
                    'max': float(all_weights.max()),
                    'num_params': len(all_weights),
                    'num_layers': len(layer_stats),
                    'layer_stats': layer_stats
                }
                return stats
            else:
                print(f"  警告: {os.path.basename(weight_path)} 没有有效的权重参数")
                return None
                
        except Exception as e:
            print(f"  分析 {os.path.basename(weight_path)} 失败: {e}")
            return None
    
    def test_model_loading(self, weight_path, backbone_type=1):
        """测试模型加载能力"""
        try:
            from nets.backbone_manager import UnifiedPSPNet
            
            model = UnifiedPSPNet(
                num_classes=6,
                backbone_type=backbone_type,
                downsample_factor=8,
                pretrained=False
            )
            
            checkpoint = torch.load(weight_path, map_location='cpu')
            
            # 策略2: 添加backbone前缀
            model_dict = model.state_dict()
            matched_keys = []
            
            for k, v in checkpoint.items():
                new_key = f"backbone_manager.backbone.{k}"
                if new_key in model_dict and model_dict[new_key].shape == v.shape:
                    model_dict[new_key] = v
                    matched_keys.append(new_key)
            
            if matched_keys:
                model.load_state_dict(model_dict, strict=False)
                return True, len(matched_keys), model
            else:
                return False, 0, None
                
        except Exception as e:
            return False, 0, None
    
    def run_comprehensive_analysis(self, logs_dir):
        """运行全面分析"""
        print("=" * 60)
        print("对比学习权重全面分析")
        print("=" * 60)
        
        # 1. 收集权重文件
        contrastive_files = self.collect_contrastive_weights(logs_dir)
        
        if not contrastive_files:
            print("未找到对比学习权重文件!")
            return
        
        # 2. 分析每个权重文件
        print("\n分析权重文件统计信息:")
        analysis_results = {}
        
        for epoch, file_path in tqdm(contrastive_files, desc="分析权重文件"):
            print(f"  Epoch {epoch}: {os.path.basename(file_path)}")
            
            stats = self.analyze_weight_statistics(file_path)
            if stats is not None:
                # 测试模型加载
                can_load, num_loaded, model = self.test_model_loading(file_path)
                
                analysis_results[epoch] = {
                    'file_path': file_path,
                    'stats': stats,
                    'can_load': can_load,
                    'num_loaded': num_loaded,
                    'model': model
                }
                
                # 使用ASCII字符替代特殊符号
                status = "[OK] 可加载" if can_load else "[FAIL] 不可加载"
                print(f"    {status}, 加载参数: {num_loaded}, 均值: {stats['mean']:.6f}, 标准差: {stats['std']:.6f}")
        
        self.results = analysis_results
        
        # 3. 选择最佳权重
        self.select_best_weight()
        
        # 4. 生成分析报告
        self.generate_analysis_report()
        
        # 5. 可视化结果
        self.create_visualizations()
        
        print(f"\n分析完成! 结果保存在: {self.output_dir}")
    
    def select_best_weight(self):
        """选择最佳权重"""
        if not self.results:
            return
        
        # 筛选可加载的文件
        loadable_results = {epoch: data for epoch, data in self.results.items() if data['can_load']}
        
        if not loadable_results:
            print("没有可加载的权重文件!")
            return
        
        # 多种选择策略
        candidates = []
        
        for epoch, data in loadable_results.items():
            stats = data['stats']
            std = stats['std']
            mean = stats['mean']
            num_params = stats['num_params']
            
            # 策略1: 标准差接近0.01 (适中的激活)
            std_score = 1.0 / (1.0 + abs(std - 0.01))
            
            # 策略2: 均值接近0 (良好的初始化)
            mean_score = 1.0 / (1.0 + abs(mean))
            
            # 策略3: 参数数量多 (完整的模型)
            param_score = min(1.0, num_params / 1000000)  # 假设100万参数为基准
            
            # 策略4: epoch数较高 (训练更充分)
            epoch_score = min(1.0, epoch / 500)  # 假设500 epoch为基准
            
            # 综合分数
            total_score = std_score * 0.4 + mean_score * 0.3 + param_score * 0.2 + epoch_score * 0.1
            
            candidates.append((epoch, data, total_score, std_score, mean_score))
        
        # 按综合分数排序
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        if candidates:
            best_epoch, best_data, best_score, std_score, mean_score = candidates[0]
            self.best_epoch = best_epoch
            self.best_metric = best_score
            self.best_reason = f"综合评分最高 (std_score: {std_score:.3f}, mean_score: {mean_score:.3f})"
            
            print(f"\n最佳权重选择: Epoch {best_epoch}")
            print(f"  文件: {os.path.basename(best_data['file_path'])}")
            print(f"  综合评分: {best_score:.3f}")
            print(f"  标准差: {best_data['stats']['std']:.6f}")
            print(f"  均值: {best_data['stats']['mean']:.6f}")
            print(f"  加载参数: {best_data['num_loaded']}")
    
    def generate_analysis_report(self):
        """生成分析报告"""
        report_path = os.path.join(self.output_dir, 'analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("对比学习预训练权重分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"分析时间: {np.datetime64('now')}\n")
            f.write(f"总权重文件数: {len(self.results)}\n\n")
            
            # 可加载的文件
            loadable_files = [epoch for epoch, data in self.results.items() if data['can_load']]
            f.write(f"可成功加载的文件: {len(loadable_files)}\n")
            
            if self.best_epoch is not None:
                f.write(f"\n=== 最佳推荐 ===\n")
                best_data = self.results[self.best_epoch]
                f.write(f"Epoch {self.best_epoch}: {os.path.basename(best_data['file_path'])}\n")
                f.write(f"选择理由: {self.best_reason}\n")
                f.write(f"综合评分: {self.best_metric:.3f}\n")
                f.write(f"标准差: {best_data['stats']['std']:.6f}\n")
                f.write(f"均值: {best_data['stats']['mean']:.6f}\n")
                f.write(f"加载参数: {best_data['num_loaded']}\n")
                f.write(f"文件路径: {best_data['file_path']}\n")
            
            f.write("\n详细统计:\n")
            for epoch, data in sorted(self.results.items()):
                f.write(f"\nEpoch {epoch}: {os.path.basename(data['file_path'])}\n")
                f.write(f"  可加载: {data['can_load']}\n")
                f.write(f"  加载参数数量: {data['num_loaded']}\n")
                f.write(f"  权重均值: {data['stats']['mean']:.6f}\n")
                f.write(f"  权重标准差: {data['stats']['std']:.6f}\n")
                f.write(f"  权重范围: [{data['stats']['min']:.6f}, {data['stats']['max']:.6f}]\n")
                f.write(f"  总参数数量: {data['stats']['num_params']}\n")
                f.write(f"  层数: {data['stats']['num_layers']}\n")
        
        print(f"分析报告已保存: {report_path}")
    
    def create_visualizations(self):
        """创建可视化图表"""
        if not self.results:
            print("没有数据可可视化")
            return
        
        epochs = list(self.results.keys())
        means = [self.results[epoch]['stats']['mean'] for epoch in epochs]
        stds = [self.results[epoch]['stats']['std'] for epoch in epochs]
        num_loaded = [self.results[epoch]['num_loaded'] for epoch in epochs]
        
        # 设置matplotlib使用支持中文的字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 权重统计变化图
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(epochs, means, 'bo-', linewidth=2, markersize=6)
        plt.xlabel('Epoch')
        plt.ylabel('Weight Mean')
        plt.title('Contrastive Learning Weight Mean Change')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(epochs, stds, 'ro-', linewidth=2, markersize=6)
        plt.xlabel('Epoch')
        plt.ylabel('Weight Std')
        plt.title('Contrastive Learning Weight Std Change')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.bar(epochs, num_loaded, color='green', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Successfully Loaded Parameters')
        plt.title('Number of Successfully Loaded Parameters')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        # 标记可加载的文件
        colors = ['green' if self.results[epoch]['can_load'] else 'red' for epoch in epochs]
        plt.scatter(means, stds, c=colors, s=100, alpha=0.7)
        plt.xlabel('Weight Mean')
        plt.ylabel('Weight Std')
        plt.title('Weight Distribution Scatter\n(Green: Loadable, Red: Not Loadable)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'contrastive_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 创建推荐图表 - 修复拥堵问题
        self.create_recommendation_chart()
        
        print(f"可视化图表已保存到: {self.output_dir}")
    
    def create_recommendation_chart(self):
        """创建推荐图表 - 修复版本"""
        if not self.results:
            return
        
        # 筛选可加载的文件
        loadable_results = {epoch: data for epoch, data in self.results.items() if data['can_load']}
        
        if not loadable_results:
            return
        
        # 计算质量分数
        quality_data = []
        for epoch, data in loadable_results.items():
            stats = data['stats']
            std = stats['std']
            mean = stats['mean']
            num_params = stats['num_params']
            
            # 质量评分标准
            std_quality = 1.0 / (1.0 + abs(std - 0.01))  # 标准差接近0.01为佳
            mean_quality = 1.0 / (1.0 + abs(mean))       # 均值接近0为佳
            param_quality = min(1.0, num_params / 1000000)  # 参数完整性
            
            # 综合质量分数
            quality_score = std_quality * 0.5 + mean_quality * 0.3 + param_quality * 0.2
            
            quality_data.append({
                'epoch': epoch,
                'data': data,
                'quality_score': quality_score,
                'std': std,
                'mean': mean,
                'num_loaded': data['num_loaded']
            })
        
        # 按质量分数排序
        quality_data.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # 限制显示数量，避免拥堵
        max_display = min(15, len(quality_data))
        display_data = quality_data[:max_display]
        
        # 创建图表
        plt.figure(figsize=(14, 8))
        
        # 使用子图来分散信息
        ax1 = plt.subplot(1, 2, 1)
        
        # 质量分数条形图
        epochs_display = [f'E{item["epoch"]}' for item in display_data]
        quality_scores = [item['quality_score'] for item in display_data]
        stds_display = [item['std'] for item in display_data]
        
        # 使用颜色渐变
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(display_data)))
        
        bars = ax1.barh(range(len(display_data)), quality_scores, color=colors, alpha=0.8)
        ax1.set_yticks(range(len(display_data)))
        ax1.set_yticklabels(epochs_display)
        ax1.set_xlabel('Quality Score')
        ax1.set_title('Contrastive Weight Quality Ranking\n(Higher is Better)')
        
        # 在条形上添加数值
        for i, (bar, score, std) in enumerate(zip(bars, quality_scores, stds_display)):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', ha='left', va='center', fontsize=9)
        
        # 第二个子图：详细信息
        ax2 = plt.subplot(1, 2, 2)
        
        # 创建详细信息的文本
        info_text = "Top Recommendations:\n\n"
        for i, item in enumerate(display_data[:8]):  # 显示前8个的详细信息
            info_text += f"{i+1}. Epoch {item['epoch']}:\n"
            info_text += f"   Score: {item['quality_score']:.3f}\n"
            info_text += f"   Std: {item['std']:.4f}\n"
            info_text += f"   Params: {item['num_loaded']}\n\n"
        
        ax2.text(0.1, 0.95, info_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        ax2.axis('off')
        ax2.set_title('Detailed Recommendations')
        
        # 标记最佳选择
        if self.best_epoch is not None:
            best_idx = None
            for i, item in enumerate(display_data):
                if item['epoch'] == self.best_epoch:
                    best_idx = i
                    break
            
            if best_idx is not None:
                ax1.text(0.02, best_idx, '★', transform=ax1.get_yaxis_transform(), 
                        fontsize=20, color='red', weight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'weight_recommendations.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 创建epoch趋势图
        self.create_epoch_trend_chart(loadable_results)
        
        # 保存推荐结果
        self.save_recommendations(quality_data)
    
    def create_epoch_trend_chart(self, loadable_results):
        """创建epoch趋势图"""
        epochs = sorted(loadable_results.keys())
        stds = [loadable_results[epoch]['stats']['std'] for epoch in epochs]
        means = [loadable_results[epoch]['stats']['mean'] for epoch in epochs]
        num_loaded = [loadable_results[epoch]['num_loaded'] for epoch in epochs]
        
        plt.figure(figsize=(12, 8))
        
        # 创建三个子图
        plt.subplot(3, 1, 1)
        plt.plot(epochs, stds, 'g-o', linewidth=2, markersize=4)
        plt.ylabel('Weight Std')
        plt.title('Training Progress Over Epochs')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 2)
        plt.plot(epochs, means, 'b-o', linewidth=2, markersize=4)
        plt.ylabel('Weight Mean')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 3)
        plt.plot(epochs, num_loaded, 'r-o', linewidth=2, markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('Loaded Params')
        plt.grid(True, alpha=0.3)
        
        # 标记最佳epoch
        if self.best_epoch in epochs:
            best_idx = epochs.index(self.best_epoch)
            plt.subplot(3, 1, 1)
            plt.axvline(x=self.best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best: E{self.best_epoch}')
            plt.legend()
            
            plt.subplot(3, 1, 2)
            plt.axvline(x=self.best_epoch, color='red', linestyle='--', alpha=0.7)
            
            plt.subplot(3, 1, 3)
            plt.axvline(x=self.best_epoch, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_trends.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_recommendations(self, quality_data):
        """保存推荐结果"""
        rec_path = os.path.join(self.output_dir, 'recommendations.txt')
        
        with open(rec_path, 'w', encoding='utf-8') as f:
            f.write("对比学习权重推荐结果\n")
            f.write("=" * 40 + "\n\n")
            f.write("推荐顺序 (从最佳到最差):\n\n")
            
            for i, item in enumerate(quality_data):
                epoch = item['epoch']
                data = item['data']
                score = item['quality_score']
                std = item['std']
                
                f.write(f"{i+1}. Epoch {epoch}: {os.path.basename(data['file_path'])}\n")
                f.write(f"   质量分数: {score:.3f}\n")
                f.write(f"   标准差: {std:.6f}\n")
                f.write(f"   均值: {data['stats']['mean']:.6f}\n")
                f.write(f"   加载参数: {data['num_loaded']}\n")
                f.write(f"   文件路径: {data['file_path']}\n\n")
            
            if quality_data:
                best_item = quality_data[0]
                best_epoch = best_item['epoch']
                best_data = best_item['data']
                f.write("=== 最佳推荐 ===\n")
                f.write(f"   使用: {os.path.basename(best_data['file_path'])}\n")
                f.write(f"   命令: --model_path {best_data['file_path']}\n")
        
        print(f"推荐结果已保存: {rec_path}")


def main():
    """主函数"""
    # 设置日志目录
    logs_dir = "/home/wuyou/pspnet-pytorch/logs"
    
    if not os.path.exists(logs_dir):
        print(f"错误: 日志目录不存在: {logs_dir}")
        return
    
    # 创建分析器
    analyzer = ContrastiveAnalysisFixed(
        output_dir='contrastive_analysis_results'
    )
    
    # 运行分析
    analyzer.run_comprehensive_analysis(logs_dir)
    
    # 显示总结
    print("\n" + "=" * 60)
    print("分析总结")
    print("=" * 60)
    
    if analyzer.results:
        loadable_count = sum(1 for data in analyzer.results.values() if data['can_load'])
        print(f"总分析文件: {len(analyzer.results)}")
        print(f"可加载文件: {loadable_count}")
        print(f"不可加载文件: {len(analyzer.results) - loadable_count}")
        
        if loadable_count > 0 and analyzer.best_epoch is not None:
            best_file = analyzer.results[analyzer.best_epoch]['file_path']
            print(f"\n=== 推荐使用 ===: {os.path.basename(best_file)}")
            print(f"   路径: {best_file}")
            print(f"   标准差: {analyzer.results[analyzer.best_epoch]['stats']['std']:.6f}")
            print(f"   选择理由: {analyzer.best_reason}")
    else:
        print("没有找到有效的对比学习权重文件")


if __name__ == "__main__":
    main()