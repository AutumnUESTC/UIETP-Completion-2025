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
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        print(f"结果将保存到: {os.path.abspath(output_dir)}")
    
    def collect_contrastive_weights(self, logs_dir):
        """收集所有对比学习权重文件"""
        print("收集对比学习权重文件...")
        
        contrastive_files = []
        for file in os.listdir(logs_dir):
            if file.startswith('contrastive_pretrained_backbone') and file.endswith('.pth'):
                # 提取epoch数字
                if 'epoch' in file:
                    epoch_str = file.split('epoch')[-1].replace('.pth', '')
                    if epoch_str.isdigit():
                        epoch = int(epoch_str)
                    else:
                        epoch = 0
                else:
                    epoch = 0
                
                file_path = os.path.join(logs_dir, file)
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
            from nets.pspnet import PSPNet
            
            model = PSPNet(
                num_classes=6,
                backbone=backbone_type,
                downsample_factor=8,
                pretrained=False
            )
            
            checkpoint = torch.load(weight_path, map_location='cpu')
            
            # 策略2: 添加backbone前缀
            model_dict = model.state_dict()
            matched_keys = []
            
            for k, v in checkpoint.items():
                new_key = f"backbone.{k}"
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
        
        for epoch, file_path in contrastive_files:
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
        
        # 3. 生成分析报告
        self.generate_analysis_report()
        
        # 4. 可视化结果
        self.create_visualizations()
        
        print(f"\n分析完成! 结果保存在: {self.output_dir}")
    
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
            
            if loadable_files:
                f.write("推荐使用的权重文件 (按标准差排序):\n")
                # 按标准差排序 (适中的标准差通常更好)
                sorted_results = sorted(
                    [(epoch, data) for epoch, data in self.results.items() if data['can_load']],
                    key=lambda x: abs(x[1]['stats']['std'] - 0.01)  # 距离理想标准差0.01的差距
                )
                
                for i, (epoch, data) in enumerate(sorted_results[:5]):  # 显示前5个
                    f.write(f"  {i+1}. Epoch {epoch}: {os.path.basename(data['file_path'])}\n")
                    f.write(f"     标准差: {data['stats']['std']:.6f}, 加载参数: {data['num_loaded']}\n")
            
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
        plt.figure(figsize=(12, 8))
        
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
        
        # 2. 创建推荐图表
        self.create_recommendation_chart()
        
        print(f"可视化图表已保存到: {self.output_dir}")
    
    def create_recommendation_chart(self):
        """创建推荐图表"""
        if not self.results:
            return
        
        # 筛选可加载的文件
        loadable_results = {epoch: data for epoch, data in self.results.items() if data['can_load']}
        
        if not loadable_results:
            return
        
        # 按标准差质量排序 (距离理想标准差0.01的差距)
        sorted_results = sorted(
            loadable_results.items(),
            key=lambda x: abs(x[1]['stats']['std'] - 0.01)
        )
        
        epochs = [epoch for epoch, _ in sorted_results]
        stds = [data['stats']['std'] for _, data in sorted_results]
        quality_scores = [1.0 / (1.0 + abs(std - 0.01)) for std in stds]  # 质量分数
        
        plt.figure(figsize=(10, 6))
        
        # 创建条形图
        bars = plt.bar(range(len(epochs)), quality_scores, 
                      color=plt.cm.viridis(quality_scores))
        
        plt.xlabel('Epoch (Sorted by Recommendation)')
        plt.ylabel('Quality Score')
        plt.title('Contrastive Learning Weight Quality Recommendation\n(Based on how close std is to 0.01)')
        
        # 添加数值标签
        for i, (bar, epoch, std, score) in enumerate(zip(bars, epochs, stds, quality_scores)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'Epoch{epoch}\nstd:{std:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.xticks(range(len(epochs)), [f'E{epoch}' for epoch in epochs])
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, max(quality_scores) * 1.2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'weight_recommendations.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存推荐结果
        self.save_recommendations(sorted_results)
    
    def save_recommendations(self, sorted_results):
        """保存推荐结果"""
        rec_path = os.path.join(self.output_dir, 'recommendations.txt')
        
        with open(rec_path, 'w', encoding='utf-8') as f:
            f.write("对比学习权重推荐结果\n")
            f.write("=" * 40 + "\n\n")
            f.write("推荐顺序 (从最佳到最差):\n\n")
            
            for i, (epoch, data) in enumerate(sorted_results):
                f.write(f"{i+1}. Epoch {epoch}: {os.path.basename(data['file_path'])}\n")
                f.write(f"   标准差: {data['stats']['std']:.6f}\n")
                f.write(f"   均值: {data['stats']['mean']:.6f}\n")
                f.write(f"   加载参数: {data['num_loaded']}\n")
                f.write(f"   文件路径: {data['file_path']}\n\n")
            
            if sorted_results:
                best_epoch, best_data = sorted_results[0]
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
        
        if loadable_count > 0:
            # 找到最佳推荐
            loadable_results = {epoch: data for epoch, data in analyzer.results.items() if data['can_load']}
            best_epoch = min(loadable_results.keys(), 
                           key=lambda x: abs(loadable_results[x]['stats']['std'] - 0.01))
            
            best_file = loadable_results[best_epoch]['file_path']
            print(f"\n=== 推荐使用 ===: {os.path.basename(best_file)}")
            print(f"   路径: {best_file}")
            print(f"   标准差: {loadable_results[best_epoch]['stats']['std']:.6f}")
    else:
        print("没有找到有效的对比学习权重文件")


if __name__ == "__main__":
    main()