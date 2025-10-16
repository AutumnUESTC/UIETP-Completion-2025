#!/usr/bin/env python3
import os
import sys
import numpy as np
import cv2

# 在导入matplotlib之前设置Agg后端
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

# 添加项目路径
sys.path.append('/home/wuyou/pspnet-pytorch')

# 导入特征工程类
from nets.feature_engineering import FeatureEngineering


class FixedFeatureDemo:
    """修复版特征工程演示 - 修复所有错误"""
    
    def __init__(self):
        self.output_dir = "/home/wuyou/pspnet-pytorch/feature_demo_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 示例图像路径
        self.demo_image_path = "/home/wuyou/pspnet-pytorch/img/CC2757-20230102-150740.jpg"
        
        # 设置matplotlib参数 - 简化字体设置避免警告
        plt.rcParams['font.family'] = ['DejaVu Sans']  # 只使用DejaVu Sans
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.titleweight'] = 'bold'
    
    def run_fixed_demo(self):
        """运行修复版演示"""
        print("=" * 60)
        print("Feature Engineering Demo - Fixed Version")
        print("=" * 60)
        
        # 检查图像
        if not os.path.exists(self.demo_image_path):
            print(f"Error: Demo image not found: {self.demo_image_path}")
            return []
        
        # 1. 加载并预处理图像
        print("1. Loading example image...")
        image = Image.open(self.demo_image_path)
        original_image = np.array(image)
        print(f"   Original size: {original_image.shape}")
        
        # 调整尺寸
        target_size = (512, 512)
        if original_image.shape[0] != target_size[0] or original_image.shape[1] != target_size[1]:
            original_image = cv2.resize(original_image, target_size, interpolation=cv2.INTER_AREA)
            print(f"   Resized to: {original_image.shape}")
        
        # 2. 提取特征
        print("2. Extracting feature engineering features...")
        try:
            features_dict, all_features = self.extract_all_features_detailed(original_image)
            print(f"   After feature engineering: {all_features.shape}")
        except Exception as e:
            print(f"   Feature extraction failed: {e}")
            return []
        
        # 3. 创建可视化
        print("3. Generating visualization results...")
        result_files = []
        
        # 创建综合对比图
        try:
            result_files.append(self.create_comprehensive_comparison(original_image, features_dict, all_features))
        except Exception as e:
            print(f"   Comprehensive comparison failed: {e}")
        
        # 创建简化对比图
        try:
            result_files.append(self.create_simple_comparison(original_image, features_dict, all_features))
        except Exception as e:
            print(f"   Simple comparison failed: {e}")
        
        # 创建统计信息图
        try:
            result_files.append(self.create_feature_statistics_report(original_image, features_dict, all_features))
        except Exception as e:
            print(f"   Statistics report failed: {e}")
        
        # 4. 打印结果
        print("\n" + "=" * 60)
        print("Demo Completed!")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print("\nGenerated files:")
        for file in result_files:
            if file and os.path.exists(file):
                print(f"  ✓ {os.path.basename(file)}")
                file_size = os.path.getsize(file) / 1024  # KB
                print(f"     Size: {file_size:.1f} KB")
        
        print(f"\nFeature Engineering Summary:")
        print(f"  • Original image: 3 channels (RGB)")
        print(f"  • After feature engineering: {all_features.shape[2]} channels")
        print(f"  • Dimension expansion: {all_features.shape[2] / 3:.1f}x")
        print(f"  • Contains 5 feature types: Multi-scale(3) + Edge(3) + Texture(3) + Frequency(3) + Color(6)")
        
        return result_files
    
    def extract_all_features_detailed(self, image_np):
        """详细提取所有特征"""
        print("\nExtracting features in detail...")
        
        features = {}
        
        # 1. 多尺度特征
        print("  1. Multi-scale features...")
        features['multi_scale'] = FeatureEngineering.multi_scale_features(image_np)
        print(f"     Shape: {features['multi_scale'].shape}")
        
        # 2. 边缘特征
        print("  2. Edge features...")
        features['edge'] = FeatureEngineering.edge_enhancement(image_np)
        print(f"     Shape: {features['edge'].shape}")
        
        # 3. 纹理特征
        print("  3. Texture features...")
        features['texture'] = FeatureEngineering.texture_features(image_np)
        print(f"     Shape: {features['texture'].shape}")
        
        # 4. 频域特征
        print("  4. Frequency features...")
        features['frequency'] = FeatureEngineering.frequency_domain_features(image_np)
        print(f"     Shape: {features['frequency'].shape}")
        
        # 5. 颜色特征
        print("  5. Color features...")
        features['color'] = FeatureEngineering.color_space_features(image_np)
        print(f"     Shape: {features['color'].shape}")
        
        # 6. 合并所有特征
        print("  6. Merging all features...")
        all_features = np.concatenate([
            features['multi_scale'],
            features['edge'], 
            features['texture'],
            features['frequency'],
            features['color']
        ], axis=2)
        print(f"     Final shape: {all_features.shape}")
        
        return features, all_features
    
    def create_comprehensive_comparison(self, original, features_dict, all_features):
        """创建综合对比可视化 - 修复键名错误"""
        print("  Creating comprehensive comparison...")
        
        # 使用更合理的布局和尺寸
        fig = plt.figure(figsize=(20, 16))
        
        # 使用更简单的网格布局
        gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # 主标题
        fig.suptitle('Feature Engineering Demo - Comprehensive Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. 原始图像
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original)
        ax1.set_title('Original Image\n(3 channels RGB)', fontsize=12, pad=10)
        ax1.axis('off')
        
        # 2. 特征统计概览
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_feature_statistics(ax2, features_dict)
        ax2.set_title('Feature Statistics Overview', fontsize=12, pad=10)
        
        # 3. 维度对比
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_dimension_comparison(ax3, original, all_features)
        ax3.set_title('Dimension Comparison', fontsize=12, pad=10)
        
        # 4. 多尺度特征 - 修复键名：使用 multi_scale 而不是 multi-scale
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_single_feature(ax4, features_dict['multi_scale'][:, :, 0], 
                                 'Multi-scale\n(Original Scale)', 'viridis')
        
        # 5. 边缘特征
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_single_feature(ax5, features_dict['edge'][:, :, 0], 
                                 'Edge Enhancement\n(Sobel Gradient)', 'gray')
        
        # 6. 纹理特征
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_single_feature(ax6, features_dict['texture'][:, :, 0], 
                                 'Texture Features\n(LBP Texture)', 'plasma')
        
        # 7. 频域特征
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_single_feature(ax7, features_dict['frequency'][:, :, 0], 
                                 'Frequency Domain\n(FFT Magnitude)', 'hot')
        
        # 8. HSV颜色特征 - Hue
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_single_feature(ax8, features_dict['color'][:, :, 0], 
                                 'HSV Color Space\n(Hue Channel)', 'hsv')
        
        # 9. LAB颜色特征 - Lightness
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_single_feature(ax9, features_dict['color'][:, :, 3], 
                                 'LAB Color Space\n(Lightness)', 'gray')
        
        # 10. 详细总结
        ax10 = fig.add_subplot(gs[3, :])
        self._create_detailed_summary(ax10, features_dict, all_features, original)
        ax10.set_title('Detailed Feature Engineering Summary', fontsize=12, pad=10)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "feature_engineering_comprehensive.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Comprehensive comparison saved: {output_path}")
        plt.close()
        
        return output_path
    
    def _plot_single_feature(self, ax, feature_data, title, cmap='viridis'):
        """绘制单个特征通道"""
        if feature_data.max() > feature_data.min():
            feature_display = (feature_data - feature_data.min()) / (feature_data.max() - feature_data.min())
        else:
            feature_display = feature_data
            
        im = ax.imshow(feature_display, cmap=cmap)
        ax.set_title(title, fontsize=11, pad=8)
        ax.axis('off')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_feature_statistics(self, ax, features_dict):
        """绘制特征统计信息"""
        feature_names = ['Multi-scale', 'Edge', 'Texture', 'Frequency', 'Color']
        means = []
        stds = []
        
        for name in feature_names:
            # 修复键名：使用下划线而不是连字符
            feature_key = name.lower().replace('-', '_')
            feature = features_dict[feature_key]
            # 计算所有通道的均值和标准差
            channel_means = [np.mean(feature[:, :, i]) for i in range(feature.shape[2])]
            channel_stds = [np.std(feature[:, :, i]) for i in range(feature.shape[2])]
            means.append(np.mean(channel_means))
            stds.append(np.mean(channel_stds))
        
        x_pos = np.arange(len(feature_names))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                     color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_ylabel('Mean Value ± Std')
        ax.grid(True, alpha=0.3)
        
        # 在柱子上添加数值
        for bar, mean_val in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    def _plot_dimension_comparison(self, ax, original, all_features):
        """绘制维度对比图"""
        categories = ['Input\n(RGB)', 'Output\n(All Features)']
        input_dims = original.shape[2]  # 3 channels
        output_dims = all_features.shape[2]  # 18 channels
        
        bars = ax.bar(categories, [input_dims, output_dims], 
                     color=['lightblue', 'lightcoral'], alpha=0.7)
        
        ax.set_ylabel('Number of Channels')
        ax.set_ylim(0, max(input_dims, output_dims) * 1.2)
        ax.grid(True, alpha=0.3)
        
        # 在柱子上添加数值和扩展比例
        for i, (bar, dim) in enumerate(zip(bars, [input_dims, output_dims])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{dim} channels', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            if i == 1:
                expansion = output_dims / input_dims
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                       f'{expansion:.1f}x', ha='center', va='center', 
                       fontsize=12, fontweight='bold', color='white')
    
    def _create_detailed_summary(self, ax, features_dict, all_features, original):
        """创建详细总结面板"""
        # 清空坐标轴
        ax.clear()
        ax.axis('off')
        
        # 计算统计信息
        original_pixels = original.shape[0] * original.shape[1] * original.shape[2]
        feature_pixels = all_features.size
        expansion_ratio = feature_pixels / original_pixels
        
        feature_info = []
        total_channels = 0
        for name, feature in features_dict.items():
            channels = feature.shape[2]
            total_channels += channels
            feature_info.append(f"{name}: {channels} channels")
        
        # 构建总结文本
        summary_text = (
            "Feature Engineering Process Summary:\n\n"
            "Input Configuration:\n"
            f"  • Original image: {original.shape[1]}×{original.shape[0]} pixels, {original.shape[2]} channels (RGB)\n"
            f"  • Total input elements: {original_pixels:,}\n\n"
            
            "Feature Extraction:\n"
            f"  • Multi-scale features: 3 channels\n"
            f"  • Edge enhancement: 3 channels\n" 
            f"  • Texture analysis: 3 channels\n"
            f"  • Frequency domain: 3 channels\n"
            f"  • Color space expansion: 6 channels\n\n"
            
            "Output Results:\n"
            f"  • Total output channels: {total_channels}\n"
            f"  • Total output elements: {feature_pixels:,}\n"
            f"  • Dimension expansion: {total_channels / original.shape[2]:.1f}x\n"
            f"  • Data volume expansion: {expansion_ratio:.1f}x\n\n"
            
            "Feature Characteristics:\n"
            f"  • Value range: [{all_features.min():.3f}, {all_features.max():.3f}]\n"
            f"  • Global mean: {all_features.mean():.3f}\n"
            f"  • Global std: {all_features.std():.3f}"
        )
        
        # 显示文本
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', linespacing=1.5,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def create_simple_comparison(self, original, features_dict, all_features):
        """创建简化对比图 - 修复索引越界问题"""
        print("  Creating simple comparison...")
        
        # 使用更安全的布局：2行3列
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Feature Engineering: Before vs After', fontsize=16, fontweight='bold')
        
        # 确保axes是2D数组
        axes = axes.reshape(2, 3)
        
        # 1. 原始图像
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original Image\n(3 channels)', fontsize=12)
        axes[0, 0].axis('off')
        
        # 2-6. 显示代表性的特征通道
        feature_examples = [
            (features_dict['multi_scale'][:, :, 0], 'Multi-scale\n(Original)', 'viridis'),
            (features_dict['edge'][:, :, 0], 'Edge\n(Sobel)', 'gray'),
            (features_dict['texture'][:, :, 0], 'Texture\n(LBP)', 'plasma'),
            (features_dict['frequency'][:, :, 0], 'Frequency\n(FFT)', 'hot'),
            (features_dict['color'][:, :, 0], 'Color\n(Hue)', 'hsv')
        ]
        
        # 安全地填充子图
        positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        
        for (row, col), (feature, title, cmap) in zip(positions, feature_examples):
            if feature.max() > feature.min():
                feature_display = (feature - feature.min()) / (feature.max() - feature.min())
            else:
                feature_display = feature
                
            axes[row, col].imshow(feature_display, cmap=cmap)
            axes[row, col].set_title(title, fontsize=11)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "before_after_comparison.png")
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        print(f"Simple comparison saved: {output_path}")
        plt.close()
        
        return output_path
    
    def create_feature_statistics_report(self, original, features_dict, all_features):
        """创建特征统计报告"""
        print("  Creating feature statistics report...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Feature Engineering Statistics Report', fontsize=16, fontweight='bold')
        
        # 确保axes是2D数组
        axes = axes.reshape(2, 2)
        
        # 1. 各特征类型统计
        feature_types = ['Multi-scale', 'Edge', 'Texture', 'Frequency', 'Color']
        feature_channels = [3, 3, 3, 3, 6]
        
        axes[0, 0].pie(feature_channels, labels=feature_types, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Channel Distribution by Feature Type', fontsize=12)
        
        # 2. 特征值分布直方图
        flattened_features = all_features.reshape(-1)
        axes[0, 1].hist(flattened_features, bins=50, alpha=0.7, density=True, color='skyblue')
        axes[0, 1].set_xlabel('Feature Values')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('All Feature Values Distribution', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 各特征类型的数值范围
        feature_ranges = []
        for name in ['multi_scale', 'edge', 'texture', 'frequency', 'color']:
            feature = features_dict[name]
            ranges = [np.ptp(feature[:, :, i]) for i in range(feature.shape[2])]
            feature_ranges.append(np.mean(ranges))
        
        x_pos = np.arange(len(feature_types))
        axes[1, 0].bar(x_pos, feature_ranges, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(feature_types, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Value Range')
        axes[1, 0].set_title('Feature Value Ranges', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 信息总结
        original_size = f"{original.shape[1]}×{original.shape[0]}×{original.shape[2]}"
        feature_size = f"{all_features.shape[1]}×{all_features.shape[0]}×{all_features.shape[2]}"
        expansion = all_features.size / original.size
        
        summary_text = (
            f"Statistical Summary:\n\n"
            f"Input Dimensions: {original_size}\n"
            f"Output Dimensions: {feature_size}\n"
            f"Total Channels: {all_features.shape[2]}\n"
            f"Data Expansion: {expansion:.1f}x\n\n"
            f"Value Statistics:\n"
            f"Min: {flattened_features.min():.3f}\n"
            f"Max: {flattened_features.max():.3f}\n"
            f"Mean: {flattened_features.mean():.3f}\n"
            f"Std: {flattened_features.std():.3f}"
        )
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, fontsize=11,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "feature_statistics_report.png")
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        print(f"Statistics report saved: {output_path}")
        plt.close()
        
        return output_path


def main():
    """主函数"""
    demo = FixedFeatureDemo()
    demo.run_fixed_demo()


if __name__ == "__main__":
    main()