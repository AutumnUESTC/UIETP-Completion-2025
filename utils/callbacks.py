import os
import matplotlib
import torch
import torch.nn.functional as F

from nets.pspnet import PSPNet

# 设置matplotlib后端和字体
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import cv2
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image
from .utils_metrics import compute_mIoU

# 设置matplotlib字体以避免中文警告
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']  # 添加中文字体支持
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []

        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        
        # 每10个epoch绘制一次损失图
        if epoch % 10 == 0:
            self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda,
            miou_out_path=".temp_miou_out", eval_flag=True, period=10):  # 默认改为10个epoch评估一次
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.image_ids          = image_ids
        self.dataset_path       = dataset_path
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.miou_out_path      = miou_out_path
        self.eval_flag          = eval_flag
        self.period             = period
        
        self.image_ids          = [image_id.split()[0] for image_id in image_ids]
        self.mious      = [0]
        self.accuracy   = [0]
        self.mAP        = [0]
        self.epoches    = [0]
        
        # 扩展：记录所有类别IoU和最佳指标
        self.class_iou_history = {}  # 每个类别的IoU历史
        self.class_names = ['Background', 'IcedTower', 'IcedCamera', 'IcedCameraSnow', 'IcedCameraFog', 'CoveredCamera']  # 根据您的6个类别修改
        
        # 初始化所有类别记录
        for i in range(num_classes):
            self.class_iou_history[i] = []
        
        # 扩展最佳指标跟踪
        self.best_metrics = {
            'miou': (0.0, 0),
            'accuracy': (0.0, 0), 
            'mAP': (0.0, 0),
            'class_iou': {}  # 每个类别的最佳IoU
        }
        
        # 初始化每个类别的最佳IoU
        for i in range(num_classes):
            self.best_metrics['class_iou'][i] = (0.0, 0)
        
        self.best_model_path = os.path.join(self.log_dir, 'best_epoch_weights.pth')
        
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write("Epoch, mIoU, Accuracy, mAP, Best_mIoU_Epoch\n")
                f.write(f"0, 0.0, 0.0, 0.0, 0\n")

    def get_miou_png(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(x=images)
            if len(pr) == 2:
                pr = pr[1]
            pr = pr[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   进行图片的resize
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
    
        image = Image.fromarray(np.uint8(pr))
        return image
    
    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            gt_dir = os.path.join(self.dataset_path, "SegmentationClass/")
            pred_dir = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            for image_id in tqdm(self.image_ids):
                #-------------------------------#
                #   从文件中读取图像
                #-------------------------------#
                image_path = os.path.join(self.dataset_path, "JPEGImages/"+image_id+".jpg")
                image = Image.open(image_path)
                #------------------------------#
                #   获得预测txt
                #------------------------------#
                image = self.get_miou_png(image)
                image.save(os.path.join(pred_dir, image_id + ".png"))
                        
            print("Calculate miou, accuracy and mAP.")
            _, IoUs, _, _, Accuracy, mAP = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, None)
            temp_miou = np.nanmean(IoUs) * 100
            temp_accuracy = Accuracy * 100
            temp_mAP = mAP * 100

            # 记录每个类别的IoU
            for i, iou in enumerate(IoUs):
                if i < len(self.class_iou_history):
                    self.class_iou_history[i].append(iou * 100 if not np.isnan(iou) else 0.0)

            # 简化的评估结果显示 - 只在重要里程碑显示详细信息
            if epoch % 50 == 0 or temp_miou > self.best_metrics['miou'][0]:  # 每50个epoch或新最佳时显示详细信息
                print(f"\n{'='*80}")
                print(f"Epoch {epoch} Detailed Results:")
                print(f"{'='*80}")
                print(f"Overall Metrics:")
                print(f"  mIoU:      {temp_miou:.2f}%")
                print(f"  Accuracy:  {temp_accuracy:.2f}%") 
                print(f"  mAP:       {temp_mAP:.2f}%")
                print(f"\nPer-Class IoU Details:")
                for i in range(self.num_classes):
                    class_name = self.class_names[i] if i < len(self.class_names) else f'Class{i}'
                    class_iou = IoUs[i] * 100 if not np.isnan(IoUs[i]) else 0.0
                    best_class_iou, best_class_epoch = self.best_metrics['class_iou'][i]
                    
                    improvement = ""
                    if class_iou > best_class_iou:
                        improvement = " New Best!"
                    elif class_iou == best_class_iou and best_class_iou > 0:
                        improvement = " (Tie Best)"
                        
                    print(f"  {class_name:<15}: {class_iou:6.2f}% {improvement}")
                print(f"{'='*80}")
            else:
                # 简化显示
                print(f"Epoch {epoch}: mIoU: {temp_miou:.2f}%, Acc: {temp_accuracy:.2f}%, mAP: {temp_mAP:.2f}%")

            # 更新所有最佳指标
            metrics_updated = False
            
            # 更新mIoU
            if temp_miou > self.best_metrics['miou'][0]:
                self.best_metrics['miou'] = (temp_miou, epoch)
                metrics_updated = True
                print(f" New Best mIoU! {temp_miou:.2f}% (Epoch {epoch})")
            
            # 更新准确率
            if temp_accuracy > self.best_metrics['accuracy'][0]:
                self.best_metrics['accuracy'] = (temp_accuracy, epoch)
                metrics_updated = True
                print(f" New Best Accuracy! {temp_accuracy:.2f}% (Epoch {epoch})")
            
            # 更新mAP
            if temp_mAP > self.best_metrics['mAP'][0]:
                self.best_metrics['mAP'] = (temp_mAP, epoch)
                metrics_updated = True
                print(f" New Best mAP! {temp_mAP:.2f}% (Epoch {epoch})")
            
            # 更新每个类别的最佳IoU
            for i in range(self.num_classes):
                class_iou = IoUs[i] * 100 if not np.isnan(IoUs[i]) else 0.0
                if class_iou > self.best_metrics['class_iou'][i][0]:
                    self.best_metrics['class_iou'][i] = (class_iou, epoch)
                    class_name = self.class_names[i] if i < len(self.class_names) else f'Class{i}'
                    print(f" {class_name} New Best IoU! {class_iou:.2f}% (Epoch {epoch})")
                    metrics_updated = True

            # 如果任何指标更新，保存最佳模型
            if metrics_updated:
                torch.save(model_eval.state_dict(), self.best_model_path)
                print(f" Saved best model to: {self.best_model_path}")
            
            # 只在重要epoch显示最佳记录
            if epoch % 50 == 0 or metrics_updated:
                self._print_current_best_metrics()

            self.mious.append(temp_miou)
            self.accuracy.append(temp_accuracy)
            self.mAP.append(temp_mAP)
            self.epoches.append(epoch)

            # 写入详细的信息到文件
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(f'{epoch}, {temp_miou:.2f}, {temp_accuracy:.2f}, {temp_mAP:.2f}, {self.best_metrics["miou"][1]}\n')
            
            # 同时写入类别详细IoU
            with open(os.path.join(self.log_dir, "class_iou_details.txt"), 'a') as f:
                f.write(f'Epoch {epoch}:\n')
                for i in range(self.num_classes):
                    class_name = self.class_names[i] if i < len(self.class_names) else f'Class{i}'
                    class_iou = IoUs[i] * 100 if not np.isnan(IoUs[i]) else 0.0
                    f.write(f'  {class_name}: {class_iou:.2f}%\n')
                f.write('\n')
            
            # 只在重要epoch绘制图表
            if epoch % 50 == 0 or metrics_updated:
                self._plot_comprehensive_metrics(epoch, IoUs)
            
            print("Get miou, Accuracy and mAP done.")
            shutil.rmtree(self.miou_out_path)
            
            # 返回mIoU值，供外部使用
            return temp_miou
        
        return None

    def _print_current_best_metrics(self):
        """打印当前最佳指标"""
        print(f"\n Current Best Records:")
        print(f"  mIoU:      {self.best_metrics['miou'][0]:.2f}% (Epoch {self.best_metrics['miou'][1]})")
        print(f"  Accuracy:  {self.best_metrics['accuracy'][0]:.2f}% (Epoch {self.best_metrics['accuracy'][1]})")
        print(f"  mAP:       {self.best_metrics['mAP'][0]:.2f}% (Epoch {self.best_metrics['mAP'][1]})")
        
        print(f"\nPer-Class Best IoU:")
        for i in range(self.num_classes):
            class_name = self.class_names[i] if i < len(self.class_names) else f'Class{i}'
            best_iou, best_epoch = self.best_metrics['class_iou'][i]
            print(f"  {class_name:<15}: {best_iou:6.2f}% (Epoch {best_epoch})")

    def _plot_comprehensive_metrics(self, epoch, IoUs):
        """绘制综合指标图表"""
        plt.figure(figsize=(15, 12))
        
        # 子图1: 整体指标趋势
        plt.subplot(2, 2, 1)
        plt.plot(self.epoches, self.mious, 'red', linewidth=2, label='mIoU')
        plt.plot(self.epoches, self.accuracy, 'blue', linewidth=2, label='Accuracy')
        plt.plot(self.epoches, self.mAP, 'green', linewidth=2, label='mAP')
        
        # 标记最佳mIoU点
        best_miou, best_epoch_miou = self.best_metrics['miou']
        if best_miou > 0:
            plt.plot(best_epoch_miou, best_miou, 'ro', markersize=8, 
                    label=f'Best mIoU: {best_miou:.2f}%')
        
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Percentage (%)')
        plt.title('Overall Metrics Trend')
        plt.legend(loc="lower right")
        
        # 子图2: 当前epoch各类别IoU
        plt.subplot(2, 2, 2)
        categories = [self.class_names[i] if i < len(self.class_names) else f'Class{i}' 
                     for i in range(self.num_classes)]
        current_ious = [IoUs[i] * 100 if not np.isnan(IoUs[i]) else 0.0 
                       for i in range(self.num_classes)]
        
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_classes))
        bars = plt.bar(categories, current_ious, color=colors)
        
        # 在柱状图上显示数值
        for bar, iou in zip(bars, current_ious):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{iou:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.xticks(rotation=45)
        plt.ylabel('IoU (%)')
        plt.title(f'Epoch {epoch} Per-Class IoU')
        plt.ylim(0, 100)
        
        # 子图3: 各类别最佳IoU对比
        plt.subplot(2, 2, 3)
        best_ious = [self.best_metrics['class_iou'][i][0] for i in range(self.num_classes)]
        bars = plt.bar(categories, best_ious, color=colors, alpha=0.7)
        
        for bar, iou in zip(bars, best_ious):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{iou:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.xticks(rotation=45)
        plt.ylabel('Best IoU (%)')
        plt.title('Per-Class Historical Best IoU')
        plt.ylim(0, 100)
        
        # 子图4: 关键指标总结
        plt.subplot(2, 2, 4)
        # 创建文本总结
        summary_text = f"Training Summary (Epoch {epoch})\n\n"
        summary_text += f"Current mIoU: {self.mious[-1]:.2f}%\n"
        summary_text += f"Best mIoU: {best_miou:.2f}%\n"
        summary_text += f"Best Accuracy: {self.best_metrics['accuracy'][0]:.2f}%\n"
        summary_text += f"Best mAP: {self.best_metrics['mAP'][0]:.2f}%\n\n"
        summary_text += f"Total Epochs: {epoch}"
        
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top', linespacing=1.5)
        plt.axis('off')
        plt.title('Training Summary')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"epoch_{epoch}_comprehensive_metrics.png"), 
                   dpi=300, bbox_inches='tight')
        plt.cla()
        plt.close("all")
        
        # 额外保存一个专门的最佳指标图表
        self._plot_best_metrics_summary()

    def _plot_best_metrics_summary(self):
        """绘制最佳指标总结图"""
        plt.figure(figsize=(12, 8))
        
        # 准备数据
        categories = ['mIoU', 'Accuracy', 'mAP'] + \
                   [self.class_names[i] if i < len(self.class_names) else f'Class{i}' 
                    for i in range(self.num_classes)]
        
        values = [
            self.best_metrics['miou'][0],
            self.best_metrics['accuracy'][0], 
            self.best_metrics['mAP'][0]
        ] + [self.best_metrics['class_iou'][i][0] for i in range(self.num_classes)]
        
        epochs = [
            self.best_metrics['miou'][1],
            self.best_metrics['accuracy'][1],
            self.best_metrics['mAP'][1]
        ] + [self.best_metrics['class_iou'][i][1] for i in range(self.num_classes)]
        
        colors = ['red', 'blue', 'green'] + \
                plt.cm.Set3(np.linspace(0, 1, self.num_classes)).tolist()
        
        # 创建柱状图
        bars = plt.bar(range(len(categories)), values, color=colors, alpha=0.7)
        
        # 添加数值和轮次标签
        for i, (bar, value, epoch) in enumerate(zip(bars, values, epochs)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
            plt.text(bar.get_x() + bar.get_width()/2, -5,
                    f'E{epoch}', ha='center', va='top', fontsize=8, rotation=45)
        
        plt.xlabel('Metric Category')
        plt.ylabel('Best Value (%)')
        plt.title('All Metrics Best Records Summary')
        plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
        plt.ylim(0, max(values) * 1.15 if values else 100)
        plt.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "best_metrics_summary.png"), 
                   dpi=300, bbox_inches='tight')
        plt.cla()
        plt.close("all")

    def get_best_metrics(self):
        """获取所有最佳指标的完整信息"""
        return self.best_metrics

    def save_final_report(self):
        """保存最终训练报告"""
        report_path = os.path.join(self.log_dir, "final_training_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Final Training Report\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Overall Best Metrics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Best mIoU:      {self.best_metrics['miou'][0]:.2f}% (Epoch {self.best_metrics['miou'][1]})\n")
            f.write(f"Best Accuracy:  {self.best_metrics['accuracy'][0]:.2f}% (Epoch {self.best_metrics['accuracy'][1]})\n")
            f.write(f"Best mAP:       {self.best_metrics['mAP'][0]:.2f}% (Epoch {self.best_metrics['mAP'][1]})\n\n")
            
            f.write("Per-Class Best IoU:\n")
            f.write("-" * 40 + "\n")
            for i in range(self.num_classes):
                class_name = self.class_names[i] if i < len(self.class_names) else f'Class{i}'
                best_iou, best_epoch = self.best_metrics['class_iou'][i]
                f.write(f"{class_name:<15}: {best_iou:6.2f}% (Epoch {best_epoch})\n")
            
            f.write(f"\nBest Model Path: {self.best_model_path}\n")
            f.write(f"Total Training Epochs: {max(self.epoches) if self.epoches else 0}\n")
        
        print(f" Final report saved: {report_path}")
        return report_path


if __name__ =='__main__':
    image = torch.randn(8,3,512,512)
    model = PSPNet(num_classes=6, backbone=1, downsample_factor=8,
                   pretrained=False, aux_branch=False)
    # model_train = model.eval()
    pr = model(image)