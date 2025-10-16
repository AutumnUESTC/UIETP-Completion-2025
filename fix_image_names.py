import os
import glob

# 检查训练文件中的名称与实际图像文件的匹配情况
dataset_path = 'VOCdevkit/VOC2007'
train_file = 'VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt'

# 读取训练文件
with open(train_file, 'r') as f:
    train_names = [line.strip() for line in f.readlines()]

print(f"训练文件中的样本数: {len(train_names)}")
print("训练文件中的前5个名称:", train_names[:5])

# 获取所有实际图像文件（不带扩展名）
jpeg_dir = os.path.join(dataset_path, 'JPEGImages')
actual_images = [os.path.splitext(f)[0] for f in os.listdir(jpeg_dir) if f.endswith('.jpg')]
print(f"实际图像文件数: {len(actual_images)}")
print("实际图像文件前5个:", actual_images[:5])

# 检查匹配情况
matched = 0
unmatched = []

for train_name in train_names:
    # 尝试精确匹配
    if train_name in actual_images:
        matched += 1
    else:
        # 尝试部分匹配（训练文件中的名称可能是实际文件名的前缀）
        matching_files = [img for img in actual_images if img.startswith(train_name)]
        if matching_files:
            matched += 1
            print(f"部分匹配: {train_name} -> {matching_files[0]}")
        else:
            unmatched.append(train_name)

print(f"\n匹配的图像数量: {matched}/{len(train_names)}")
print(f"不匹配的图像数量: {len(unmatched)}")
if unmatched:
    print("前5个不匹配的名称:", unmatched[:5])

# 创建一个映射字典，将训练文件中的名称映射到实际文件名
name_mapping = {}
for train_name in train_names:
    if train_name in actual_images:
        name_mapping[train_name] = train_name
    else:
        matching_files = [img for img in actual_images if img.startswith(train_name)]
        if matching_files:
            name_mapping[train_name] = matching_files[0]
        else:
            print(f"警告: 无法为 {train_name} 找到匹配的图像文件")

print(f"\n成功映射的数量: {len(name_mapping)}")

# 保存映射关系供后续使用
import json
with open('name_mapping.json', 'w') as f:
    json.dump(name_mapping, f, indent=2)

print("名称映射已保存到 name_mapping.json")
