# import os
#
# with open('E:\yby\lab\Dataset\ImageSets\Segmentation\\val.txt', 'r') as f:
#     lines = f.readlines()
#
#     # 处理每一行并保存到临时文件
# new_lines = []
# for line in lines:
#     filename = line.strip().split(" ")[0]
#     basename = os.path.splitext(os.path.basename(filename))[0]
#     new_lines.append(basename + "\n")
#
# with open("E:\yby\lab\Dataset\ImageSets\Segmentation\\val1.txt", "w") as f:
#     f.writelines(new_lines)
#
# # 将临时文件内容覆盖原文件
# os.replace("E:\yby\lab\Dataset\ImageSets\Segmentation\\val1.txt", "E:\yby\lab\Dataset\ImageSets\Segmentation\\val.txt")
import os
import shutil
txt_path="E:\yby\lab\Dataset\ImageSets\Segmentation\\train.txt"
images_dir="E:\yby\lab\Dataset\JPEGImages"
target_dir="E:\yby\lab\Dataset\\train_image"
with open(txt_path, "r") as f:
    lines = f.readlines()
    for line in lines:
        image_id = line.strip().split(" ")[0]
        image_path = os.path.join(images_dir, image_id + ".jpg")
        if os.path.exists(image_path):
            # 如果图像存在，则保存到目标文件夹
            shutil.copy(image_path, target_dir)

# import os
#
# # path表示路径
# path="C:/Users/PC/Desktop/split/val/"
# # 返回path下所有文件构成的一个list列表
# filelist=os.listdir(path)
# # 遍历输出每一个文件的名字和类型
# f = open("C:/Users/PC/Desktop/split/val.txt","w")
# for item in filelist:
#     # 输出指定后缀类型的文件
#     # if(item.endswith('.jpg')):
#     #     print(item)
#     f.write(item.split(".")[0] + "\n")
# f.close()