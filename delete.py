import os

def remove_unwanted_pth_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.pth'):
                if filename != 'best_epoch_weights.pth':
                    file_path = os.path.join(dirpath, filename)
                    print(f"Deleting {file_path}")
                    os.remove(file_path)

if __name__ == "__main__":
    # 获取当前目录
    current_directory = "E:/yby/lab/Dataset/result/PSPNet"
    remove_unwanted_pth_files(current_directory)
