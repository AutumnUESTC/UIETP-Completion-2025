import os

def get_files_with_extension(folder):
    files_dict = {}
    for file in os.listdir(folder):
        filename, extension = os.path.splitext(file)
        files_dict[filename] = extension
    return files_dict

def compare_folders(folder1, folder2):
    files1 = get_files_with_extension(folder1)
    files2 = get_files_with_extension(folder2)

    unique_to_folder1 = {k: v for k, v in files1.items() if k not in files2}
    unique_to_folder2 = {k: v for k, v in files2.items() if k not in files1}

    return unique_to_folder1, unique_to_folder2

if __name__ == "__main__":
    folder1 = "E:/yby/lab/Dataset/JPEGImages"  # 替换为第一个文件夹的路径
    folder2 = "E:/yby/lab/IceDataset/before/SegmentationClass"  # 替换为第二个文件夹的路径

    unique1, unique2 = compare_folders(folder1, folder2)

    print("Files unique to folder 1:")
    for filename, extension in unique1.items():
        print(f"{filename}{extension}")

    print("\nFiles unique to folder 2:")
    for filename, extension in unique2.items():
        print(f"{filename}{extension}")
