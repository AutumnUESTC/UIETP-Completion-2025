import os
import argparse
import random
import shutil
from shutil import copyfile
# from misc import printProgressBar
from tqdm import tqdm

def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s' % dir_path)
    os.makedirs(dir_path)
    print('Create path - %s' % dir_path)


def main(config):
    rm_mkdir(config.train_path)
    # rm_mkdir(config.train_GT_path)
    rm_mkdir(config.valid_path)
    # rm_mkdir(config.valid_GT_path)
    rm_mkdir(config.test_path)
    # rm_mkdir(config.test_GT_path)

    filenames = os.listdir(config.origin_data_path)
    data = []
    data_list = []
    # GT_list = []
    # .../aug_image/0_20_17563_train.png
    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        if ext == '.jpg':
            filename = filename.split('train')[-1]
            data.append(filename)
            # data_list.append('train\\' + filename)
            # GT_list.append('label' + filename)

    num_total = len(data)
    num_train = int((config.train_ratio / (config.train_ratio + config.valid_ratio + config.test_ratio)) * num_total)
    num_valid = int((config.valid_ratio / (config.train_ratio + config.valid_ratio + config.test_ratio)) * num_total)
    num_test = num_total - num_train - num_valid

    print('\nNum of train set : ', num_train)
    print('\nNum of valid set : ', num_valid)
    print('\nNum of test set : ', num_test)

    Arange = list(range(num_total))
    random.shuffle(Arange)

    # for i in range(num_train):
    for i in tqdm(range(num_train), desc='Producing train set'):
        idx = Arange.pop()
        print(idx)
        # print(config.origin_data_path)
        # print(config.train_path)
        # print((data_list[idx]))
        src = os.path.join(config.origin_data_path, data[idx])
        dst = os.path.join(config.train_path, data[idx])
        # print("src:"+src)
        # print("dst:"+dst)
        copyfile(src, dst)
        # try:
        #     copyfile(src, dst)
        # except FileNotFoundError as e:
        #     print(f"Error copying file: {e}")
        #     print(f"Source file: {src}")
        #     print(f"Destination file: {dst}")
        #     raise e

        print(src, dst)
        # src = os.path.join(config.origin_GT_path, GT_list[idx])
        # dst = os.path.join(config.train_GT_path, GT_list[idx])
        # copyfile(src, dst)
        # print(src, dst)
        # printProgressBar(i + 1, num_train, prefix='Producing train set:', suffix='Complete', length=50)


    # for i in range(num_valid):
    for i in tqdm(range(num_valid), desc='Producing valid set'):
        idx = Arange.pop()

        src = os.path.join(config.origin_data_path, data[idx])
        dst = os.path.join(config.valid_path, data[idx])
        copyfile(src, dst)

        # src = os.path.join(config.origin_GT_path, GT_list[idx])
        # dst = os.path.join(config.valid_GT_path, GT_list[idx])
        # copyfile(src, dst)

        # printProgressBar(i + 1, num_valid, prefix='Producing valid set:', suffix='Complete', length=50)

    # for i in range(num_test):
    for i in tqdm(range(num_test), desc='Producing test set'):
        idx = Arange.pop()

        src = os.path.join(config.origin_data_path, data[idx])
        dst = os.path.join(config.test_path, data[idx])
        copyfile(src, dst)

        # src = os.path.join(config.origin_GT_path, GT_list[idx])
        # dst = os.path.join(config.test_GT_path, GT_list[idx])
        # copyfile(src, dst)

        # printProgressBar(i + 1, num_test, prefix='Producing test set:', suffix='Complete', length=50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.1)

    # data path
    parser.add_argument('--origin_data_path', type=str, default='E:\\yby\\lab\\Dataset\\JPEGImages\\')
    # parser.add_argument.*task/task1/label')

    parser.add_argument('--train_path', type=str, default='C:\\Users\\PC\\Desktop\\split\\train\\')
    # parser.add_argument('--train_GT_path', type=str, default='./dataset/train_labels/')
    parser.add_argument('--valid_path', type=str, default='C:\\Users\\PC\\Desktop\\split\\val\\')
    # parser.add_argument('--valid_GT_path', type=str, default='./dataset/val_labels/')
    parser.add_argument('--test_path', type=str, default='C:\\Users\\PC\\Desktop\\split\\test\\')
    # parser.add_argument('--test_GT_path', type=str, default='./dataset/test_labels/')

    config = parser.parse_args()
    print(config)
    main(config)
