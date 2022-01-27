# coding=gbk
# 构建数据集
import numpy as np
import re
import os
import cv2

img_width = 128
img_height = 128


def sort_image_mask():
    # 读取文件列表
    filelist = os.listdir('../image/train')
    # fileList[0:50]

    # 文件乱序，需要进行排序并分为图片和mask

    # 使用正则表达式
    # 文件名格式为：序号1_序号2.tif 或 序号1_序号2_mask.tif
    regex = '[0-9]+'
    reg = re.compile(regex)

    # 匹配序号1
    res1 = list(map(lambda x: reg.match(x).group(), filelist))
    # 转为整数以排序
    res1 = list(map(int, res1))
    # print(res1)
    # 匹配序号2
    res2 = list(map(lambda x: reg.match(x.split('_')[1]).group(), filelist))
    res2 = list(map(int, res2))
    # print(res2)
    # 排序
    # 利用Python列表解析创建新的有序列表
    # 用zip函数将序号1，序号2和文件名合并为元组，以便排序
    sortedList = [x for x in sorted(zip(res1, res2, filelist))]
    # print(sortedList[0:50])

    # 排序结果为按序号1-序号2以及图片-mask的顺序
    # 分离图片和mask
    train_image = []
    train_mask = []
    # 遍历排序后的列表，将图片和mask分别添加到对应列表
    for index, filename in enumerate(sortedList):
        if index % 2 == 0:
            train_image.append(filename[2])
        else:
            train_mask.append(filename[2])

    return train_image, train_mask


# 读取图片数据并保存为npy文件
def data_to_npy(image_list, mask_list):
    data_path = '../image/train/'
    # 存放图片的list
    imgs = np.empty((len(image_list), img_height, img_width), dtype='float32')
    # 存放遮罩的list
    msks = np.empty((len(mask_list), img_height, img_width), dtype='float32')

    # 读取图片
    i = 0
    print('开始读取图片')
    for img_name in image_list:
        # cv2读取灰度图片
        img = cv2.imread(data_path + img_name, cv2.IMREAD_GRAYSCALE)

        # 缩小尺寸
        img = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_AREA)
        # 转为ndarray，加入imgs
        img = np.array([img])
        # 归一化处理(最大值归一化)
        img = img / 255.0
        imgs[i] = img
        i += 1
    print('图片读取完成')

    # 读取遮罩
    j = 0
    print('开始读取遮罩')
    for msk_name in mask_list:
        # cv2读取灰度图片
        msk = cv2.imread(data_path + msk_name, cv2.IMREAD_GRAYSCALE)
        # 缩小尺寸
        msk = cv2.resize(msk, (img_height, img_width), interpolation=cv2.INTER_AREA)
        # 转为ndarray，加入msks
        msk = np.array([msk])
        # 归一化处理(最大值归一化)
        msk = msk / 255.0
        msk[msk > 0.] = 1
        msks[j] = msk
        j += 1
    print('遮罩读取完成')

    # 将读取的数据保存为npy文件，方便访问
    np.save('image.npy', imgs)
    np.save('mask.npy', msks)
    print('保存为npy文件完成')


# 获取训练数据
def data_gen():
    if not os.path.exists('image.npy'):
        print('image.npy数据文件不存在')
        return
    if not os.path.exists('mask.npy'):
        print('mask.npy数据文件不存在')
        return
    imgs = np.load('image.npy')
    msks = np.load('mask.npy')
    return imgs, msks


if __name__ == '__main__':
    # 整理数据文件，读取保存为npy文件
    image_list, mask_list = sort_image_mask()
    data_to_npy(image_list, mask_list)
