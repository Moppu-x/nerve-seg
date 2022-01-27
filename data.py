# coding=gbk
# �������ݼ�
import numpy as np
import re
import os
import cv2

img_width = 128
img_height = 128


def sort_image_mask():
    # ��ȡ�ļ��б�
    filelist = os.listdir('../image/train')
    # fileList[0:50]

    # �ļ�������Ҫ�������򲢷�ΪͼƬ��mask

    # ʹ��������ʽ
    # �ļ�����ʽΪ�����1_���2.tif �� ���1_���2_mask.tif
    regex = '[0-9]+'
    reg = re.compile(regex)

    # ƥ�����1
    res1 = list(map(lambda x: reg.match(x).group(), filelist))
    # תΪ����������
    res1 = list(map(int, res1))
    # print(res1)
    # ƥ�����2
    res2 = list(map(lambda x: reg.match(x.split('_')[1]).group(), filelist))
    res2 = list(map(int, res2))
    # print(res2)
    # ����
    # ����Python�б���������µ������б�
    # ��zip���������1�����2���ļ����ϲ�ΪԪ�飬�Ա�����
    sortedList = [x for x in sorted(zip(res1, res2, filelist))]
    # print(sortedList[0:50])

    # ������Ϊ�����1-���2�Լ�ͼƬ-mask��˳��
    # ����ͼƬ��mask
    train_image = []
    train_mask = []
    # �����������б���ͼƬ��mask�ֱ���ӵ���Ӧ�б�
    for index, filename in enumerate(sortedList):
        if index % 2 == 0:
            train_image.append(filename[2])
        else:
            train_mask.append(filename[2])

    return train_image, train_mask


# ��ȡͼƬ���ݲ�����Ϊnpy�ļ�
def data_to_npy(image_list, mask_list):
    data_path = '../image/train/'
    # ���ͼƬ��list
    imgs = np.empty((len(image_list), img_height, img_width), dtype='float32')
    # ������ֵ�list
    msks = np.empty((len(mask_list), img_height, img_width), dtype='float32')

    # ��ȡͼƬ
    i = 0
    print('��ʼ��ȡͼƬ')
    for img_name in image_list:
        # cv2��ȡ�Ҷ�ͼƬ
        img = cv2.imread(data_path + img_name, cv2.IMREAD_GRAYSCALE)

        # ��С�ߴ�
        img = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_AREA)
        # תΪndarray������imgs
        img = np.array([img])
        # ��һ������(���ֵ��һ��)
        img = img / 255.0
        imgs[i] = img
        i += 1
    print('ͼƬ��ȡ���')

    # ��ȡ����
    j = 0
    print('��ʼ��ȡ����')
    for msk_name in mask_list:
        # cv2��ȡ�Ҷ�ͼƬ
        msk = cv2.imread(data_path + msk_name, cv2.IMREAD_GRAYSCALE)
        # ��С�ߴ�
        msk = cv2.resize(msk, (img_height, img_width), interpolation=cv2.INTER_AREA)
        # תΪndarray������msks
        msk = np.array([msk])
        # ��һ������(���ֵ��һ��)
        msk = msk / 255.0
        msk[msk > 0.] = 1
        msks[j] = msk
        j += 1
    print('���ֶ�ȡ���')

    # ����ȡ�����ݱ���Ϊnpy�ļ����������
    np.save('image.npy', imgs)
    np.save('mask.npy', msks)
    print('����Ϊnpy�ļ����')


# ��ȡѵ������
def data_gen():
    if not os.path.exists('image.npy'):
        print('image.npy�����ļ�������')
        return
    if not os.path.exists('mask.npy'):
        print('mask.npy�����ļ�������')
        return
    imgs = np.load('image.npy')
    msks = np.load('mask.npy')
    return imgs, msks


if __name__ == '__main__':
    # ���������ļ�����ȡ����Ϊnpy�ļ�
    image_list, mask_list = sort_image_mask()
    data_to_npy(image_list, mask_list)
