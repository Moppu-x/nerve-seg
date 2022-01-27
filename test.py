# coding=gbk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import array_to_img
import os

from train import unet

img_width = 128
img_height = 128

model = unet()
model.load_weights('model.h5')

img = cv2.imread('../image/train/20_33.tif', cv2.IMREAD_GRAYSCALE)
# 缩小尺寸
img = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_AREA)
# 转为ndarray，加入imgs
img = np.array([img])
# 归一化处理(最大值归一化)
img = img / 255.0

img_pred = model.predict(img)
print(img_pred)
# 图像还原
img_pred = (img_pred * 255).astype(np.uint8)
# img_pred = np.clip(img_pred, 0, 255)
print(img_pred[0, :, :, 0].shape)
print(type(img_pred))

img_pred = cv2.resize(img_pred[0, :, :, 0], (580, 420), interpolation=cv2.INTER_NEAREST)
# img_pred = array_to_img(img_pred[:, :, np.newaxis])

if not os.path.exists('./predict'):
    os.makedirs('./predict')
cv2.imwrite('./predict/img_red.png', img_pred)
#
# print(img_pred.shape)
# print(type(img_pred))
# print(img_pred)
#
# pred = cv2.imread('./predict/img_pred.png', cv2.IMREAD_GRAYSCALE)
# msk = cv2.imread('../image/train/20_19_mask.tif', cv2.IMREAD_GRAYSCALE)
# figure, axis = plt.subplots(1, 2, figsize=(16, 12))
# axis[0].imshow(pred, cmap='gray')
# axis[1].imshow(msk, cmap='gray')
