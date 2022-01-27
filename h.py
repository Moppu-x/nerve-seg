# coding=gbk
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../image/train/1_1_mask.tif', cv2.IMREAD_GRAYSCALE)
print(img[120])


plt.imshow(img)
plt.show()
