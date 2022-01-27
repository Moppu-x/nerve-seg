# coding=gbk

from keras.models import Model, Input
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import adam_v2
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

from data import data_gen

img_width = 128
img_height = 128


# dice系数
def dice_coef(y_true, y_pred):
    # 转一维
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)
    # 计算两者点乘并结果相加
    prod_sum = K.sum(y_pred * y_true)
    # 计算dice系数
    coef = (2.0 * prod_sum + 1) / (K.sum(y_pred) + K.sum(y_true) + 1)

    return coef


# 损失函数dice loss
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


# Unet网络结构定义
def unet():
    # 输入,单通道
    inputs = Input(shape=(img_height, img_width, 1))
    # 下采样部分(特征提取)
    # 第一层， 32个3x3卷积核，2x2最大值池化
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # 第二层， 64个3x3卷积核，2x2最大值池化
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # 第三层，128个3x3卷积核，2x2最大值池化
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # 第四层，256个3x3卷积核，2x2最大值池化
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    # 中间
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)
    # 上采样部分(特征融合)
    # 2x2卷积，并与对等层的上采样卷积拼接
    # 拼接后再进行两次卷积操作
    up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(drop5)
    up6 = concatenate([up6, drop4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    # 最后再进行一次1x1卷积
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=adam_v2.Adam(learning_rate=0.00001), loss=dice_loss, metrics=[dice_coef])
    model.summary()
    return model


# 训练
def train(train_image, train_mask):
    # 新建模型
    model = unet()
    # 模型保存点
    chkpt = ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True)
    # 拟合
    history = model.fit(train_image, train_mask, batch_size=16, epochs=50, verbose=1,
                        validation_split=0.2, callbacks=[chkpt], shuffle=True)
    print('模型训练完毕')
    with open('log.txt', 'wb') as txt:
        pickle.dump(history.history, txt)
    print('开始绘制曲线')
    plt.figure(figsize=(16, 16))
    plt.title('学习曲线')
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(labels=['loss', 'val_loss'], loc='best')
    plt.show()


if __name__ == '__main__':
    # 读取npy文件，获取数据
    image, mask = data_gen()
    # print(image.shape)
    train_image = image[0: 4000]
    train_mask = mask[0: 4000]

    # 训练
    # train(image, mask)
    train(train_image, train_mask)
