import os, cv2, pickle, matplotlib, platform, imageio, datetime, argparse, time
from keras.layers import Activation, Dense
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from base64 import b64encode
# import imgaug.augmenters as iaa
# import imgaug as ia
from sklearn.model_selection import StratifiedShuffleSplit
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Dropout, Flatten, Dense
import matplotlib.pyplot as plt
from keras import regularizers
# from keras.applications import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
# from keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from sklearn import metrics
from scikitplot.metrics import plot_roc_curve
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, classification_report, confusion_matrix
from keras.models import load_model
from tensorflow.python.client import device_lib
from collections import deque
from sklearn.model_selection import train_test_split
from random import shuffle
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
# import natsort
import random
from sklearn import metrics


### 동영상 폴더
# train_folder = 'D:/abnormal_detection_dataset/UBI_FIGHTS/videos/All/'
### UBI-FIGHTS
# train_folder = 'D:/abnormal_detection_dataset/UBI_FIGHTS/frames1/' #train
# test_folder = 'D:/abnormal_detection_dataset/UBI_FIGHTS/frames/' #test
#
# train_fight_label = train_folder + '1/'          # fight
# train_normal_label = train_folder + '0/'         # normal
# train_fight_png = os.listdir(train_fight_label)
# train_normal_png = os.listdir(train_normal_label)
#
# train_fight_and_normal_label = len(os.listdir(train_fight_label)) + len(os.listdir(train_normal_label))
#
# test_fight_label = test_folder + '1/'
# test_normal_label = test_folder + '0/'
# test_fight_png = os.listdir(test_fight_label)
# test_normal_png = os.listdir(test_normal_label)
#
# test_fight_and_normal_label = len(os.listdir(test_fight_label)) + len(os.listdir(test_normal_label))
# print(test_fight_and_normal_label)   # 42217
#
# total_num = train_fight_and_normal_label + test_fight_and_normal_label

##### RWF-2000
train_folder = 'D:/abnormal_detection_dataset/RWF_2000_Dataset_frame/train/' #train
test_folder = 'D:/abnormal_detection_dataset/RWF_2000_Dataset_frame/val/' #test

train_fight_label = train_folder + 'Fight/'          # fight
train_normal_label = train_folder + 'NonFight/'         # normal
train_fight_png = os.listdir(train_fight_label)
train_normal_png = os.listdir(train_normal_label)

train_fight_and_normal_label = len(os.listdir(train_fight_label)) + len(os.listdir(train_normal_label))

test_fight_label = test_folder + 'Fight/'
test_normal_label = test_folder + 'NonFight/'
test_fight_png = os.listdir(test_fight_label)
test_normal_png = os.listdir(test_normal_label)

test_fight_and_normal_label = len(os.listdir(test_fight_label)) + len(os.listdir(test_normal_label))
print(test_fight_and_normal_label)   # 42217

total_num = train_fight_and_normal_label + test_fight_and_normal_label

# print(len(os.listdir(train_fight_label)))   # 7983
# print(len(os.listdir(train_normal_label)))  # 76825
#
# print(len(os.listdir(test_fight_label)))    # 3953
# print(len(os.listdir(test_normal_label)))   # 38264

print('Train data normal 갯수: ', len(os.listdir(train_normal_label)))
print('Train data fight 갯수: ', len(os.listdir(train_fight_label)))
print('총 Train data 갯수: ', train_fight_and_normal_label)

print('-' * 100)

print('Test data normal 갯수: ', len(os.listdir(test_normal_label)))
print('Test data fight 갯수: ', len(os.listdir(test_fight_label)))
print('총 Test data 갯수: ', test_fight_and_normal_label)
print('-' * 100)

# print(train_fight_and_normal_label)   # 84808
# print(os.listdir(train_fight_label))  # 파일 이름 리스트에 저장
# print(os.listdir(train_fight_label))  # 파일 이름 리스트에 저장
print(os.listdir(train_folder))   # ['fight', 'normal'] >>>> ['1', '0']
test_normal_labeling = []
train_normal_labeling = []

train_fight_labeling = []
test_fight_labeling = []


train_fight_img_count = len(train_fight_png)
train_normal_img_count = len(train_normal_png)

print('train_fight_img_count: ', train_fight_img_count)
print('train_normal_img_count: ', train_normal_img_count)
print('-' * 100)

def train_gen():
    global train_fight_and_normal_label, train_normal_label, train_fight_label, train_fight_png, train_normal_png, train_normal_labeling, train_fight_labeling


    fight_img_count = 0
    normal_img_count = 0


    for _ in range(train_fight_img_count//5+train_normal_img_count//5): #fight와 normal 폴더안의 이미지 갯수만큼 for문 돌림

        random_folder = random.randint(0,1) # 폴더 0,1 중에 랜덤으로 선택

        if random_folder == 1:

            if fight_img_count <= train_fight_img_count//5 -1:

                img0 = cv2.imread(train_fight_label + train_fight_png[fight_img_count])
                img1 = cv2.imread(train_fight_label + train_fight_png[fight_img_count + 1])
                img2 = cv2.imread(train_fight_label + train_fight_png[fight_img_count + 2])
                img3 = cv2.imread(train_fight_label + train_fight_png[fight_img_count + 3])
                img4 = cv2.imread(train_fight_label + train_fight_png[fight_img_count + 4])


                fight_img_count += 5

                train_fight_img = np.concatenate((img0, img1, img2, img3, img4), axis=2, dtype=np.float32)
                train_fight_img = cv2.resize(train_fight_img, (512, 512))
                label = np.array([1])

                yield(train_fight_img, label)

            else:
                continue

        else:

            if normal_img_count <= train_normal_img_count//5 -1:
                img0 = cv2.imread(train_normal_label + train_normal_png[normal_img_count])
                img1 = cv2.imread(train_normal_label + train_normal_png[normal_img_count + 1])
                img2 = cv2.imread(train_normal_label + train_normal_png[normal_img_count + 2])
                img3 = cv2.imread(train_normal_label + train_normal_png[normal_img_count + 3])
                img4 = cv2.imread(train_normal_label + train_normal_png[normal_img_count + 4])

                normal_img_count += 5

                normal_fight_img = np.concatenate((img0, img1, img2, img3, img4), axis=2, dtype=np.float32)
                normal_fight_img = cv2.resize(normal_fight_img, (512, 512))
                label = np.array([0])

                yield (normal_fight_img, label)




    #
    #
    #
    # ################################################## train - fight
    # train_fight_img_list = os.listdir(train_fight_label)
    # train_fight_labeling =[]
    # train_normal_labeling = []
    # for i in range(0, (len(os.listdir(train_fight_label))-4), 5):
    #
    #     img0 = cv2.imread(train_fight_label + train_fight_png[i])
    #     img1 = cv2.imread(train_fight_label + train_fight_png[i + 1])
    #     img2 = cv2.imread(train_fight_label + train_fight_png[i + 2])
    #     img3 = cv2.imread(train_fight_label + train_fight_png[i + 3])
    #     img4 = cv2.imread(train_fight_label + train_fight_png[i + 4])
    #
    #     img0 = cv2.resize(img0, (224, 224))
    #     img1 = cv2.resize(img1, (224, 224))
    #     img2 = cv2.resize(img2, (224, 224))
    #     img3 = cv2.resize(img3, (224, 224))
    #     img4 = cv2.resize(img4, (224, 224))
    #
    #     train_fight_img = np.concatenate((img0, img1, img2, img3, img4), axis=2, dtype=object)
    #     # train_fight_img = train_fight_img/255.
    #     train_fight_img = np.reshape(train_fight_img, (-1, 224, 224, 15))
    #
    #
    #     # train_fight_labeling = np.array(os.listdir(train_folder))
    #     # train_fight_labeling = keras.utils.to_categorical(train_fight_labeling, 2)
    #     # train_fight_labeling = np.array(train_fight_labeling)
    #     # train_fight_labeling = train_fight_labeling.astype(np.int8)
    #     # train_fight_labeling = np.reshape(train_fight_labeling, (1, 1))
    #     train_fight_labeling.append([[1]])
    #
    #
    #     # yield (train_fight_img, train_fight_labeling)
    #
    # ################################################## train - nomal
    # train_normal_img_list = os.listdir(train_normal_label)
    #
    # for i in range(0, (len(os.listdir(train_normal_label))-4), 5):
    #
    #     img0 = cv2.imread(train_normal_label + train_normal_png[i])
    #     img1 = cv2.imread(train_normal_label + train_normal_png[i + 1])
    #     img2 = cv2.imread(train_normal_label + train_normal_png[i + 2])
    #     img3 = cv2.imread(train_normal_label + train_normal_png[i + 3])
    #     img4 = cv2.imread(train_normal_label + train_normal_png[i + 4])
    #
    #     img0 = cv2.resize(img0, (224, 224))
    #     img1 = cv2.resize(img1, (224, 224))
    #     img2 = cv2.resize(img2, (224, 224))
    #     img3 = cv2.resize(img3, (224, 224))
    #     img4 = cv2.resize(img4, (224, 224))
    #
    #     train_normal_img = np.concatenate((img0, img1, img2, img3, img4), axis=2, dtype=object)
    #     # train_normal_img = train_normal_img / 255.
    #     train_normal_img = np.reshape(train_normal_img, (-1, 224, 224, 15))
    #
    #     # train_normal_labeling = np.array(os.listdir(train_folder))
    #     # train_normal_labeling = keras.utils.to_categorical(train_normal_labeling, 2)
    #     # train_normal_labeling = np.array(train_normal_labeling)
    #     # train_normal_labeling = train_normal_labeling.astype(np.int8)
    #     # train_normal_labeling = np.reshape(train_normal_labeling, (1, 1))
    #     train_normal_labeling.append([[0]])
    #
    #
    #
    #
    #
    # yield (train_normal_img, train_normal_labeling), (train_fight_img, train_fight_labeling)


# img0 = cv2.imread(test_fight_label + test_fight_png[0])
# cv2.imshow("test", img0)
# cv2.waitKey(0)
#

test_fight_img_count = len(test_fight_png)
test_noraml_img_count = len(test_normal_png)

print('test_fight_img_count: ', test_fight_img_count)
print('test_noraml_img_count: ', test_noraml_img_count)
print('-' * 100)


def test_gen():


    global test_fight_and_normal_label, test_normal_label, test_fight_label, test_fight_png, test_normal_png, test_fight_labeling,test_normal_labeling

    fight_img_count = 0
    normal_img_count = 0

    for _ in range(test_fight_img_count // 5 + test_noraml_img_count // 5):  # fight와 normal 폴더안의 이미지 갯수만큼 for문 돌림

        random_folder = random.randint(0, 1)  # 폴더 0,1 중에 랜덤으로 선택

        if random_folder == 1:

            if fight_img_count <= test_fight_img_count // 5 - 1:

                img0 = cv2.imread(test_fight_label + test_fight_png[fight_img_count])
                img1 = cv2.imread(test_fight_label + test_fight_png[fight_img_count + 1])
                img2 = cv2.imread(test_fight_label + test_fight_png[fight_img_count + 2])
                img3 = cv2.imread(test_fight_label + test_fight_png[fight_img_count + 3])
                img4 = cv2.imread(test_fight_label + test_fight_png[fight_img_count + 4])
                img5 = cv2.imread(test_fight_label + test_fight_png[fight_img_count + 5])
                # img6 = cv2.imread(test_fight_label + test_fight_png[fight_img_count + 6])
                # img7 = cv2.imread(test_fight_label + test_fight_png[fight_img_count + 7])
                # img8 = cv2.imread(test_fight_label + test_fight_png[fight_img_count + 8])
                # img9 = cv2.imread(test_fight_label + test_fight_png[fight_img_count + 9])
                # img10 = cv2.imread(test_fight_label + test_fight_png[fight_img_count + 10])
                # img11 = cv2.imread(test_fight_label + test_fight_png[fight_img_count + 11])
                # img12 = cv2.imread(test_fight_label + test_fight_png[fight_img_count + 12])
                # img13 = cv2.imread(test_fight_label + test_fight_png[fight_img_count + 13])
                # img14 = cv2.imread(test_fight_label + test_fight_png[fight_img_count + 14])
                # img15 = cv2.imread(test_fight_label + test_fight_png[fight_img_count + 154])

                fight_img_count += 5

                train_fight_img = np.concatenate((img0, img1, img2, img3, img4), axis=2, dtype=np.float32)
                train_fight_img = cv2.resize(train_fight_img, (512, 512))
                label = np.array([1])

                yield (train_fight_img, label)

            else:
                continue

        else:

            if normal_img_count <= test_noraml_img_count // 5 - 1:
                img0 = cv2.imread(test_normal_label + test_normal_png[normal_img_count])
                img1 = cv2.imread(test_normal_label + test_normal_png[normal_img_count + 1])
                img2 = cv2.imread(test_normal_label + test_normal_png[normal_img_count + 2])
                img3 = cv2.imread(test_normal_label + test_normal_png[normal_img_count + 3])
                img4 = cv2.imread(test_normal_label + test_normal_png[normal_img_count + 4])

                normal_img_count += 5

                normal_fight_img = np.concatenate((img0, img1, img2, img3, img4), axis=2, dtype=np.float32)
                normal_fight_img = cv2.resize(normal_fight_img, (512, 512))
                label = np.array([0])

                yield (normal_fight_img, label)


class ChannelAttention(layers.Layer):
    def __init__(self, reduction=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        self.filters = input_shape[-1]
        self.hidden_units = self.filters // self.reduction

        self.dense_1 = layers.Dense(units=self.hidden_units, activation='relu', name='fc1', kernel_initializer='he_normal')
        self.dense_2 = layers.Dense(units=self.filters, activation='relu', name='fc2', kernel_initializer='he_normal')

        self.global_pooling = layers.GlobalAveragePooling2D()

        super(ChannelAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        avg_pool = self.global_pooling(inputs)
        avg_pool = tf.reshape(avg_pool, (-1, 1, 1, self.filters))

        dense_1_out = self.dense_1(avg_pool)
        dense_2_out = self.dense_2(dense_1_out)

        channel_attention = tf.nn.sigmoid(dense_2_out)

        return channel_attention

class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.filters = input_shape[-1]

        self.conv2d = layers.Conv2D(filters=1,
                                    kernel_size=self.kernel_size,
                                    padding='same',
                                    activation='sigmoid',
                                    kernel_initializer='he_normal')

        super(SpatialAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)

        spatial_attention = self.conv2d(max_pool)

        return spatial_attention

class CBAM(layers.Layer):
    def __init__(self, reduction=8, kernel_size=7, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.reduction = reduction
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.channel_attention = ChannelAttention(reduction=self.reduction)
        self.spatial_attention = SpatialAttention(kernel_size=self.kernel_size)

        super(CBAM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        channel_attention = self.channel_attention(inputs)
        spatial_attention = self.spatial_attention(inputs)

        cbam_features = tf.multiply(inputs, channel_attention)
        cbam_features = tf.multiply(cbam_features, spatial_attention)

        return cbam_features

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'reduction': self.reduction,
            'kernel_size': self.kernel_size
        })
        return config

import keras
# is_50 : True --> resnet_50
# is_plain :True --> no skip connection
# def build_resnet_block(input_layer, num_cnn=3, channel=64, block_num=1,is_50 = True,is_plain = False):
#     # 입력 레이어
#     x = input_layer
#     if not is_50:
#     # CNN 레이어
#         for cnn_num in range(num_cnn):
#             identity = x
#             x = keras.layers.Conv2D(
#                 filters=channel,
#                 kernel_size=(3,3),
#                 activation='relu',
#                 kernel_initializer='he_normal',
#                 padding='same',
#                 name=f'block{block_num}_conv{cnn_num}'
#             )(x)
#             x = keras.layers.BatchNormalization()(x)
#             x = keras.layers.Conv2D(
#                 filters=channel,
#                 kernel_size=(3,3),
#                 activation='relu',
#                 kernel_initializer='he_normal',
#                 padding='same',
#                 name=f'block{block_num}_1_conv{cnn_num}'
#             )(x)
#             if not is_plain:
#                 identity_channel = identity.shape.as_list()[-1]
#
#                 if identity_channel != channel:
#                     identity = keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=(1, 1), padding="same")(identity)
#                 # skip connection
#                 x = keras.layers.Add()([x,identity])
#             else:
#                 pass
#     else :
#         identity = x
#         x = keras.layers.Conv2D(
#             filters=channel,
#             kernel_size=(1,1),
#             activation='relu',
#             kernel_initializer='he_normal',
#             padding='same',
#             name=f'block{block_num}_conv{cnn_num}'
#         )(x)
#         x = keras.layers.BatchNormalization()(x)
#         x = keras.layers.Conv2D(
#             filters=channel,
#             kernel_size=(3,3),
#             activation='relu',
#             kernel_initializer='he_normal',
#             padding='same',
#             name=f'block{block_num}_1_conv{cnn_num}'
#         )(x)
#         x = keras.layers.Conv2D(
#             filters=channel * 4,
#             kernel_size=(1,1),
#             activation='relu',
#             kernel_initializer='he_normal',
#             padding='same',
#             name=f'block{block_num}_2_conv{cnn_num}'
#         )(x)
#         if not is_plain:
#             identity_channel = identity.shape.as_list()[-1]
#
#             if identity_channel != channel:
#                 identity = keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=(1, 1), padding="same")(identity)
#             # skip connection
#             x = keras.layers.Add()([x,identity])
#         else:
#             pass
#     #     Max Pooling 레이어
#     # 마지막 블록 뒤에는 pooling을 하지 않음
#     if identity.shape[1] != 1:
#         x = keras.layers.MaxPooling2D(
#             pool_size=(2, 2),
#             strides=2,
#             name=f'block{block_num}_pooling'
#         )(x)
#
#     return x
#
#
# def build_resnet(input_shape=(32, 32, 3),
#                  num_cnn_list=[3, 4, 6, 3],
#                  channel_list=[64, 128, 256, 512],
#                  num_classes=10, is_50=False, is_plain=False):
#     assert len(num_cnn_list) == len(channel_list)  # 모델을 만들기 전에 config list들이 같은 길이인지 확인합니다.
#     if is_50:
#         num_cnn_list = [3, 4, 6, 3]
#         channel_list = [64, 128, 256, 512]
#         num_classes = 10
#
#     input_layer = keras.layers.Input(shape=input_shape)  # input layer를 만들어둡니다.
#     output = input_layer
#     # conv1층
#     output = keras.layers.Conv2D(filters=64,
#                                  kernel_size=(2, 2),
#                                  strides=2,
#                                  padding='valid')(output)
#     output = keras.layers.BatchNormalization()(output)
#
#     # conv2_x pooling
#     output = keras.layers.MaxPooling2D(pool_size=(2, 2),
#                                        strides=2, )(output)
#     # config list들의 길이만큼 반복해서 블록을 생성합니다.
#     for i, (num_cnn, channel) in enumerate(zip(num_cnn_list, channel_list)):
#         output = build_resnet_block(
#             output,
#             num_cnn=num_cnn,
#             channel=channel,
#             block_num=i
#         )
#     output = keras.layers.AveragePooling2D(padding='same')(output)
#     output = keras.layers.Flatten(name='flatten')(output)
#     output = keras.layers.Dense(512, activation='relu', name='fc1')(output)
#     output = keras.layers.Dense(num_classes, activation='softmax', name='predictions')(output)
#
#     model = keras.Model(
#         inputs=input_layer,
#         outputs=output
#     )
#     return model

# resnet_34 = build_resnet(is_50 = False)
# resnet_50 = build_resnet(is_50 = True)
# plain_resnet_34 = build_resnet(is_50 = False, is_plain = True)
# plain_resnet_50 = build_resnet(is_50 = True, is_plain = True)

def identity_block(X, filters, kernel_size):
    X_shortcut = X

    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)

    # Add
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X


def convolutional_block(X, filters, kernel_size):
    X_shortcut = X

    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X)
    X = tf.keras.layers.BatchNormalization()(X)

    X_shortcut = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME')(X_shortcut)
    X_shortcut = tf.keras.layers.BatchNormalization()(X_shortcut)

    # Add
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X


def ResNet50CL(input_shape=(512, 512, 15), classes=10):
    X_input = tf.keras.layers.Input(input_shape)
    X = X_input

    X = convolutional_block(X, 64, (3, 3))  # conv
    X = identity_block(X, 64, (3, 3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)

    X = convolutional_block(X, 128, (3, 3))  # 64->128, use conv block
    X = identity_block(X, 128, (3, 3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)

    X = convolutional_block(X, 256, (3, 3))  # 128->256, use conv block
    X = identity_block(X, 256, (3, 3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)

    X = convolutional_block(X, 512, (3, 3))  # 256->512, use conv block
    X = identity_block(X, 512, (3, 3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)

    X = tf.keras.layers.GlobalAveragePooling2D()(X)
    X = tf.keras.layers.Dense(1, activation='sigmoid')(X)  # ouput layer (10 class)

    model = tf.keras.models.Model(inputs=X_input, outputs=X, name="ResNet50CL")

    return model


def ResNet50C(input_shape=(512, 512, 15), classes=2):
    X_input = tf.keras.layers.Input(input_shape)
    X = X_input

    X = tf.keras.layers.Conv2D(64, (3, 3), padding='SAME')(X)
    X = CBAM()(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = convolutional_block(X, 64, (3, 3))  # use conv block (?)
    X = CBAM()(X)
    X = identity_block(X, 64, (3, 3))
    X = identity_block(X, 64, (3, 3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)

    X = convolutional_block(X, 128, (3, 3))  # 64->128, use conv block
    X = CBAM()(X)
    X = identity_block(X, 128, (3, 3))
    X = identity_block(X, 128, (3, 3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)

    X = convolutional_block(X, 256, (3, 3))  # 128->256, use conv block
    X = CBAM()(X)
    X = identity_block(X, 256, (3, 3))
    X = identity_block(X, 256, (3, 3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)

    X = convolutional_block(X, 512, (3, 3))  # 256->512, use conv block
    X = CBAM()(X)
    X = identity_block(X, 512, (3, 3))
    X = identity_block(X, 512, (3, 3))
    X = tf.keras.layers.MaxPooling2D(2, 2, padding='SAME')(X)

    X = tf.keras.layers.GlobalAveragePooling2D()(X)
    X = tf.keras.layers.Dense(1, activation='sigmoid')(X)  # ouput layer (10 class)

    model = tf.keras.models.Model(inputs=X_input, outputs=X, name="ResNet50C")

    return model

# ################################################## test - fight
#     test_fight_img_list = os.listdir(test_fight_label)
#     test_fight_labeling = []
#     test_normal_labeling = []
#     for i in range(0, (len(os.listdir(test_fight_label))-4), 5):
#         img0 = cv2.imread(test_fight_label + test_fight_png[i])
#         img1 = cv2.imread(test_fight_label + test_fight_png[i + 1])
#         img2 = cv2.imread(test_fight_label + test_fight_png[i + 2])
#         img3 = cv2.imread(test_fight_label + test_fight_png[i + 3])
#         img4 = cv2.imread(test_fight_label + test_fight_png[i + 4])
#
#         img0 = cv2.resize(img0, (224, 224))
#         img1 = cv2.resize(img1, (224, 224))
#         img2 = cv2.resize(img2, (224, 224))
#         img3 = cv2.resize(img3, (224, 224))
#         img4 = cv2.resize(img4, (224, 224))
#
#         test_fight_img = np.concatenate((img0, img1, img2, img3, img4), axis=2, dtype=object)
#         # test_fight_img = test_fight_img / 255.
#         test_fight_img = np.reshape(test_fight_img, (-1, 224, 224, 15))
#
#         # test_fight_labeling = np.array(os.listdir(test_folder))
#         # test_fight_labeling = keras.utils.to_categorical(test_fight_labeling, 2)
#         # test_fight_labeling = np.array(test_fight_labeling)
#         # test_fight_labeling = test_fight_labeling.astype(np.int8)
#         # test_fight_labeling = np.reshape(test_fight_labeling, (1, 1))
#         test_fight_labeling.append([[1]])
#
#
#
#
#
#         # yield (test_fight_img, test_fight_labeling)
#
# ################################################## test - normal
#     test_normal_img_list = os.listdir(test_normal_label)
#
#     for i in range(0, (len(os.listdir(test_normal_label))-4), 5):
#         img0 = cv2.imread(test_normal_label + test_normal_png[i])
#         img1 = cv2.imread(test_normal_label + test_normal_png[i + 1])
#         img2 = cv2.imread(test_normal_label + test_normal_png[i + 2])
#         img3 = cv2.imread(test_normal_label + test_normal_png[i + 3])
#         img4 = cv2.imread(test_normal_label + test_normal_png[i + 4])
#
#         img0 = cv2.resize(img0, (224, 224))
#         img1 = cv2.resize(img1, (224, 224))
#         img2 = cv2.resize(img2, (224, 224))
#         img3 = cv2.resize(img3, (224, 224))
#         img4 = cv2.resize(img4, (224, 224))
#
#         test_normal_img = np.concatenate((img0, img1, img2, img3, img4), axis=2, dtype=object)
#         # test_normal_img = test_normal_img / 255.
#         test_normal_img = np.reshape(test_normal_img, (-1, 224, 224, 15))
#
#
#         # test_normal_labeling = np.array(os.listdir(test_folder))
#         # test_normal_labeling = keras.utils.to_categorical(test_normal_labeling, 2)
#         # test_normal_labeling = np.array(test_normal_labeling)
#         # test_normal_labeling = test_normal_labeling.astype(np.int8)
#         # test_normal_labeling = np.reshape(test_normal_labeling, (1, 1))
#         test_normal_labeling.append([[0]])
#
#
#     yield (test_normal_img, test_normal_labeling), (test_fight_img, test_fight_labeling)

# test_normal_labeling = np.array(test_normal_labeling)
# train_normal_labeling = np.array(train_normal_labeling)
#
# train_fight_labeling = np.array(train_fight_labeling)
# test_fight_labeling = np.array(test_fight_labeling)


# train_dataset = tf.data.Dataset.from_generator(train_gen, (tf.float32, tf.float16), ((360, 640, 15), (1)))
train_dataset = tf.data.Dataset.from_generator(train_gen, (tf.float32, tf.float16), ((512, 512, 15), (1)))
train_dataset = train_dataset.batch(4)

# test_dataset = tf.data.Dataset.from_generator(test_gen, (tf.float32, tf.float16), ((360, 640, 15), (1)))
test_dataset = tf.data.Dataset.from_generator(test_gen, (tf.float32, tf.float16), ((512, 512, 15), (1)))
test_dataset = test_dataset.batch(4)

######## 모델 생성
# model = Sequential()  ### CNN-LSTM
# input_shape = (512, 512, 15)  # (224, 224, 3), (360, 640, 15)
# base_model = ResNet50(include_top=False,  weights=None, input_shape=input_shape) # input_shape=input_shape,
# base_model.summary()
# # for layer in base_model.layer:
#     print(layer[0])
# print(base_model[0])
# base_model.summary()
# config = resnet50_.get_config()
# config["layers"][0]["config"]["batch_input_shape"] = (None, img_height, img_width, input_channel)
# base_model.trainable = False
# x = base_model.output
# x = input_shape
# x = CBAM(x)
# x = BatchNormalization(x)
# x = CBAM(x)
# x = BatchNormalization(x)
# x = CBAM(x)
# x = BatchNormalization(x)
# x = CBAM(x)
# x = BatchNormalization(x)

# x = Dense(1024)(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = Dense(512)(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = Dense(1)(x)
# x = BatchNormalization()(x)
# x = Dropout(0.3)
# x = Activation('sigmoid')(x)

# print(ABCSD)
# 원래 모델 시작 # 원래 모델 시작 # 원래 모델 시작  # 원래 모델 시작 # 원래 모델 시작 # 원래 모델 시작 # 원래 모델 시작
# model.add(layers.Input(shape=(input_shape)))
# model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D(strides=(2, 2)))

# model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(0.2))
# model.add(layers.MaxPooling2D(strides=(2, 2)))

# model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(0.2))
# model.add(layers.MaxPooling2D(strides=(2, 2)))

# model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(0.2))
# model.add(layers.MaxPooling2D(strides=(2, 2)))

# model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(0.2))

# model.add(layers.TimeDistributed(Flatten()))

# model.add(layers.LSTM(1024))

# model.add(layers.Dense(1024, activation='relu'))
# model.add(layers.Dropout(0.2))

# model.add(layers.Dense(512, activation='relu'))

# model.add(layers.Dense(1, activation='sigmoid'))
# model = Model(inputs=model.input, outputs=model.output)
# 원래 모델 끝# 원래 모델 끝# 원래 모델 끝# 원래 모델 끝# 원래 모델 끝# 원래 모델 끝# 원래 모델 끝# 원래 모델 끝# 원래 모델 끝# 원래 모델 끝# 원래 모델 끝# 원래 모델 끝
# model = Model(base_model.input, x)
model = ResNet50CL()
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
# for layer in model.layers:
#     layer.trainable = True

EPOCH = 100
BATCH_SIZE = 16

# earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
#                               patience=10,
#                              )
#
# reduceLR = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor='val_accuracy',
#     factor=0.5,
#     patience=4,
# )

### 모델 컴파일
print('모델 컴파일 중')
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

### 콜백 함수 정의
patience = 10

start_lr = 0.00001
min_lr = 0.00001
max_lr = 0.00005

batch_size = 32  # 128

rampup_epochs = 5
sustain_epochs = 0
exp_decay = 0.0001


def lrfn(epoch):
    if epoch < rampup_epochs:
        return (max_lr - start_lr) / rampup_epochs * epoch + start_lr
    elif epoch < rampup_epochs + sustain_epochs:
        return max_lr
    else:
        return (max_lr - min_lr) * exp_decay ** (epoch - rampup_epochs - sustain_epochs) + min_lr


class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if ((logs.get('accuracy') >= 0.999)):
            print("\nLimits Reached cancelling training!")
            self.model.stop_training = True


end_callback = myCallback()
lr_callback = LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=False)
early_stopping = EarlyStopping(patience=patience, monitor='val_loss', mode='min', restore_best_weights=True, verbose=1, min_delta=.00075)
lr_plat = ReduceLROnPlateau(patience=2, mode='min')
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_filepath = './cnn_lstm_3ch_attention_RFW_checkpoint.h5'
model_checkpoints = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='val_loss',
                                    mode='min', verbose=1, save_best_only=True)
callbacks = [model_checkpoints, early_stopping]

### 모델 핏
# train_gen = ImageDataGenerator(rescale=1./255)
# test_gen = ImageDataGenerator(rescale=1./255)
#
# train_flow_gen = train_gen.flow_from_directory(directory=frame_folder, target_size=(224, 224),
#                                                class_mode='binary', batch_size=batch_size,
#                                                shuffle=False)
#
# test_flow_gen = test_gen.flow_from_directory(directory=test_folder, target_size=(224, 224),
#                                              class_mode='binary', batch_size=batch_size,
#                                              shuffle=False)


history = model.fit(train_dataset, epochs=100, callbacks=callbacks, validation_data=test_dataset, batch_size=batch_size)
print(history.history.keys())
print('model.metrics_names: ', model.metrics_names)

print(history.history)
model.load_weights(checkpoint_filepath)

# print(history.history)
#
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Train and Validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Train and Validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

score = model.evaluate(test_dataset, batch_size=batch_size , verbose=1)
print('accuray: ', score[1], 'loss: ', score[0])


# evaluate the network
print("Evaluating network...")

x_test_img = []
y_test_list = []
Y_list = []
preds_list = []
for x_test, y_test in test_gen():
    # x_test_img.append(x_test)
    # print('x_test: ', '\n', np.array(x_test))
    # print('y_test: ', '\n', np.array(y_test))
    x_test = x_test.reshape(-1, x_test.shape[0], x_test.shape[1], 15) / 255.

    Y_list.append(y_test)
    y_predict = model.predict(x_test)
    preds = y_predict > 0.5
    preds_list.append(preds)
    y_test_list.append(y_predict)






# Y_list = np.reshape(Y_list, (1689, 1, 1))
Y_list = np.reshape(Y_list, (-1, 1))
y_test_list = np.reshape(y_test_list, (-1, 1))

# Y_list = np.array(Y_list)                                   # [[[0]]                                 실제 라벨
# Y_list = np.array(Y_list).astype(int) #                   # [[[0]]

# y_test_list = np.array(y_test_list)                         # [[2.36555678e-07]]] >> [[[0.130483  ]]  예측
# y_test_list = y_test_list.astype(np.int32)                # [[[0]]

# preds_list = np.array(preds_list)                           # [[False]]]
# preds_list = np.array(preds_list).astype(int)      # [[[0.]]
#
print('Y_list.type: ', type(Y_list))
print('y_test_list.type', type(y_test_list))
print('preds_list.type', type(preds_list))

print('Y_list.shape: ', str(np.shape(Y_list)))
print('y_test_list.shape: ', str(np.shape(y_test_list)))
print('preds.shape: ', str(np.shape(preds_list)))


# fpr, tpr, threshold = metrics.roc_curve(Y_list, y_test_list)
fpr, tpr, threshold = roc_curve(Y_list, y_test_list)
print('metrics.auc(fpr, tpr): ', metrics.auc(fpr, tpr))
metrics.auc(fpr, tpr)
plt.plot(fpr, tpr)
plt.show()


fprs, tprs, thresholds = roc_curve(Y_list, y_test_list)
precisions, recalls, thresholds = roc_curve(Y_list, y_test_list)
plt.figure(figsize=(15, 5))
plt.plot([0, 1], [0, 1], label='STR')
# ROC
plt.plot(fprs, tprs, label='ROC')
plt.title('ROC curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.grid()
plt.show()
print('roc auc value: {}'.format(roc_auc_score(Y_list, y_test_list)))


false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_list, y_test_list)

roc_auc = metrics.roc_auc_score(Y_list, y_test_list)

plt.title('Receiver Operating Characteristic(ROC)')
plt.xlabel('False Positive Rate(1 - Specificity)')
plt.ylabel('True Positive Rate(Sensitivity)')

plt.plot(false_positive_rate, true_positive_rate, 'b', label='Model (AUC = %0.2f)'% roc_auc)
plt.plot([0, 1], [1, 1], 'y--')
plt.plot([0, 1], [0, 1], 'r--')

plt.legend(loc='lower right')
plt.show()

print('roc_auc: ', roc_auc)

auc = metrics.roc_auc_score(Y_list, y_test_list)
print('auc: ', auc)

##################################### 절취선 #####################################
### 12
corr_pred = metrics.confusion_matrix(Y_list, y_test_list.astype(int))

n_correct = int((corr_pred[0][0] + corr_pred[1][1]))
print('> Correct Predictions:', n_correct)
n_wrongs = int((corr_pred[0][1] + (corr_pred[1][0])))
print('> Wrong Predictions:', n_wrongs)

sns.heatmap(corr_pred,annot=True, fmt="d",cmap="Blues")
plt.show()
##################################### 절취선 #####################################

preds_1d = y_test_list.flatten()
pred_class = np.where(preds_1d > 0.5, 1, 0)
print(classification_report(Y_list, pred_class))

# print(classification_report(Y_list, y_test_list, target_names=["0", "1"]))


# for x_data, y_data in train_gen():
#      for idx, image in enumerate(x_data):
#          image = Image.open(image)
         # cv2.resize(img, (512, 512))
         # cv2.imshow('1', img)
         # cv2.waitKey()