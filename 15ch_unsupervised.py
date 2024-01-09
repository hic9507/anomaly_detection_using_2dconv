import os, cv2, pickle, matplotlib, platform, imageio, datetime, argparse, time
import keras.utils
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from base64 import b64encode
import imgaug.augmenters as iaa
import imgaug as ia
from sklearn.model_selection import StratifiedShuffleSplit
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Dropout, Flatten, Dense
import matplotlib.pyplot as plt
from keras import regularizers
from keras.applications import MobileNetV2
from keras.applications import ResNet50, ResNet101, ResNet152
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
import natsort
import random
from sklearn import metrics
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, LSTM, Reshape
from keras.layers import Layer, TimeDistributed


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

        if random_folder == 0:

            if normal_img_count <= train_normal_img_count // 5 - 1:
                img0 = cv2.imread(train_normal_label + train_normal_png[normal_img_count])
                img1 = cv2.imread(train_normal_label + train_normal_png[normal_img_count + 1])
                img2 = cv2.imread(train_normal_label + train_normal_png[normal_img_count + 2])
                img3 = cv2.imread(train_normal_label + train_normal_png[normal_img_count + 3])
                img4 = cv2.imread(train_normal_label + train_normal_png[normal_img_count + 4])


                normal_img_count += 5

                train_normal_img = np.concatenate((img0, img1, img2, img3, img4), axis=2, dtype=np.float32)
                train_normal_img = cv2.resize(train_normal_img, (512, 512))
                # label = np.array([1])
                label = np.array([0])

                yield(train_normal_img, label)

            else:
                continue

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

    normal_img_count = 0

    for _ in range(test_fight_img_count // 5 + test_noraml_img_count // 5):  # fight와 normal 폴더안의 이미지 갯수만큼 for문 돌림

        random_folder = random.randint(0, 1)  # 폴더 0,1 중에 랜덤으로 선택

        if random_folder == 0:

            if normal_img_count <= test_noraml_img_count // 5 - 1:

                img0 = cv2.imread(test_normal_label + test_normal_png[normal_img_count])
                img1 = cv2.imread(test_normal_label + test_normal_png[normal_img_count + 1])
                img2 = cv2.imread(test_normal_label + test_normal_png[normal_img_count + 2])
                img3 = cv2.imread(test_normal_label + test_normal_png[normal_img_count + 3])
                img4 = cv2.imread(test_normal_label + test_normal_png[normal_img_count + 4])

                normal_img_count += 5

                train_normal_img = np.concatenate((img0, img1, img2, img3, img4), axis=2, dtype=np.float32)
                train_normal_img = cv2.resize(train_normal_img, (512, 512))
                label = np.array([0])

                yield (train_normal_img, label)

            else:
                continue



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
input_shape = (512, 512, 15)  # (224, 224, 3), (360, 640, 15)

# model.add(layers.Input(shape=(input_shape)))
# model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D(strides=(2, 2)))
#
# model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(0.2))
# model.add(layers.MaxPooling2D(strides=(2, 2)))
#
# model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(0.2))
# model.add(layers.MaxPooling2D(strides=(2, 2)))
#
# model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(0.2))
# model.add(layers.MaxPooling2D(strides=(2, 2)))
#
# model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(0.2))
#
# model.add(layers.TimeDistributed(Flatten()))
#
# model.add(layers.LSTM(1024))
#
# model.add(layers.Dense(1024, activation='relu'))
# model.add(layers.Dropout(0.2))
#
# model.add(layers.Dense(512, activation='relu'))
#
# model.add(layers.Dense(1, activation='sigmoid'))
class CBAM(Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.maxpool = tf.keras.layers.GlobalMaxPooling2D()
        self.reshape = tf.keras.layers.Reshape((1, 1, self.channels))
        self.fc1 = tf.keras.layers.Dense(units=self.channels // self.reduction_ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
        self.fc2 = tf.keras.layers.Dense(units=self.channels, activation='relu', kernel_initializer='he_normal', use_bias=True)
        self.fc3 = tf.keras.layers.Dense(units=self.channels, activation=None, kernel_initializer='he_normal', use_bias=True)
        self.softmax = tf.keras.layers.Activation('softmax')

        self.conv_after_concat = tf.keras.layers.Conv2D(filters=1, kernel_size=7, padding='same', kernel_initializer='he_normal', use_bias=False)

    def call(self, inputs):
        avgpool = self.avgpool(inputs)
        maxpool = self.maxpool(inputs)
        avg_out = self.fc2(self.fc1(self.reshape(avgpool)))
        max_out = self.fc2(self.fc1(self.reshape(maxpool)))
        att = self.softmax(self.fc3(tf.keras.layers.concatenate([avg_out, max_out])))
        att = tf.keras.layers.Reshape((1, 1, self.channels))(att)
        out = inputs * att
        out = self.conv_after_concat(out)
        return out


class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.f = tf.keras.layers.Conv2D(filters=self.channels // 8, kernel_size=1, strides=(1, 1), padding='same', kernel_initializer='he_normal')
        self.g = tf.keras.layers.Conv2D(filters=self.channels // 8, kernel_size=1, strides=(1, 1), padding='same', kernel_initializer='he_normal')
        self.h = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=1, strides=(1, 1), padding='same', kernel_initializer='he_normal')

    def call(self, inputs):
        f = self.f(inputs)
        g = self.g(inputs)
        h = self.h(inputs)
        shape_f = tf.shape(f)
        shape_g = tf.shape(g)
        shape_h = tf.shape(h)
        # Flatten f, g, h
        flatten_f = tf.reshape(f, [shape_f[0], shape_f[1]*shape_f[2], shape_f[3]])
        flatten_g = tf.reshape(g, [shape_g[0], shape_g[1]*shape_g[2], shape_g[3]])
        flatten_h = tf.reshape(h, [shape_h[0], shape_h[1]*shape_h[2], shape_h[3]])
        # Calculate attention map
        attention = tf.matmul(flatten_g, flatten_f, transpose_b=True)
        attention = tf.nn.softmax(attention)
        # Attend to values
        out = tf.matmul(attention, flatten_h)
        return out


def create_model():
    inputs = Input(shape=(512, 512, 15))

    # 15-layer CNN
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    print(x.shape)
    x = CBAM()(x)  # or SelfAttention()(x)
    print(x.shape)
    print('=' * 50)

    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    print(x.shape)
    x = BatchNormalization()(x)
    print(x.shape)
    x = MaxPooling2D((2, 2))(x)
    print(x.shape)
    print('=' * 50)

    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    print(x.shape)
    x = CBAM()(x)  # or SelfAttention()(x)
    print(x.shape)
    print('=' * 50)

    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    print(x.shape)
    x = BatchNormalization()(x)
    print(x.shape)
    x = MaxPooling2D((2, 2))(x)
    print(x.shape)
    print('=' * 50)

    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    print(x.shape)
    x = CBAM()(x)  # or SelfAttention()(x)
    print(x.shape)
    print('=' * 50)

    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    print(x.shape)
    print('=' * 50)

    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    print(x.shape)
    x = BatchNormalization()(x)
    print(x.shape)
    x = MaxPooling2D((2, 2))(x)
    print(x.shape)
    print('=' * 50)

    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    print(x.shape)
    x = CBAM()(x)  # or SelfAttention()(x)
    print(x.shape)
    print('=' * 50)

    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
    print(x.shape)
    x = BatchNormalization()(x)
    print(x.shape)

    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
    print(x.shape)
    x = BatchNormalization()(x)
    print(x.shape)
    print('=' * 50)

    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
    print(x.shape)
    x = BatchNormalization()(x)
    print(x.shape)
    x = MaxPooling2D((2, 2))(x)
    print(x.shape)
    print('=' * 50)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    print(x.shape)
    x = CBAM()(x)  # or SelfAttention()(x)
    print(x.shape)
    print('=' * 50)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    print(x.shape)
    x = BatchNormalization()(x)
    print(x.shape)
    print('=' * 50)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    print(x.shape)
    x = MaxPooling2D((2, 2))(x)
    print(x.shape)
    print('=' * 50)

    # x = TimeDistributed(Flatten())(x)
    # x = Flatten()(x)
    print(x.shape)
    print('=' * 50)
    # x = np.expand_dims(x, axis=2)
    # x = np.expand_dims(x, axis=0)
    x = TimeDistributed(LSTM(512))(x)

    x = LSTM(256)(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)

    # x = Reshape((1, 64))(x)
    # x = Dropout(0.2)(x)

    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model
model = create_model()

for layer in model.layers:
    layer.trainable = True

### 모델 컴파일
print('모델 컴파일 중')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

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
checkpoint_filepath = './15ch_semisupervised_withcbam.h5'
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

model.load_weights(checkpoint_filepath)

# print(history.history)
print('history.history.keys(): ', history.history.keys())
print('history.history: ', history.history)
print('model.metrics_names: ', model.metrics_names)
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

cnt = 0
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

    cnt += 1

    if cnt == 20:
        break






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