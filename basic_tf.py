# import os
# import cv2
# import tensorflow as tf
# import numpy as np
# import keras
#
# ### 동영상 폴더
# train_folder = 'D:/abnormal_detection_dataset/UBI_FIGHTS/videos/All/'
#
# ### 프레임 추출
# # for file in train_folder:
# video = cv2.VideoCapture(train_folder)
# for i in train_folder:
#     if i.isOpened():
#         print('open: ', i)
# ### normal/abnormal 분리
# # for file_name in train_folder:
# #     values = []
# #     labels = []
# #
# #     if file_name[0:2] == 'F_':
# #         values.append(file_name)
# #         labels.append([1, 0])
# #
# #     elif file_name[0:2] == 'N_':
# #         values.append(file_name)
# #         labels.append([0, 1])
#################################################################### 절취선 ############################

import os
import numpy as np
import cv2
import pickle
import matplotlib
import platform
from tqdm import tqdm
from IPython.display import clear_output
import tensorflow as tf
from IPython.display import HTML
from base64 import b64encode
import imageio
import imgaug.augmenters as iaa
import imgaug as ia
from sklearn.model_selection import StratifiedShuffleSplit
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Dropout,Flatten,Dense
import matplotlib.pyplot as plt
from keras import regularizers
from keras.applications import MobileNetV2
from keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.utils import to_binary
import datetime
import seaborn as sns
from sklearn import metrics
from scikitplot.metrics import plot_roc_curve
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, classification_report, confusion_matrix
import argparse
import time
from keras.models import load_model
from tensorflow.python.client import device_lib
from collections import deque
from sklearn.model_selection import train_test_split
from random import shuffle
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.utils import plot_model

### 동영상 폴더
train_folder = 'D:/abnormal_detection_dataset/UBI_FIGHTS/videos/All/'
frame_folder = 'D:/abnormal_detection_dataset/UBI_FIGHTS/frames1/'
test_folder = 'D:/abnormal_detection_dataset/UBI_FIGHTS/frames/'

### 프레임 추출
# frame_list = []
# cnt = 0
# for file in os.listdir(train_folder):
#
#     video = cv2.VideoCapture(train_folder+file)
#     ret, images = video.read()
#     cnt = 0
#
#     while True:
#         ret, images = video.read()
#
#         if not ret:
#             break
#
#         if int(video.get(1)) % 200 == 0:
#             cv2.imwrite(frame_folder + file[:-4] + '__' + str(cnt) + '.png', images)
#             print('Saved frame number: ', str(int(video.get(1))))
#         cnt += 1
#     video.release()

### normal/abnormal 분리
frame_list = []
label_list = []
name_list = []
# for file in os.listdir(frame_folder):
#     image = cv2.imread(frame_folder+file)
#     image = cv2.resize(image, (224, 224))
#    # frame_list.append([image, file])
#     frame_list.append(file)
#     # print(frame_list)
# # print('-' * 100)
#     for i in range(len(frame_list)):
#         # if frame_list[0][1][0:2] == 'F_':
#         if file[0:2] == 'F_':
#             class_num = 1
#             label_list.append(class_num)
#           #  name_list.append(image)
#         elif file[0:2] == 'N_':
#             class_num = 0
#             label_list.append(class_num)
            #name_list.append(image)

# merge = list(zip(name_list, label_list)) ## frame_list[0][1] > frame_list[0][0]으로 바꿔야할듯
# shuffle(merge)
# name_list, label_list = zip(*merge)
# print(zip(*merge))
print('-' * 100)
    # print(frame_list)
    # print('-' * 100)
    # print(frame_list[0])
    # print('-' * 100)
    # print(file_name)
    # print('-' * 100)
    # print(image)
    # print('-' * 100)
    # print('len(frame_list): ', len(frame_list))
    # # print('frame_list[0][1]: ', frame_list[0][1]) # frame_list[0][1]:  F_0_1_0_0_0__0.png
    # print('frame_list[0][1][0:2] :', frame_list[0][1][0:2])

    # print(frame_list[1][0:2]) # 걍 오류
    # break

##################  이 아래로는 원래 했던거#################
values = []
labels = []

# for current_dir, dir_names, file_names in os.walk(frame_folder):
#     # for file_name in frame_folder:
#     for file_name in file_names:
#
#
#         if file_name[0:2] == 'F_':
#             values.append(file_name)
#             labels.append([1, 0])
#
#         elif file_name[0:2] == 'N_':
#             values.append(file_name)
#             labels.append([0, 1])
#
# merge = list(zip(values, labels))
# shuffle(merge)
#
# values, labels = zip(*merge)
################## 이 위로는 원래 했던거#################

### 모델 생성
model = Sequential()
input_tensor = Input(shape=(224, 224, 3))
input_shape = (224, 224, 3)  # (128, 128, 3)

model.add(layers.Input(shape=(input_shape)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(strides=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling2D(strides=(2, 2)))

model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling2D(strides=(2, 2)))

model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling2D(strides=(2, 2)))

model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.Dropout(0.2))

model.add(layers.TimeDistributed(Flatten()))

model.add(layers.LSTM(1024))

model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model = Model(inputs=model.input, outputs=model.output)

plot_model(model, to_file='./model.png')
plot_model(model, to_file='./model_shapes.png', show_shapes=True)

# model.add(layers.LSTM(1024, input_shape=(None, 224, 224, 3)))
# model.add(layers.Dropout(0.2))

# model.add(layers.LSTM(512, input_shape=(224, 224, 3)))
# model.add(layers.Dropout(0.2))


# rnn_model.add(layers.Dense(1024, activation='relu'))
# rnn_model.add(layers.Dropout(0.2))

# rnn_model.add(layers.Dense(512, activation='relu'))


# rnn_model.add(layers.Dense(1, activation='sigmoid'))
# rnn_modelmodel = Model(inputs=model.input, outputs=rnn_model.output)


# model.add(layers.Dense(1024, activation='relu'))
# model.add(layers.Dropout(0.2))
#
# model.add(layers.Dense(512, activation='relu'))
#
# model.add(layers.Dense(1, activation='sigmoid'))
# model = Model(inputs=model.input, outputs=model.output)

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

batch_size = 16  # 128

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

checkpoint_filepath = './checkpoint.h5'

model_checkpoints = ModelCheckpoint(filepath=checkpoint_filepath,
                                    save_weights_only=True,
                                    monitor='val_loss',
                                    mode='min',
                                    verbose=1,
                                    save_best_only=True)

callbacks = [model_checkpoints, early_stopping]

### 모델 핏
# n_sample = int(len(frame_folder))
# n_train = int(n_sample * 0.8)
# n_test = n_sample - n_train
# frame_list[0][0], label_list = np.array(frame_list[0][0]), np.array(label_list)

# train_generator = data_generator.flow_from_directory(frame_folder, target_size=(224, 224), batch_size=batch_size, class_mode='binary')
# x_train, y_train = train_generator.next()
# print(x_train[0].shape)
# plt.show(x_train[0])
# plt.show()
#print('원래 label_list: ', label_list)
#print('원래 name_list: ', name_list[0])
name_list = np.array(name_list).reshape(-1, 224*224*3)

train_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_flow_gen = train_gen.flow_from_directory(directory=frame_folder, target_size=(224, 224),
                                               class_mode='binary', batch_size=batch_size,
                                               shuffle=True)

test_flow_gen = test_gen.flow_from_directory(directory=test_folder, target_size=(224, 224),
                                             class_mode='binary', batch_size=batch_size,
                                             shuffle=False)

label_list = np.array(label_list)
print('label_list: ', label_list)
# label_list = [label_list]
# label_list = to_categorical(label_list)
print(name_list.shape)
print('아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아')
print(label_list.shape)
print('아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아')
#label_list = np.expand_dims(label_list, axis=(0))
# label_list = tf.one_hot(label_list, depth=1)

# stratified_sample = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=73)

# for train_idx, test_idx in stratified_sample.split(name_list, label_list):
# for train_idx, test_idx in stratified_sample.split(name_list, label_list):
#     x_train, x_test = name_list[train_idx], name_list[test_idx]
#     y_train, y_test = label_list[train_idx], label_list[test_idx]

#x_train_nn = x_train.reshape(-1, 1) / 255
# x_train_nn = x_train.reshape(-1, 224, 224, 3) / 255
#x_test_nn = x_test.reshape(-1, 1) / 255
# x_test_nn = x_test.reshape(-1, 224, 224, 3) / 255
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# y_train = y_train.reshape()



# y_train = to_binary(y_train)
# print('x_train_nn.shape: ', x_train_nn.shape)
# print('x_test_nn.shape: ', x_test_nn.shape)
# print('아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아')
# print('y_train.shape: ', y_train.shape)
# print('아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아')
# # y_test = to_categorical(y_test)
# print('아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아')
# print('y_test.shape: ', y_test.shape)
# print('아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아아')
# # y_train = tf.one_hot(y_train, depth=1)
# # y_test = tf.one_hot(y_test, depth=1)
# #y_train = y_train.reshape(-1, 1) / 255
# #y_test = y_test.reshape(-1, 1) / 255
# # print(x_train_nn)
# print('시시시시시시ㅣ시ㅣㅣㅣㅣㅣㅣ')
# print(x_test_nn[0])
# print('m'*100)
# print(y_train)
# print('n'*100)
# print(y_test)
# print('o'*100)
# print('x_train_nn.shape', x_train_nn.shape)
# print('x_test_nn.shape: ', x_test_nn.shape)
# print('y_train.shape: ', y_train.shape)
# print('y_test.shape: ', y_test.shape)
# y_train = tf.one_hot(y_train)
# y_test = tf.one_hot(y_test)

# print(a)

# x_train, x_test, y_train, y_test = train_test_split(frame_list[0][0], label_list, test_size=0.2, shuffle=True, random_state=34)

# x_train = np.array(x_train)
#y_train = np.array(y_train)
# x_test = np.array(x_test)
#y_test = np.array(y_test)
# x_train = np.expand_dims(224, 224, 3)

# x_train = np.reshape(x_train, (224, 224, 3))
# x_test = np.reshape(x_test, (224, 224, 3))


# print('len(x_train): ', len(x_train))
# print('len(y_train: ', len(y_train))
# # x_train = x_train.reshape(-1, 224, x_train[0].shape[0], 1)
# # x_train = x_train.reshape(list(x_train.shape) + 1)
# # x_test = x_test.reshape(-1, 224, 224, 3)
# print('len(x_test): ', len(x_test))
# print('len(y_test: ', len(y_test))
# print('len(values): ', len(values))
# print('len(labels): ', len(labels))

# history = model.fit(x_train_nn, y_train, epochs=150, callbacks=callbacks, validation_data=(x_test, y_test), batch_size=batch_size)

history = model.fit(train_flow_gen, epochs=100, callbacks=callbacks, validation_data=test_flow_gen, batch_size=batch_size)

model.load_weights(checkpoint_filepath)

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


# def print_results(video, limit=None):
#     fig = plt.figure(figsize=(16, 30))
#     if not os.path.exists('output'):
#         os.mkdir('output')
#
#     print("Loading model ...")
#     model = load_model('./model.h5')
#     Q = deque(maxlen=128)
#
#     vs = cv2.VideoCapture(video)
#     writer = None
#     (W, H) = (None, None)
#     count = 0
#     while True:
#         (grabbed, frame) = vs.read()
#         ID = vs.get(1)
#         if not grabbed:
#             break
#         try:
#             if (ID % 7 == 0):
#                 count = count + 1
#                 n_frames = len(frame)
#
#                 if W is None or H is None:
#                     (H, W) = frame.shape[:2]
#
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 output = cv2.resize(frame, (512, 360)).copy()
#                 frame = cv2.resize(frame, (128, 128)).astype("float32")
#                 frame = frame.reshape(224, 224, 3) / 255
#                 preds = model.predict(np.expand_dims(frame, axis=0))[0]
#                 Q.append(preds)
#
#                 results = np.array(Q).mean(axis=0)
#                 i = (preds > 0.56)[0]  # np.argmax(results)
#
#                 label = i
#
#                 text = "Violence: {}".format(label)
#
#                 file = open("output.txt", 'w')
#                 file.write(text)
#                 file.close()
#
#                 color = (0, 255, 0)
#
#                 if label:
#                     color = (255, 0, 0)
#                 else:
#                     color = (0, 255, 0)
#
#                 cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                             1, color, 3)
#
#                 # saving mp4 with labels but cv2.imshow is not working with this notebook
#                 if writer is None:
#                     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#                     writer = cv2.VideoWriter("output.mp4", fourcc, 60,
#                                              (W, H), True)
#
#                 writer.write(output)
#                 cv2.imshow("Output", output)
#                 cv2.waitkey()
#
#                 fig.add_subplot(8, 3, count)
#                 plt.imshow(output)
#
#             if limit and count > limit:
#                 break
#
#         except:
#             break
#
#     plt.show()
#     print("Cleaning up...")
#     if writer is not None:
#         writer.release()
#     vs.release()
#
# Violence = "C:/Users/user/Desktop/download.mp4"
# print(print_results(Violence, limit=30))
#
# ### 16
# NonViolence = "D:/abnormal_detection_dataset/UBI_FIGHTS/videos/All/N_0_0_0_1_0.mp4"
# print(print_results(NonViolence, limit=50))