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

### 동영상 폴더
# train_folder = 'D:/abnormal_detection_dataset/UBI_FIGHTS/videos/All/'
train_folder = 'D:/abnormal_detection_dataset/UBI_FIGHTS/frames1/' #train
test_folder = 'D:/abnormal_detection_dataset/UBI_FIGHTS/frames/' #test

train_fight_label = train_folder + '1/'          # fight
train_normal_label = train_folder + '0/'         # normal
train_fight_and_normal_label = len(os.listdir(train_fight_label)) + len(os.listdir(train_normal_label))

test_fight_label = test_folder + '1/'
test_normal_label = test_folder + '0/'
test_fight_and_normal_label = len(os.listdir(test_fight_label)) + len(os.listdir(test_normal_label))
print(test_fight_and_normal_label)   # 42217

# print(train_fight_and_normal_label)   # 84808
# print(os.listdir(train_fight_label))  # 파일 이름 리스트에 저장
# print(os.listdir(train_fight_label))  # 파일 이름 리스트에 저장
print(os.listdir(train_folder))   # ['fight', 'normal'] >>>> ['1', '0']

def train_gen():
    global train_fight_and_normal_label, train_normal_label, train_fight_label

    ################################################## train - fight
    train_fight_img_list = os.listdir(train_fight_label)
    train_fight_img_list_png = [img for img in train_fight_img_list if img.endswith(".png")]

    train_fight_img_list_np = []

    for i in train_fight_img_list_png:

        path = train_fight_label + i

        train_fight_img = cv2.imread(path) #train_fight_label + i
        train_fight_img_array = np.array(train_fight_img)
        train_fight_img_list_np.append(train_fight_img_array)

        train_fight_img_np = np.array(train_fight_img_list_np)  # 리스트를 넘파이로 변환
        train_fight_img_np = train_fight_img_np.astype(np.int16)
        np.reshape(train_fight_img_np, (224,224,15))
        print('=-===========================================================')
        print(train_fight_img_np.shape)
        print('=-===========================================================')
        # for i in range(5):
        #     train_img_list.append(train)

        # train_fight_img_np = np.reshape(train_fight_img_np, (224, 224, 15))

        train_fight_labeling = np.array(os.listdir(train_folder))
        train_fight_labeling = keras.utils.to_categorical(train_fight_labeling, 2)
        train_fight_labeling = np.array(train_fight_labeling)
        train_fight_labeling = train_fight_labeling.astype(np.int8)
        # train_fight_labeling = np.reshape(train_fight_labeling, (1, 1))

        yield (train_fight_img_np, train_fight_labeling)

    ################################################## train - nomal
    train_normal_img_list = os.listdir(train_normal_label)
    train_normal_img_list_png = [img for img in train_normal_img_list if img.endswith(".png")]

    train_normal_img_list_np = []

    for i in train_normal_img_list_png:

        path = train_normal_label + i

        train_normal_img = cv2.imread(path)
        train_normal_img_array = np.array(train_normal_img)
        train_normal_img_list_np.append(train_normal_img_array)

        train_normal_img_np = np.array(train_normal_img_list_np)
        train_normal_img_np = train_normal_img_np.astype(np.int16)
        train_normal_img_np = np.reshape(train_normal_img_np, (224, 224, 15))

        train_normal_labeling = np.array(os.listdir(train_folder))
        train_normal_labeling = keras.utils.to_categorical(train_normal_labeling, 2)
        train_normal_labeling = np.array(train_normal_labeling)
        train_normal_labeling = train_normal_labeling.astype(np.int8)
        # train_normal_labeling = np.reshape(train_normal_labeling, (1, 1))

        yield (train_normal_img_np, train_normal_labeling)

def test_gen():
    global test_fight_and_normal_label, test_normal_label, test_fight_label

################################################## test - fight
    test_fight_img_list = os.listdir(test_fight_label)
    test_fight_img_list_png = [img for img in test_fight_img_list if img.endswith(".png")]

    test_fight_img_list_np = []

    for i in test_fight_img_list_png:

        path = test_fight_label + i

        test_fight_img = cv2.imread(path)
        test_fight_img_array = np.array(test_fight_img)
        test_fight_img_list_np.append(test_fight_img_array)

        test_fight_img_np = np.array(test_fight_img_list_np)
        test_fight_img_np = test_fight_img_np.astype(np.int16)
        test_fight_img_np = np.reshape(test_fight_img_np, (1, 224, 224, 15))

        test_fight_labeling = np.array(os.listdir(test_folder))
        test_fight_labeling = keras.utils.to_categorical(test_fight_labeling, 2)
        test_fight_labeling = np.array(test_fight_labeling)
        test_fight_labeling = test_fight_labeling.astype(np.int8)
        test_fight_labeling = np.reshape(test_fight_labeling, (1, 1))

        yield (test_fight_img_np, test_fight_labeling)

################################################## test - normal
    test_normal_img_list = os.listdir(test_normal_label)
    test_normal_img_list_png = [img for img in test_normal_img_list if img.endswith(".png")]

    test_normal_img_list_np = []

    for i in test_normal_img_list_png:
        path = test_normal_label + i

        test_normal_img = cv2.imread(path)
        test_normal_img_array = np.array(test_normal_img)
        test_normal_img_list_np.append(test_normal_img_array)

        test_normal_img_np = np.array(test_normal_img_list_np)
        test_normal_img_np = test_normal_img_np.astype(np.int16)
        test_normal_img_np = np.reshape(test_normal_img_np, (1, 224, 224, 15))

        test_normal_labeling = np.array(os.listdir(test_folder))
        test_normal_labeling = keras.utils.to_categorical(test_normal_labeling, 2)
        test_normal_labeling = np.array(test_normal_labeling)
        test_normal_labeling = test_normal_labeling.astype(np.int8)
        test_normal_labeling = np.reshape(test_normal_labeling, (1, 1))


        yield (test_normal_img_np, test_normal_labeling)

train_dataset = tf.data.Dataset.from_generator(train_gen, (tf.int16, tf.int8), ((224, 224, 15), (1)))
train_dataset = train_dataset.batch(4)

test_dataset = tf.data.Dataset.from_generator(test_gen, (tf.int16, tf.int8), ((224, 224, 15), (1)))
test_dataset = test_dataset.batch(4)

# print('=-===========================================================')
# print(tf.shape(train_dataset).numpy())
# print(tf.shape(test_dataset).numpy())
######## 모델 생성
model = Sequential()  ### CNN-LSTM
input_shape = (224, 224, 15)  # (224, 224, 3)

model.add(layers.Input(shape=(input_shape)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(strides=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling2D(strides=(2, 2)))

model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling2D(strides=(2, 2)))

model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling2D(strides=(2, 2)))

model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))

model.add(layers.TimeDistributed(Flatten()))

model.add(layers.LSTM(1024))

model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model = Model(inputs=model.input, outputs=model.output)

for layer in model.layers:
    layer.trainable = True

### 모델 컴파일
print('모델 컴파일 중')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

frame_list = []
label_list = []
name_list = []

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

checkpoint_filepath = './checkpoint.h5'

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

label_list = np.array(label_list)

history = model.fit(train_dataset, epochs=100, callbacks=callbacks, validation_data=test_dataset, batch_size=batch_size)

model.load_weights(checkpoint_filepath)

print(history.history)

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