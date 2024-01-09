### 1
import os
import cv2
import numpy as np
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
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau
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

print(platform.platform())

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(device_lib.list_local_devices())

def resolve_dir(Dir):
    if not os.path.exists(Dir):
        os.mkdir(Dir)

def reset_path(Dir):
    if not os.path.exists(Dir):
        os.mkdir(Dir)
    # else:
    #     os.system('rm -f {}/*'.format( Dir))

tf.random.set_seed(73)

MyDrive = './'

PROJECT_DIR = 'D:/abnormal_detection_dataset/UBI_FIGHTS/videos'



def play(filename):
    html = ''
    video = open(filename,'rb').read()
    src = 'data:video/mp4;base64,' + b64encode(video).decode()
    html += '<video width=640 muted controls autoplay loop><source src="%s" type="video/mp4"></video>' % src 
    return HTML(html)


### 2

IMG_SIZE = 512
ColorChannels = 3

def video_to_frames(video):
    vidcap = cv2.VideoCapture(video)
    
    import math
    rate = math.floor(vidcap.get(3))
    count = 0
    
    ImageFrames = []
    while vidcap.isOpened():
        ID = vidcap.get(1)
        success, image = vidcap.read()
        
        if success:
            # skipping frames to avoid duplications 
            if (ID % 7 == 0):
                flip = iaa.Fliplr(1.0)
                zoom = iaa.Affine(scale=1.3)
                random_brightness = iaa.Multiply((1, 1.3))
                rotate = iaa.Affine(rotate=(-25, 25))
                
                image_aug = flip(image = image)
                image_aug = random_brightness(image = image_aug)
                image_aug = zoom(image = image_aug)
                image_aug = rotate(image = image_aug)
                
                rgb_img = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
                ImageFrames.append(resized)
                
            count += 1
        else:
            break
    
    vidcap.release()
    
    return ImageFrames


### 3

VideoDataDir = PROJECT_DIR + '/videos'
print('we have \n{} Violence videos \n{} NonViolence videos'.format(
              len(os.listdir(VideoDataDir + '/fight')), 
              len(os.listdir(VideoDataDir + '/normal'))))

X_original = []
y_original = []

CLASSES = ["normal", "fight"]
#700 <- 350 + 350

for category in os.listdir(VideoDataDir):
    path = os.path.join(VideoDataDir, category)
    class_num = CLASSES.index(category)
    
    for i, video in enumerate(tqdm(os.listdir(path)[:10])):
        frames = video_to_frames(path + '/' + video)
        for j, frame in enumerate(frames):
            X_original.append(frame)
            y_original.append(class_num)

print('i choose {} videos out of {}, cuz of memory issue'.format(i, (len(os.listdir(VideoDataDir + '/fight'))+ 
                                                                            len((os.listdir(VideoDataDir + '/normal'))))))


### 4
X_original = np.array(X_original).reshape(-1 , IMG_SIZE * IMG_SIZE * 3)
y_original = np.array(y_original)
print('len(X_original): ', len(X_original))


stratified_sample = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=73)

for train_index, test_index in stratified_sample.split(X_original, y_original):
#     print(len(train_index), len(test_index))
    X_train, X_test = X_original[train_index], X_original[test_index]
    y_train, y_test = y_original[train_index], y_original[test_index]
# print(len(X_train))
# print(len(X_test))
# print(len(y_train))
# print(len(y_test))

# print(X_train)
# print('=' * 100)
# print(y_train)

# print('=' * 100)

# print(X_test)
# print('=' * 100)
# print(y_test)

# print('IMG_SIZE: ', IMG_SIZE)
# print('=' * 100)
# print('len(X_original): ', len(X_original))
# print('X_original.shape: ', X_original.shape)
# print('y_original.shape: ', y_original.shape)
# print('=' * 100)
# print('X_train.shape: ', X_train.shape)
# print('y_train.shape: ', y_train.shape)
# print('X_test.shape: ', X_test.shape)
# print('y_test.shape: ', y_test.shape)
# print('=' * 100)

### 5
X_train_nn = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255
X_test_nn = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255


### 6
epochs = 0

kernel_regularizer = regularizers.l2(0.0001)

def load_layers():    
    # model = Sequential()
    input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, ColorChannels))
    input_shape = (IMG_SIZE, IMG_SIZE, ColorChannels) # (128, 128, 3)

    baseModel = VGG16(pooling='avg', include_top=False, input_shape=input_shape, weights='imagenet')
    baseModel.trainable = False

    headModel = baseModel.output
    headModel = Dense(1, activation='sigmoid')(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)
    # model = Model(inputs=input_shape)

    #######################쓰는거 #######################
    # model.add(layers.Input(shape=(input_shape)))
    # model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation = 'relu'))
    # model.add(layers.MaxPooling2D(strides=(2,2)))
    #
    # model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation = 'relu'))
    # model.add(layers.Dropout(0.2))
    # model.add(layers.MaxPooling2D(strides=(2, 2)))
    #
    # model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation = 'relu'))
    # model.add(layers.Dropout(0.2))
    # model.add(layers.MaxPooling2D(strides=(2, 2)))
    #
    # model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
    # model.add(layers.Dropout(0.2))
    # model.add(layers.MaxPooling2D(strides=(2, 2)))
    #
    # model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
    # model.add(layers.Dropout(0.2))
    #######################쓰는거 #######################

    ################## 이건 안씀 ####################
    # model.add(layers.MaxPooling2D(strides=(2, 2)))

    # model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    # model.add(layers.Dropout(0.2))
    # model.add(layers.MaxPooling2D(strides=(2, 2)))

    # model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    # model.add(layers.Dropout(0.2))
    # model.add(layers.MaxPooling2D(strides=(2, 2)))

    # model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    # model.add(layers.MaxPooling2D(strides=(2, 2)))
    ################## 이건 안씀 ####################
     #######################쓰는거 #######################
    # model.add(layers.Flatten())
    # model.add(layers.Dense(1024, activation='relu'))
    # model.add(layers.Dropout(0.2))
    # model.add(layers.Dense(512, activation='relu'))
    # # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(1, activation='sigmoid'))
    # model = Model(inputs=model.input, outputs=model.output)
    #######################쓰는거 #######################
    ###### 안씀 ############################
    # model.add(layers.Input(shape=(input_shape)))
    # model.add(layers.Conv2D(filters=32, kernel_size=(9,9), padding='valid'))
    # model.add(layers.MaxPooling2D(strides=(1, 1)))
    # model.add(layers.Conv2D(filters=32, kernel_size=(9,9), padding='valid', activation='relu'))
    # model.add(layers.Dropout(0.4))
    # model.add(layers.MaxPooling2D(strides=(1, 1)))
    # model.add(layers.Conv2D(filters=32, kernel_size=(9,9), padding='valid', activation='relu'))
    # model.add(layers.Dropout(0.4))
    # model.add(layers.MaxPooling2D(strides=(1, 1)))
    # model.add(layers.Conv2D(filters=32, kernel_size=(6,6), padding='valid', activation='relu'))
    # model.add(layers.Dropout(0.4))
    # model.add(layers.MaxPooling2D(strides=(1, 1)))
    # model.add(layers.Conv2D(filters=32, kernel_size=(6,6), padding='valid', activation='relu'))
    # model.add(layers.Dropout(0.4))
    # model.add(layers.MaxPooling2D(strides=(1, 1)))
    # model.add(layers.Conv2D(filters=64, kernel_size=(6,6), padding='valid', activation='relu'))
    # model.add(layers.Dropout(0.4))
    # model.add(layers.MaxPooling2D(strides=(1, 1)))
    # model.add(layers.Conv2D(filters=64, kernel_size=(5,5), padding='valid', activation='relu'))
    # model.add(layers.Dropout(0.4))
    # model.add(layers.MaxPooling2D(strides=(1, 1)))
    # model.add(layers.Conv2D(filters=64, kernel_size=(5,5), padding='valid', activation='relu'))
    # model.add(layers.Dropout(0.4))
    # model.add(layers.MaxPooling2D(strides=(1, 1)))
    # model.add(layers.Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation='relu'))
    # model.add(layers.MaxPooling2D(strides=(1, 1)))
    # model.add(layers.Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation='relu'))
    # model.add(layers.MaxPooling2D(strides=(1, 1)))
    # model.add(layers.Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation='relu'))
    # model.add(layers.MaxPooling2D(strides=(1, 1)))
    # model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
    # model.add(layers.MaxPooling2D(strides=(1, 1)))
    # model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
    # model.add(layers.MaxPooling2D(strides=(1, 1)))
    # model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
    # model.add(layers.MaxPooling2D(strides=(1, 1)))
    # model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
    # model.add(layers.MaxPooling2D(strides=(1, 1)))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(1024, activation='relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(1, activation='relu'))
    # model = Model(inputs=model.input, outputs=model.output)
    ###### 안씀 ############################

    # for layer in model.layers:
    #     layer.trainable = False

    print("Compiling model...")
    # adam = optimizers.adam(lr=0.01, decay=1e-6)
    model.compile(loss="binary_crossentropy",
                    optimizer='adam',
                    metrics=["accuracy"])

    return model


model = load_layers()

model.summary()
# plot_model(model, to_file='model.png')
# plot_model(model, to_file='model_shapes.png', show_shapes=True)


### 7
patience = 10

start_lr = 0.00001
min_lr = 0.00001
max_lr = 0.00005

batch_size = 32 #128


rampup_epochs = 5
sustain_epochs = 0
exp_decay = 0.0001

def lrfn(epoch):
    if epoch < rampup_epochs:
        return (max_lr - start_lr)/rampup_epochs * epoch + start_lr
    elif epoch < rampup_epochs + sustain_epochs:
        return max_lr
    else:
        return (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr


class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if ((logs.get('accuracy')>=0.999)):
            print("\nLimits Reached cancelling training!")
            self.model.stop_training = True


end_callback = myCallback()

lr_callback = LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=False)

early_stopping = EarlyStopping(patience = patience, monitor='val_loss',
                                 mode='min', restore_best_weights=True, 
                                 verbose = 1, min_delta = .00075)

PROJECT_DIR = MyDrive + '/RiskDetection'

lr_plat = ReduceLROnPlateau(patience = 2, mode = 'min')

# os.system('rm -rf ./logs/')

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = TensorBoard(log_dir = log_dir, write_graph=True, histogram_freq=1)

checkpoint_filepath = './checkpoint.h5'

model_checkpoints = ModelCheckpoint(filepath=checkpoint_filepath,
                                        save_weights_only=True,
                                        monitor='val_loss',
                                        mode='min',
                                        verbose = 1,
                                        save_best_only=True)


# callbacks = [end_callback, lr_callback, model_checkpoints, early_stopping, lr_plat]
callbacks = [model_checkpoints, early_stopping]



### 8
print('Training head...')
history = model.fit(X_train_nn ,y_train, epochs=epochs, callbacks=callbacks, validation_data = (X_test_nn, y_test),
                        batch_size=batch_size)
print(history.history.keys())
print('model.metrics_names: ', model.metrics_names)

print(history.history)
model.load_weights(checkpoint_filepath)


### 9
def print_graph(item, index, history):
    plt.figure()
    train_values = history.history[item][0:index]
    plt.plot(train_values)
    test_values = history.history['val_' + item][0:index]
    plt.plot(test_values)
    plt.legend(['training','validation'])
    plt.title('Training and validation '+ item)
    plt.xlabel('epoch')
    plt.show()
    plot = '{}.png'.format(item)
    plt.savefig(plot)


def get_best_epoch(test_loss, history):
    for key, item in enumerate(history.history.items()):
        (name, arr) = item
        if name == 'val_loss':
            for i in range(len(arr)):
                if round(test_loss, 2) == round(arr[i], 2):
                    return i
                
def model_summary(model, history):
    print('---'*30)
    test_loss, test_accuracy = model.evaluate(X_test_nn, y_test, verbose=0)

    if history:
        index = get_best_epoch(test_loss, history)
        print('Best Epochs: ', index)

        train_accuracy = history.history['accuracy'][index]
        train_loss = history.history['loss'][index]

        print('Accuracy on train:',train_accuracy,'\tLoss on train:',train_loss)
        print('Accuracy on test:',test_accuracy,'\tLoss on test:',test_loss)
        print_graph('loss', index, history)
        print_graph('accuracy', index, history)
        print('---'*30)


### 10
# model_summary(model, history)
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


### 11
# evaluate the network
print("Evaluating network...")
predictions = model.predict(X_test_nn)
preds = predictions > 0.5


### 12
corr_pred = metrics.confusion_matrix(y_test, preds)

n_correct = int((corr_pred[0][0] + corr_pred[1][1]))
print('> Correct Predictions:', n_correct)
n_wrongs = int((corr_pred[0][1] + (corr_pred[1][0])))
print('> Wrong Predictions:', n_wrongs)

sns.heatmap(corr_pred,annot=True, fmt="d",cmap="Blues")
plt.show()

print(metrics.classification_report(y_test, preds, 
                           target_names=["NonViolence", "Violence"]))


### 13
args_model = "model.h5"
model.save(args_model)

### 14
### Testing
def print_results(video, limit=None):
        fig=plt.figure(figsize=(16, 30))
        if not os.path.exists('output'):
            os.mkdir('output')

        print("Loading model ...")
        model = load_model('./model.h5')
        Q = deque(maxlen=128)

        vs = cv2.VideoCapture(video)
        writer = None
        (W, H) = (None, None)
        count = 0     
        while True:
                (grabbed, frame) = vs.read()
                ID = vs.get(1)
                if not grabbed:
                    break
                try:
                    if (ID % 7 == 0):
                        count = count + 1
                        n_frames = len(frame)
                        
                        if W is None or H is None:
                            (H, W) = frame.shape[:2]

                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        output = cv2.resize(frame, (512, 360)).copy()
                        frame = cv2.resize(frame, (128, 128)).astype("float32")
                        frame = frame.reshape(IMG_SIZE, IMG_SIZE, 3) / 255
                        preds = model.predict(np.expand_dims(frame, axis=0))[0]
                        Q.append(preds)

                        results = np.array(Q).mean(axis=0)
                        i = (preds > 0.56)[0] #np.argmax(results)

                        label = i

                        text = "Violence: {}".format(label)
                        
                        file = open("output.txt",'w')
                        file.write(text)
                        file.close()

                        color = (0, 255, 0)

                        if label:
                            color = (255, 0, 0) 
                        else:
                            color = (0, 255, 0)

                        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, color, 3)


                        # saving mp4 with labels but cv2.imshow is not working with this notebook
                        if writer is None:
                                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                                writer = cv2.VideoWriter("output.mp4", fourcc, 60,
                                        (W, H), True)

                        writer.write(output)
                        cv2.imshow("Output", output)
                        cv2.waitkey()

                        fig.add_subplot(8, 3, count)
                        plt.imshow(output)

                    if limit and count > limit:
                        break

                except:
                    break 
        
        plt.show()
        print("Cleaning up...")
        if writer is not None:
            writer.release()
        vs.release()


### 15
# Violence="D:/abnormal_detection_dataset/UBI_FIGHTS/videos/All/F_0_1_0_0_0.mp4"
Violence = "C:/Users/user/Desktop/download.mp4"
print(print_results(Violence, limit=30))

### 16
NonViolence = "D:/abnormal_detection_dataset/UBI_FIGHTS/videos/All/N_0_0_0_1_0.mp4"
print(print_results(NonViolence, limit=50))