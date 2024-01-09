import os, cv2, random

train_fight_path = 'D:/abnormal_detection_dataset/RWF-2000 Dataset/train/Fight/'
train_nonfight_path = 'D:/abnormal_detection_dataset/RWF-2000 Dataset/train/NonFight/'

valid_fight_path = 'D:/abnormal_detection_dataset/RWF-2000 Dataset/val/Fight/'
valid_nonfight_path = 'D:/abnormal_detection_dataset/RWF-2000 Dataset/val/NonFight/'

# def get_frames(dir):
#     global train_fight_path, train_nonfight_path
#
#     fight_img_cnt = len(os.listdir(train_fight_path))
#     nonfight_img_cnt = len(os.listdir())
#
#     for i in range(fight_img_cnt + nonfight_img_cnt):
#
#         random_folder = random.randint(0, 1)
#
#         if random_folder == 1:
#
#             for file in os.listdir(train_fight_path):
#                 video = cv2.VideoCapture(train_fight_path)
#                 fps = video.get(cv2.CAP_PROP_FPS)
#                 ret, images = video.read()
#                 cnt = 0
#
#                 while True:
#                     ret, images = video.read()
#
#                     if not ret:
#                         break
#
#                     if int(video.get(1)) % fps == 0:
#                         cv2.imwrite('RWF-2000 Dataset_frame/train/Fight' + file[:-4] +  '__' + str(cnt) + '.png', images)
#                         print('Saved frame number: ', str(int(video.get(1))))
#                     cnt += 1
#                 video.release()
#         else:
#             for file in os.listdir(train_nonfight_path):
#                 video = cv2.VideoCapture(train_nonfight_path)
#                 fps = video.get(cv2.CAP_PROP_FPS)
#                 ret, images = video.read()
#                 cnt = 0
#
#                 while True:
#                     ret, images = video.read()
#
#                     if not ret:
#                         break
#
#                     if int(video.get(1)) % fps == 0:
#                         cv2.imwrite('RWF-2000 Dataset_frame/train/NonFight' + file[:-4] +  '__' + str(cnt) + '.png', images)
#                         print('Saved frame number: ', str(int(video.get(1))))
#                     cnt += 1
#                 video.release()

for i in os.listdir(train_fight_path):
    cap = cv2.VideoCapture(0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print('file name, width, height, fps: %s, %d, %d, %d' % (i, width, height, fps))
    cap.release()
print('train_fight_path 끝 ')
print('-' * 100)

for i in os.listdir(train_nonfight_path):
    cap = cv2.VideoCapture(0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print('Video width, height, fps: %d, %d, %d' % (width, height, fps))
    cap.release()
print('train_nonfight_path 끝 ')
print('-' * 100)


for i in os.listdir(valid_fight_path):
    cap = cv2.VideoCapture(0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print('Video width, height, fps: %d, %d, %d' % (width, height, fps))
    cap.release()
print('valid_fight_path 끝 ')
print('-' * 100)


for i in os.listdir(valid_nonfight_path):
    cap = cv2.VideoCapture(0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print('Video width, height, fps: %d, %d, %d' % (width, height, fps))
    cap.release()
print('valid_nonfight_path 끝 ')
print('-' * 100)


    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width / 3)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height / 3)
    #
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # print('changed size: %d, %d' % (width, height))