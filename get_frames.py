import os, cv2, random

train_fight_path = 'D:/abnormal_detection_dataset/RWF_2000_Dataset/train/Fight/'
train_nonfight_path = 'D:/abnormal_detection_dataset/RWF_2000_Dataset/train/NonFight/'

train_fight_img_cnt = len(os.listdir(train_fight_path))
train_nonfight_img_cnt = len(os.listdir(train_nonfight_path))

#####################################    Train data 프레임 추출 시작   #####################################
train_fight_frame_cnt = 0
train_nonfight_frame_cnt = 0

# for i in range(train_fight_img_cnt + train_nonfight_img_cnt):
#
#     random_folder = random.randint(0, 1)
#
#     if random_folder == 1:
#
#         if train_fight_frame_cnt <= train_fight_img_cnt:

for file in os.listdir(train_fight_path):
    video = cv2.VideoCapture(train_fight_path + file)
    fps = video.get(cv2.CAP_PROP_FPS)
    ret, images = video.read()
    cnt = 0

    while True:
        ret, images = video.read()

        if not ret:
            break

        if int(video.get(1)) % 3 == 0:
            cv2.imwrite('D:/abnormal_detection_dataset/RWF_2000_Dataset_frame/train/Fight/' + file[:-4] + '__' + str(cnt) + '.png', images)
            print('Saved frame number: ', str(int(video.get(1))))
        cnt += 1
    video.release()
        #     train_fight_frame_cnt += 1
        # else:
        #     continue

    # else:
    #     if train_nonfight_frame_cnt <= train_nonfight_img_cnt:
print('-' * 100)
print('Train fight data 프레임 추출 완료')
print('-' * 100)
print('-' * 100)
print('Train fight data 프레임 추출 완료')
print('-' * 100)
print('-' * 100)
print('Train fight data 프레임 추출 완료')
print('-' * 100)
print('-' * 100)
print('Train fight data 프레임 추출 완료')
print('-' * 100)

for file in os.listdir(train_nonfight_path):
    video = cv2.VideoCapture(train_nonfight_path + file)
    fps = video.get(cv2.CAP_PROP_FPS)
    ret, images = video.read()
    cnt = 0

    while True:
        ret, images = video.read()

        if not ret:
            break

        if int(video.get(1)) % 3 == 0:
            cv2.imwrite('D:/abnormal_detection_dataset/RWF_2000_Dataset_frame/train/NonFight/' + file[:-4] + '__' + str(cnt) + '.png', images)
            print('Saved frame number: ', str(int(video.get(1))))
        cnt += 1
    video.release()

print('-' * 100)
print('Train Nonfight data 프레임 추출 완료')
print('-' * 100)
print('-' * 100)
print('Train Nonfight data 프레임 추출 완료')
print('-' * 100)
print('-' * 100)
print('Train Nonfight data 프레임 추출 완료')
print('-' * 100)
print('-' * 100)
print('Train Nonfight data 프레임 추출 완료')
print('-' * 100)

#             train_nonfight_frame_cnt += 1
#####################################    Train data 프레임 추출 끝    #####################################

val_fight_path = 'D:/abnormal_detection_dataset/RWF_2000_Dataset/val/Fight/'
val_nonfight_path = 'D:/abnormal_detection_dataset/RWF_2000_Dataset/val/NonFight/'

val_fight_img_cnt = len(os.listdir(val_fight_path))
val_nonfight_img_cnt = len(os.listdir(val_nonfight_path))

#####################################    Valid data 프레임 추출 시작    #####################################
val_fight_frame_cnt = 0
val_nonfight_frame_cnt = 0

# for i in range(val_fight_img_cnt + val_nonfight_img_cnt):
#
#     random_folder = random.randint(0, 1)
#
#     if random_folder == 1:
#
#         if val_fight_frame_cnt <= val_fight_img_cnt:

for file in os.listdir(val_fight_path):
    video = cv2.VideoCapture(val_fight_path + file)
    fps = video.get(cv2.CAP_PROP_FPS)
    ret, images = video.read()
    cnt = 0

    while True:
        ret, images = video.read()

        if not ret:
            break

        if int(video.get(1)) % 3 == 0:
            cv2.imwrite('D:/abnormal_detection_dataset/RWF_2000_Dataset_frame/val/Fight/' + file[:-4] + '__' + str(cnt) + '.png', images)
            print('Saved frame number: ', str(int(video.get(1))))
        cnt += 1
    video.release()

print('-' * 100)
print('Test fight data 프레임 추출 완료')
print('-' * 100)
print('-' * 100)
print('Test fight data 프레임 추출 완료')
print('-' * 100)
print('-' * 100)
print('Test fight data 프레임 추출 완료')
print('-' * 100)
print('-' * 100)
print('Test fight data 프레임 추출 완료')
print('-' * 100)

# val_fight_frame_cnt += 1
        # else:
        #     continue

    # else:
    #     if val_nonfight_frame_cnt <= val_nonfight_img_cnt:
for file in os.listdir(val_nonfight_path):
    video = cv2.VideoCapture(val_nonfight_path + file)
    fps = video.get(cv2.CAP_PROP_FPS)
    ret, images = video.read()
    cnt = 0

    while True:
        ret, images = video.read()

        if not ret:
            break

        if int(video.get(1)) % 3 == 0:
            cv2.imwrite('D:/abnormal_detection_dataset/RWF_2000_Dataset_frame/val/NonFight/' + file[:-4] + '__' + str(cnt) + '.png', images)
            print('Saved frame number: ', str(int(video.get(1))))
        cnt += 1
    video.release()
# val_nonfight_frame_cnt += 1
#####################################    Test data 프레임 추출 끝    #####################################

print('-' * 100)
print('Test Nonfight data 프레임 추출 완료')
print('-' * 100)
print('-' * 100)
print('Test Nonfight data 프레임 추출 완료')
print('-' * 100)
print('-' * 100)
print('Test Nonfight data 프레임 추출 완료')
print('-' * 100)
print('-' * 100)
print('Test Nonfight data 프레임 추출 완료')
print('-' * 100)
print('-' * 100)
print('Test Nonfight data 프레임 추출 완료')
print('-' * 100)