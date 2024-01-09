import os, cv2

# video = cv2.VideoCapture('D:/abnormal_detection_dataset/RWF-2000 Dataset/val/Fight/fi0.avi')
video = cv2.VideoCapture('D:/abnormal_detection_dataset/UBI_FIGHTS/videos/All/F_0_1_0_0_0.mp4')

while video.isOpened():
    check, frame = video.read()

    if not check:
        print('frame이 끝났습니다.')
        break
    print(frame.shape)
    cv2.imshow('cute cats', frame)
    cv2.waitKey()

video.release()
# cv2.destroyAllWindows()

train_folder = 'D:/abnormal_detection_dataset/RWF_2000_Dataset/val/'

# for i in os.listdir('D:/abnormal_detection_dataset/RWF_2000_Dataset/val/'):
#     video = cv2.VideoCapture('D:/abnormal_detection_dataset/RWF_2000_Dataset/val/' + i)
#     print(i)
#     print(video)

val_nonfight = os.path.join(train_folder + 'NonFight')
val_fight = os.path.join(train_folder + 'Fight')
# for i in train_folder:
# print(os.listdir(val_nonfight))
print(os.walk(val_fight))
