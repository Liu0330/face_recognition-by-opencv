# -*- coding: utf-8 -*-
import numpy as np

import cv2

import os

#数据预处理大小
dirs = os.listdir('./faces/')
for d in dirs:
    filename = [f for f in os.listdir('./faces/%s'%(d))]
    for fn in filename:
        img = cv2.imread('./faces/%s/%s'%(d,fn))
        w,h,c = img.shape
        if w != 1:
            img2 = cv2.resize(img,(640,360))
            cv2.imwrite('./faces/%s/%s'%(d,fn),img2)


#准备工作 文件整理 训练
filenames = os.listdir('./faces/')
faces = []
# targerts == labels标签
targets = []
for f in filenames:
    for fn in os.listdir('./faces/%s'%(f)):
        faces.append(cv2.imread('./faces/%s/%s'%(f,fn)))
        targets.append(f.split('.')[0])
faces = np.asarray(faces)
targets = np.asarray(targets)
print(len(targets)) #200


labels = np.asarray([i for i in range(1,21)]*10)
labels.sort()
print(labels)  #生成二百个数的顺序列表作为标签



labels_train = labels[::2]
print("labels_train",len(labels_train)) #训练的标签个数100

faces_train = faces[::2]
print("faces_train",len(faces_train)) #训练的面孔个数100


# 灰度化处理
faces_train2 = []
for face in faces_train:
    gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
    faces_train2.append(gray)
faces_train2 = np.asarray(faces_train2)
# faces_train2.shape


# face-recognizer人脸的识别
# Eigen特征，根据特征值相似，认为同一个人
fr = cv2.face.EigenFaceRecognizer_create()
# 训练
fr.train(faces_train2,labels_train)


#叫什么名字的标签
labels_test =labels[1::2]
targets_labels =targets[::10]


#开启摄像头
v = cv2.VideoCapture(0)
flag = True
face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

while True:
    flag, frame = v.read()

    if not flag:
        break

    #     调整摄像头尺寸 将摄像头内容降维
    frame = cv2.resize(frame, (640, 360))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 通过face_detector 识别人脸位置
    face_zones = face_detector.detectMultiScale(gray, scaleFactor=1.1,
                                                #minNeighbors=3,
                                                #minSize =(60,60),
                                                #maxSize =(110,110)
                                                )
    # 接收人脸位置的坐标 注意是要相加得出下一个点坐标，画出线条
    for x, y, w, h in face_zones:
        #     cv2.rectangle(img,pt1=(x,y),pt2=(x+w,y+h),color= [0,0,255],thickness=2,)
        cv2.circle(frame, center=(x + w // 2, y + h // 2), radius=w // 2, color=[0, 0, 255], thickness=2, )

        # 人脸位置
        face = frame[y + 2:y + h - 2, x + 2:x + w - 2]
        face_re = cv2.resize(face, (640, 360))

        gray = cv2.cvtColor(face_re, cv2.COLOR_BGR2GRAY)
        #    通过学习的内容进行识别
        label, confidence = fr.predict(gray)
        #         cv2.imshow(targets_labels[label-1],face)

        print(label, confidence)
        print('------------------', targets_labels[label - 1])
        #         cv2.putText(frame,'text' ,(50,150))
        cv2.putText(frame, '%s' % (targets_labels[label - 1]), (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 1)
    # 显示摄像头内容每毫秒一帧
    cv2.imshow('dzd', frame)

    key = cv2.waitKey(1)

    # 退出
    if key == ord('q'):
        break

v.release()  # 释放视频流
cv2.destroyAllWindows()