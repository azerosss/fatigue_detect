#coding=utf-8  
import numpy as np 
import cv2
import dlib
from scipy.spatial import distance
import os
from imutils import face_utils
from sklearn import svm
#from sklearn.externals import joblib
import joblib
import pygame

#计时函数
import time

#开始计时
time_start = time.time()

VECTOR_SIZE = 3
def queue_in(queue, data):
	ret = None
	if len(queue) >= VECTOR_SIZE:
		ret = queue.pop(0)
	queue.append(data)
	return ret, queue

def eye_aspect_ratio(eye):
	# print(eye)
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def countX(lst, x):
    count = 0
    for ele in lst:
        if (ele == x):
            count = count + 1
    return count


pwd = os.getcwd()
model_path = os.path.join(pwd, 'model')
shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_detector_path)

# 导入模型
clf = joblib.load("ear_svm.m")

EYE_AR_THRESH = 0.3# EAR阈值
EYE_AR_CONSEC_FRAMES = 3# 当EAR小于阈值时，接连多少帧一定发生眨眼动作

# 对应特征点的序号
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1

frame_counter = 0
blink_counter = 0
ear_vector = []
cap = cv2.VideoCapture(1)

#读取次数
flag = 0
perclos = []

pygame.mixer.init()
pygame.mixer.music.load("beep.wav")


while(1):
	ret, img = cap.read()
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	for rect in rects:
		print('-'*60)
		shape = predictor(gray, rect)
		points = face_utils.shape_to_np(shape)# convert the facial landmark (x, y)-coordinates to a NumPy array
		# points = shape.parts()
		leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]
		rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		'''
		print('leftEAR = {0}'.format(leftEAR))
		print('rightEAR = {0}'.format(rightEAR))
		'''

		ear = (leftEAR + rightEAR) / 2.0

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)
		
		ret, ear_vector = queue_in(ear_vector, ear)
		if(len(ear_vector) == VECTOR_SIZE):   #将连续的3个EAR存入队列
			print(ear_vector)
			input_vector = []
			input_vector.append(ear_vector)
			res = clf.predict(input_vector)
			print(res)

			if res == 1:
				frame_counter += 1
			else:
				if frame_counter >= EYE_AR_CONSEC_FRAMES:
					blink_counter += 1
				frame_counter = 0

			perclos.append(res[0])

		time_tmp = time.time()
		time_lim = time_tmp - time_start
		print (time_lim)
		if (int(time_lim)%10 == 0):
			if (countX(perclos,0) / len(perclos)) >= 0.2:
				print('*'*8,int(time_lim),'*'*8)
				pygame.mixer.music.play()
			perclos.clear()

		cv2.putText(img, "Blinks:{0}".format(blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
		cv2.putText(img, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

		flag = flag + 1
		
	cv2.imshow("Frame", img)#格式问题
	
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()
print ("blink=",blink_counter)

print('-'*60)
#print (cap.get(5))   #摄像头可能不能用此方法实时显示帧率
print (flag)
#计时结束
time_end = time.time()
total_time = time_end-time_start
print('totally cost',total_time)
#帧率
print ("FPS:",flag/total_time)
print (type(res[0]))
print (perclos)
print (len(perclos))

