#coding=utf-8  
import numpy as np 
import cv2
import dlib
from scipy.spatial import distance
import os
from imutils import face_utils

# 计算EAR
def eye_aspect_ratio(eye):
	# print(eye)
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	
pwd = os.getcwd()# 获取当前路径
model_path = os.path.join(pwd, 'model')# 模型文件夹路径
shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')# 人脸特征点检测模型路径

detector = dlib.get_frontal_face_detector()# 人脸检测器
predictor = dlib.shape_predictor(shape_detector_path)# 人脸特征点检测器

EYE_AR_THRESH = 0.2# EAR阈值
EYE_AR_CONSEC_FRAMES = 3# 当EAR小于阈值时，接连多少帧一定发生眨眼动作

# 对应特征点的序号
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1

frame_counter = 0
blink_counter = 0
cap = cv2.VideoCapture(1)
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
		print('leftEAR = {0}'.format(leftEAR))
		print('rightEAR = {0}'.format(rightEAR))

		ear = (leftEAR + rightEAR) / 2.0

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

		# for pt in leftEye:
		# 	pt_pos = (pt[0], pt[1])
		# 	cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)
			
		# for pt in rightEye:
		# 	pt_pos = (pt[0], pt[1])
		# 	cv2.circle(img, pt_pos, 2, (0, 255, 0), 1)
		
		if ear < EYE_AR_THRESH:
			frame_counter += 1
		else:
			if frame_counter >= EYE_AR_CONSEC_FRAMES:
				blink_counter += 1
			frame_counter = 0

		cv2.putText(img, "Blinks:{0}".format(blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
		cv2.putText(img, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

		
	cv2.imshow("Frame", img)

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()

