#coding=utf-8  
import numpy as np 
import os
import dlib
import cv2
from scipy.spatial import distance
from imutils import face_utils
import pickle

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

pwd = os.getcwd()
model_path = os.path.join(pwd, 'model')
shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_detector_path)

cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)

cap = cv2.VideoCapture(1)

# 对应特征点的序号
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1

print('Prepare to collect images with your eyes open')
print('Press s to start collecting images.')
print('Press e to end collecting images.')
print('Press q to quit')
flag = 0
txt = open('train_open.txt', 'wb')
data_counter = 0
ear_vector = []
while(1):
	ret, frame = cap.read()
	key = cv2.waitKey(1)
	if key & 0xFF == ord("s"):
		print('Start collecting images.')
		flag = 1
	elif key & 0xFF == ord("e"):
		print('Stop collecting images.')
		flag = 0
	elif key & 0xFF == ord("q"):
		print('quit')
		break

	if flag == 1:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rects = detector(gray, 0)
		for rect in rects:
			shape = predictor(gray, rect)
			points = face_utils.shape_to_np(shape)# convert the facial landmark (x, y)-coordinates to a NumPy array
			# points = shape.parts()
			leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]
			rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			# print('leftEAR = {0}'.format(leftEAR))
			# print('rightEAR = {0}'.format(rightEAR))

			ear = (leftEAR + rightEAR) / 2.0

			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

			ret, ear_vector = queue_in(ear_vector, ear)
			if(len(ear_vector) == VECTOR_SIZE):
				# print(ear_vector)
				# input_vector = []
				# input_vector.append(ear_vector)

				txt.write(str(ear_vector).encode())
				txt.write('\n'.encode())
				data_counter += 1
				print(data_counter)

			cv2.putText(frame, "EAR:{:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

	cv2.imshow("frame", frame)
txt.close()

print('-'*40)
print('Prepare to collect images with your eyes close')
print('Press s to start collecting images.')
print('Press e to end collecting images.')
print('Press q to quit')
flag = 0
txt = open('train_close.txt', 'wb')
data_counter = 0
ear_vector = []
while(1):
	ret, frame = cap.read()
	key = cv2.waitKey(1)
	if key & 0xFF == ord("s"):
		print('Start collecting images.')
		flag = 1
	elif key & 0xFF == ord("e"):
		print('Stop collecting images.')
		flag = 0
	elif key & 0xFF == ord("q"):
		print('quit')
		break

	if flag == 1:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rects = detector(gray, 0)
		for rect in rects:
			shape = predictor(gray, rect)
			points = face_utils.shape_to_np(shape)# convert the facial landmark (x, y)-coordinates to a NumPy array
			# points = shape.parts()
			leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]
			rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			# print('leftEAR = {0}'.format(leftEAR))
			# print('rightEAR = {0}'.format(rightEAR))

			ear = (leftEAR + rightEAR) / 2.0

			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

			ret, ear_vector = queue_in(ear_vector, ear)
			if(len(ear_vector) == VECTOR_SIZE):
				# print(ear_vector)
				# input_vector = []
				# input_vector.append(ear_vector)

				txt.write(str(ear_vector).encode())
				txt.write('\n'.encode())
			
				data_counter += 1
				print(data_counter)

			cv2.putText(frame, "EAR:{:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

	cv2.imshow("frame", frame)
txt.close()


cap.release()
cv2.destroyAllWindows()