import numpy as np 
from sklearn import svm
from sklearn.externals import joblib

train_open_txt = open('train_open.txt', 'rb')
train_close_txt = open('train_close.txt', 'rb')

train = []
labels = []

print('Reading train_open.txt...')
line_ctr = 0
for txt_str in train_open_txt.readlines():
	temp = []
	# print(txt_str)
	datas = txt_str.strip()
	datas = datas.replace('[', '')
	datas = datas.replace(']', '')
	datas = datas.split(',')
	print(datas)
	for data in datas:
		# print(data)
		data = float(data)
		temp.append(data)
	# print(temp)
	train.append(temp)
	labels.append(0)

	# data = float(txt_str)

	# if line_ctr <= 12:
	# 	line_ctr += 1
	# 	temp.append(data)
	# elif line_ctr == 13:
	# 	# print(temp)
	# 	# print(len(temp))
	# 	train.append(temp)
	# 	labels.append(0)
	# 	temp = []
	# 	line_ctr = 1
	# 	temp.append(data)


print('Reading train_close.txt...')
line_ctr = 0
temp = []
for txt_str in train_close_txt.readlines():
	temp = []
	# print(txt_str)
	datas = txt_str.strip()
	datas = datas.replace('[', '')
	datas = datas.replace(']', '')
	datas = datas.split(',')
	print(datas)
	for data in datas:
		# print(data)
		data = float(data)
		temp.append(data)
	# print(temp)
	train.append(temp)
	labels.append(1)

	# data = float(txt_str)

	# if line_ctr <= 12:
	# 	line_ctr += 1
	# 	temp.append(data)
	# elif line_ctr == 13:
	# 	# print(temp)
	# 	# print(len(stemp))
	# 	train.append(temp)
	# 	labels.append(1)
	# 	temp = []
	# 	line_ctr = 1
	# 	temp.append(data)

for i in range(len(labels)):
	print("{0} --> {1}".format(train[i], labels[i]))

train_close_txt.close()
train_open_txt.close()

print(train)
print(labels)
clf = svm.SVC(C=0.8, kernel='linear', gamma=20, decision_function_shape='ovo')
clf.fit(train, labels)
joblib.dump(clf, "ear_svm.m")

# print('predicting [[0.34, 0.34, 0.31, 0.32, 0.32, 0.32, 0.33, 0.31, 0.32, 0.32, 0.32, 0.31, 0.32]]')
# res = clf.predict([[0.34, 0.34, 0.31, 0.32, 0.32, 0.32, 0.33, 0.31, 0.32, 0.32, 0.32, 0.31, 0.32]])
# print(res)

# print('predicting [[0.19, 0.18, 0.18, 0.19, 0.18, 0.18, 0.17, 0.16, 0.18, 0.17, 0.17, 0.17, 0.18]]')
# res = clf.predict([[0.19, 0.18, 0.18, 0.19, 0.18, 0.18, 0.17, 0.16, 0.18, 0.17, 0.17, 0.17, 0.18]])
# print(res)

# print('predicting [[0.34, 0.34, 0.31, 0.32, 0.32, 0.32]]')
# res = clf.predict([[0.34, 0.34, 0.31, 0.32, 0.32, 0.32]])
# print(res)

# print('predicting [[0.19, 0.18, 0.18, 0.19, 0.18, 0.18]]')
# res = clf.predict([[0.19, 0.18, 0.18, 0.19, 0.18, 0.18]])
# print(res)

print('predicting [[0.34, 0.34, 0.31]]')
res = clf.predict([[0.34, 0.34, 0.31]])
print(res)

print('predicting [[0.19, 0.18, 0.18]]')
res = clf.predict([[0.19, 0.18, 0.18]])
print(res)

# print('predicting [[0.34]]')
# res = clf.predict([[0.34]])
# print(res)

# print('predicting [[0.19]]')
# res = clf.predict([[0.19]])
# print(res)