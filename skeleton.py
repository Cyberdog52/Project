import numpy as np
import time
import os
import datetime
import math
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


#change these parameters
input_dir = './train/'
input_file_format = 'dataTrain_%d.pkl'
#change this according to the number of files that need to be read
file_ids = range(1,59)

#the number of frames that are considered for a video
#if the number of frames is larger, the middle part of the list is considered
video_length = 50

file_names = [os.path.join( input_dir, input_file_format % i) for i in file_ids]
i = 0

X = []
y = []

#import all skeletons of all files
for file_name in file_names:
	
	print("Importing " + file_name)
	data = pickle.load(open(file_name, 'rb'))

	for sample in data:
		skeleton = sample['skeleton']
		length = skeleton.shape[0]
		deleted_frames_offset = int((length - video_length) / 2)
		skeleton = skeleton[deleted_frames_offset: deleted_frames_offset + video_length, :]
		skeleton = skeleton.reshape((-1,))
		#only add the skeleton if its the right shape
		if len(skeleton) == video_length * 180:
			X.append(skeleton)
			y.append(sample['label'])
		else:
			print("found skeleton with length " + str(len(skeleton)))

X = np.asarray(X)
y = np.asarray(y)

print("X shape: " + str(X.shape))
print("y shape: " + str(y.shape))

bestscore = 0
for estimators in [500, 1000, 1500,  2000]:
	for depth in [30, 50, 80, 100, 150]:

		print("Gridsearch. Estimators: " + str(estimators) + " Depth: " + str(depth))

		
		rf = RandomForestClassifier(n_estimators = estimators, max_depth=depth)
		scores = cross_val_score(estimator=rf, X=X, y=y, cv=5)

		m = np.mean(scores)
		s = np.std(scores)
		print("RF cv mean: " + str(m))
		print("RF cv std: " + str(s))

		if m > bestscore:
			print("Found best score!")
			bestscore = m