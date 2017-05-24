import numpy as np
import time
import os
import datetime
import math
import pickle


#change these parameters
input_dir = './train/'
input_file_format = 'dataTrain_%d.pkl'
#change this according to the number of files that need to be read
file_ids = range(1,4)

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
		X.append(skeleton)
		y.append(sample['label'])

X = np.asarray(X)
y = np.asarray(y)

print("X shape: " + str(X.shape))
print("y shape: " + str(y.shape))

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, max_depth=None)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator=rf, X=X, y=y, cv=5)

print("RF cv mean: " + str(np.mean(scores)))
print("RF cv std: " + str(np.std(scores)))