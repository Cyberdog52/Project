
#---------------------------------
# Readme
#---------------------------------

# This script predicts the classes of the test files only based on the skeletons
# Assumes that all data is in ./train/ with newTrain_1.pkl until newTrain_76.pkl and in ./test/ with new_Test_1 until new_Train22.pkl
# If already gridsearched for interpolation kind, random forest tree estimators and depth and if it's better to take the middle frames or stretch them all to a fixed size and then take the middle of the frames
# it's best to only consider videos of skeletons with more than 50 frames (discard the rest, interpolate linearly for test set)
# if there are more than 50 frames per video, use the middle 50 frames (the gesture is most likely to be in there)

# this takes around 35mins to run on my pc


#---------------------------------
# Import dependencies
#---------------------------------

import numpy as np
import time
import os
import datetime
import math
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

#---------------------------------
# Helper methods
#---------------------------------

from scipy.interpolate import interp1d
#interpolates the skeletons to have video_length frames
def interpolate_skeleton(skeleton):
	length = skeleton.shape[0]

	x = np.linspace(1, length, num=length)
	y = skeleton
	interpolated_function = interp1d(x,y, kind='linear', axis=0) #try out quadratic, cubic is not as good as linear
	x_new = np.linspace(1, length, num=video_length)
	new_skeleton_video = interpolated_function(x_new)
	return new_skeleton_video

#the number of frames that are considered for a video
#if the number of frames is larger, the middle part of the list is considered
#if it is smaller, the skeletons are interpolated
video_length = 50

#imports all skeletons given the filenames
#returns X and y 
#optimal video_length 50
#if the video of the skeletons has more than video_length frames, the middle video_length are taken, the others discarded
# test: if there are less than video_length frames per skelton video, the video is interpolated such that the video has video_length frames
# train: if there are less than video_length frames per skelton video, this sample is discarded
def import_files(file_names, train=True):
	X = []
	y = []
	#import all skeletons of all files
	for file_name in file_names:
		
		print("Importing " + file_name)
		data = pickle.load(open(file_name, 'rb'))

		for sample in data:
			skeleton = sample['skeleton']
			#skeleton = interpolate_skeleton(skeleton)
			length = skeleton.shape[0]
			if length >= video_length:
				deleted_frames_offset = int((length - video_length) / 2)
				skeleton = skeleton[deleted_frames_offset: deleted_frames_offset + video_length, :]
				skeleton = skeleton.reshape((-1,))
				X.append(skeleton)
				if train:
					y.append(sample['label'])
			else:
				print("Found skeleton that is only " + str(length) + " long, interpolating this")			
				if not train:
					X.append(interpolate_skeleton(skeleton).reshape((-1,)))
	return (np.asarray(X), np.asarray(y))


#this does not seem to be as good as the first import method
#interpolates all videos to have video_length frames and then takes the new_length middle frames
#the best parameter for this import method seems to be 30 if video_length is 50
def import_files2(file_names, new_length=30):
	X = []
	y = []
	#import all skeletons of all files
	for file_name in file_names:
		
		print("Importing " + file_name)
		data = pickle.load(open(file_name, 'rb'))

		for sample in data:
			skeleton = sample['skeleton']
			#skeleton = interpolate_skeleton(skeleton)
			length = skeleton.shape[0]
			deleted_frames_offset = int((video_length - new_length) / 2)
			skeleton = interpolate_skeleton(skeleton)
			skeleton = skeleton[deleted_frames_offset: deleted_frames_offset + new_length, :]
			skeleton = skeleton.reshape((-1,))
			X.append(skeleton)
			y.append(sample['label'])
	return (np.asarray(X), np.asarray(y))


#---------------------------------
# Import train data
#---------------------------------

#change these parameters
input_dir = './train/'
input_file_format = 'newTrain_%d.pkl'
#change this according to the number of files that need to be read
file_ids = range(1,77)

file_names = [os.path.join( input_dir, input_file_format % i) for i in file_ids]
		
X, y = import_files(file_names)

np.save('y_skeleton', y)
np.save('X_skeleton', X)

print("X shape: " + str(X.shape))
print("y shape: " + str(y.shape))

#---------------------------------
# Do 5-fold cross-validation
#---------------------------------

rf = RandomForestClassifier(n_estimators = 750, max_depth=50)

print("Doing cross-validation:")
scores = cross_val_score(estimator=rf, X=X, y=y, cv=5)
m = np.mean(scores)
s = np.std(scores)

print("RF cv mean: " + str(m))
print("RF cv std: " + str(s))


#---------------------------------
# Import test data
#---------------------------------

input_dir = './test/'
input_file_format = 'newTest_%d.pkl'
#change this according to the number of files that need to be read
file_ids = range(1,23)
#the number of frames that are considered for a video
#if the number of frames is larger, the middle part of the list is considered
file_names = [os.path.join( input_dir, input_file_format % i) for i in file_ids]

X_val, _ = import_files(file_names)
np.save('X_val_skeleton', X_val)


#---------------------------------
# Predict and write submission file as csv
#---------------------------------

print("Fitting random forest with all data...")
rf.fit(X,y)
y_val = rf.predict(X_val)

# Write to csv
import csv
import time
if m is None: #if cross-validation is deleted above, then m is None. Delete CV if you want it faster
	m = 0.0000
valString = "-Val-%.2f" % m
y_val = np.asarray(y_val)
with open('submission-'+(time.strftime('%Y-%m-%d-%a-%Hh%Mmin'))+valString+'.csv', "w") as ofile:
    writer = csv.writer(ofile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    tmp = writer.writerow(['Id' , 'Prediction'])
    for i in range(0, y_val.shape[0]):
        tmp = writer.writerow([i+1, y_val[i]])

#---------------------------------
# Appendix
#---------------------------------


#use this for gridsearch of finding rf parameters

# bestscore = 0
# for estimators in [725, 750, 775]:
# 	for depth in [48, 50, 52]:

# 		print("Gridsearch. Estimators: " + str(estimators) + " Depth: " + str(depth))

		
# 		rf = RandomForestClassifier(n_estimators = estimators, max_depth=depth)
# 		scores = cross_val_score(estimator=rf, X=X, y=y, cv=5)

# 		m = np.mean(scores)
# 		s = np.std(scores)
# 		print("RF cv mean: " + str(m))
# 		print("RF cv std: " + str(s))

# 		if m > bestscore:
# 			print("Found best score!")
# 			bestscore = m