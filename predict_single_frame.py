
#---------------------------------
# Readme
#---------------------------------

#Predicts on the given model single frames
#takes 9 frames and let each vote for their class, take class that has most votes

# takes less than a minute to run


#---------------------------------
# Import dependencies
#---------------------------------

import numpy as np
import time
import os
import datetime
import math
import pickle
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tflearn
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data
import datetime
import math
from tensorflow.contrib import learn
import tensorflow as tf

#---------------------------------
# Helper methods
#---------------------------------

#the number of frames that are considered for a video
#if the number of frames is larger, the middle part of the list is considered
#if it is smaller, the sample is discarded
#chose 9 because this is the size of the fellowship of the ring (and uneven)
video_length = 9


def import_files(file_names):
	X = []
	#import all skeletons of all files
	for file_name in file_names:
		
		print("Importing " + file_name)
		data = pickle.load(open(file_name, 'rb'))

		for sample in data:
			depth = sample['depth']
			rgb = sample['rgb']
			value = np.zeros((video_length, 64, 64, 4))
			#skeleton = interpolate_skeleton(skeleton)
			length = depth.shape[0]
			if length >= video_length:
				deleted_frames_offset = int((length - video_length) / 2)
				depth = depth[deleted_frames_offset: deleted_frames_offset + video_length]
				rgb = rgb[deleted_frames_offset: deleted_frames_offset + video_length]
				value[:,:,:,0] = depth.reshape((9, 64, 64))
				value[:,:,:,1:4] = rgb
				X.extend(value)
			else:
				print("Found video that is only " + str(length) + " long, ignoring this")            
	return np.asarray(X)


# helper method

def data_iterator_samples(data, batch_size):
	"""
	A simple data iterator for samples.
	@param data: Numpy tensor where the samples are in the first dimension.
	@param batch_size:
	@param num_epochs:
	"""
	data_size = data.shape[0]
	for batch_idx in range(0, data_size, batch_size):
		batch_samples = data[batch_idx:batch_idx + batch_size]
		yield batch_samples

# will be called by tf.app.run()
# predicts data based on restored model

def main(unused_argv):

	sess = tf.Session()
	# Restore computation graph.
	saver = tf.train.import_meta_graph(modelPath + '.meta')
	# Restore variables.
	saver.restore(sess,modelPath)
	# Restore ops.
	predictions = tf.get_collection('predictions')[0]
	input_samples_op = tf.get_collection('input_samples_op')[0]
	mode = tf.get_collection('mode')[0]

	def do_prediction(sess, samples):
		batches = data_iterator_samples(samples, batch_size)
		test_predictions = []
		for batch_samples in batches:
			feed_dict = {input_samples_op: batch_samples,
						 mode: False}
			test_predictions.extend(sess.run(predictions, feed_dict=feed_dict))
		return test_predictions

	y_val =  do_prediction(sess, X_val)
	y_val = np.asarray(y_val)

	#make sure that classes are between 1-20 and not 0-19 (which was needed for the cnn)
	y_val += 1

	#split y_val into chunks of length video_length
	predict_chunks = (y_val[i:i+video_length] for i in range(0, len(y_val), video_length))
	#let each of the video frames vote and pick the one that was voted the most
	y_voted = []
	for chunk in predict_chunks:
		counts = np.bincount(chunk)
		y_voted.append(np.argmax(counts))
	y_val = np.asarray(y_voted)

	# Write to csv
	import csv
	import time
	valString = "-Val-%.2f" % 0.64
	with open('submission-'+(time.strftime('%Y-%m-%d-%a-%Hh%Mmin'))+valString+'.csv', "w") as ofile:
		writer = csv.writer(ofile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		tmp = writer.writerow(['Id' , 'Prediction'])
		for i in range(0, y_val.shape[0]):
			tmp = writer.writerow([i+1, y_val[i]])


#change this path to select the appropriate model
modelPath = './runs/1496088195/model-15848'

#---------------------------------
# Import data
#---------------------------------

#change these parameters
input_dir = './test/'
input_file_format = 'newTest_%d.pkl'
#change this according to the number of files that need to be read
file_ids = range(1,23)
file_names = [os.path.join( input_dir, input_file_format % i) for i in file_ids]
   
#X_val = import_files(file_names)
#np.save('X_val_single_frame', X_val)
X_val = np.load('X_val_single_frame.npy')

print("X_val shape: " + str(X_val.shape))


#---------------------------------
# Run prediction of CNN
#---------------------------------

# call main

batch_size = 119

tf.app.run()










