
import numpy as np
import time
import os
import math
import pickle
from PIL import Image

def readLargeFile(filePath):
    n_bytes = 2**31
    max_bytes = 2**31 - 1

    ## read
    bytes_in = bytearray(0)
    input_size = os.path.getsize(filePath)
    with open(filePath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)

#Masking with segmentation images
def mask(images, segment_images):
    assert(len(images.shape) == 4 and len(segment_images.shape) == 4)

    masked_images = np.zeros(images.shape)
    channels = images.shape[3]
    for i in range(images.shape[0]):
        mask = np.mean(segment_images[i], axis=2) > 150
        #mask3 =np. tile(mask, (3,1,1))
        #mask3= mask3.transpose((1,2,0))
        for c in range(channels):
            masked_images[i,:,:,c] = images[i,:,:,c] * mask
    return masked_images




from scipy.misc import imresize
def crop_image(img,tol=0):
    # img is image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def resize(arr):
    cropped_images = np.zeros((arr.shape[0], new_height, new_width, arr.shape[3]))
    for i in range(arr.shape[0]):
    	for j in range(arr.shape[3]):
            im = crop_image(arr[i,:,:,j])
            try:
            	cropped_images[i,:,:,j] = imresize(im, (new_height,new_width))
            except ValueError:
            	print("Found image that has only 0s at segmentation, only resizing image without cropping..")
            	cropped_images[i,:,:,j] = imresize(arr[i,:,:,j], (new_height,new_width))

    return cropped_images





#split into 4 groups where no subject appears in two groups
def splitIntoGroups(no_of_groups):
    print("Creating splits...")
    #create empty list of lists
    splits = []
    for i in range(no_of_groups):
        splits.append([])
        
    #split them into groups
    for i in range(len(subjects)):
        splits[subjects[i] % no_of_groups].append(i)   
    #print the length of each group and see if they are equally split
    for i in range(no_of_groups):
        print(len(splits[i]))
    return splits



#change these parameters
#input_dir = './train/'
input_dir = './test/'
#input_file_format = 'dataTrain_%d.pkl'
input_file_format = 'dataTest_%d.pkl'
#change this according to the number of files that need to be read
#file_ids = range(1,77)
file_ids = range(1,23)

output_file_format = 'newTest_%d.pkl'



file_names = [os.path.join( input_dir, input_file_format % i) for i in file_ids]
i = 0
for file_name in file_names:

	print("Importing " + file_name)

	data = pickle.load(open(file_name, 'rb'))

	new_data_list = []
	for sample in data:

		segment = sample['segmentation']
		depth = sample['depth'] 
		rgb = sample['rgb']

		#get rid of 4th dimension
		rgb = mask(rgb, segment)
		depth = mask(depth, segment)


		#resize all images such that they are centered
		new_height = 64
		new_width = 64

		#crop the images such that the people are centered
		segment = resize(segment)
		depth = np.uint(resize(depth))
		rgb = np.uint8(resize(rgb))
		#now the images have shape new_width and new_height


		new_data = {}
		new_data['rgb'] = rgb 
		new_data['depth'] = depth
		new_data['segmentation'] = segment
		new_data['skeleton'] = sample['skeleton']
		new_data['label'] = sample['label']
		new_data['length'] = sample['length']

		new_data_list.append(new_data)
	#index to store file with different names
	i += 1

	pickle.dump(new_data_list, open(os.path.join( input_dir, output_file_format % i), 'wb'))