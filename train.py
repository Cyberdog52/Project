

import numpy as np
import time
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tflearn
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data
import datetime
import math
import pickle
from tensorflow.contrib import learn
import tensorflow as tf

def main(unused_argv):
    
        validation_data = X_test
        validation_labels = y_test

        # Get input dimensionality.
        IMAGE_HEIGHT = X_train.shape[1]
        IMAGE_WIDTH = X_train.shape[2]
        NUM_CHANNELS = X_train.shape[3]

        # Placeholder variables are used to change the input to the graph.
        # This is where training samples and labels are fed to the graph.
        # These will be fed a batch of training data at each training step
        # using the {feed_dict} argument to the sess.run() call below.

  

        input_samples_op = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS], name="input_samples")
        input_label_op = tf.placeholder(tf.int32, shape=[None], name="input_labels")
        # Some layers/functions have different behaviours during training and evaluation.
        # If model is in the training mode, then pass True.
        mode = tf.placeholder(tf.bool, name="mode")
        # loss_avg and accuracy_avg will be used to update summaries externally.
        # Since we do evaluation by using batches, we may want average value.
        # (1) Keep counting number of correct predictions over batches.
        # (2) Calculate the average value, evaluate the corresponding summaries
        # by using loss_avg and accuracy_avg placeholders.
        loss_avg = tf.placeholder(tf.float32, name="loss_avg")
        accuracy_avg = tf.placeholder(tf.float32, name="accuracy_avg")

        # Call the function that builds the network. You should pass all the
        # parameters that controls external inputs.
        # It returns "logits" layer, i.e., the top-most layer of the network.
        logits = conv_model_with_layers_api(input_samples_op, dropout_rate, mode)

        # Optional:
        # Tensorflow provides a very simple and useful API (summary) for
        # monitoring the training via tensorboard
        # (https://www.tensorflow.org/get_started/summaries_and_tensorboard)
        # However, it is not trivial to visualize average accuracy over whole
        # dataset. Create two tensorflow variables in order to count number of
        # samples fed and correct predictions made. They are attached to
        # a summary op (see below).
        counter_correct_prediction = tf.Variable(0, name='counter_correct_prediction', trainable=False)
        counter_samples_fed = tf.Variable(0, name='counter_samples_fed', trainable=False)

        # Loss calculations: cross-entropy
        with tf.name_scope("cross_entropy_loss"):
            # Takes predictions of the network (logits) and ground-truth labels
            # (input_label_op), and calculates the cross-entropy loss.
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=input_label_op))

        # Accuracy calculations.
        with tf.name_scope("accuracy"):
            # Return list of predictions (useful for making a submission)
            predictions = tf.argmax(logits, 1, name="predictions")
            # Return a bool tensor with shape [batch_size] that is true for the
            # correct predictions.
            correct_predictions = tf.nn.in_top_k(logits, input_label_op, 1)
            # Calculate the accuracy per minibatch.
            batch_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            # Number of correct predictions in order to calculate average accuracy afterwards.
            num_correct_predictions = tf.reduce_sum(tf.cast(correct_predictions, tf.int32))


        def do_evaluation(sess, samples, labels):
            '''
            Evaluation function.
            @param sess: tensorflow session object.
            @param samples: input data (numpy tensor)
            @param labels: ground-truth labels (numpy array)
            '''
            batches = data_iterator(samples, labels, batch_size)
            # Keep track of this run.
            counter_accuracy = 0.0
            counter_loss = 0.0
            counter_batches = 0
            for batch_samples, batch_labels in batches:
                counter_batches += 1
                feed_dict = {input_samples_op: batch_samples,
                             input_label_op: batch_labels,
                             mode: False}
                results = sess.run([loss, num_correct_predictions], feed_dict=feed_dict)
                counter_loss += results[0]
                counter_accuracy += results[1]
            return (counter_loss/counter_batches, counter_accuracy/(counter_batches*batch_size))

        # Create summary ops for monitoring the training.
        # Each summary op annotates a node in the computational graph and collects
        # data data from it.
        summary_trian_loss = tf.summary.scalar('loss', loss)
        summary_train_acc = tf.summary.scalar('accuracy_training', batch_accuracy)
        summary_avg_accuracy = tf.summary.scalar('accuracy_avg', accuracy_avg)
        summary_avg_loss = tf.summary.scalar('loss_avg', loss_avg)

        # Group summaries.
        summaries_training = tf.summary.merge([summary_trian_loss, summary_train_acc])
        summaries_evaluation = tf.summary.merge([summary_avg_accuracy, summary_avg_loss])

        # Generate a variable to contain a counter for the global training step.
        # Note that it is useful if you save/restore your network.
        global_step = tf.Variable(1, name='global_step', trainable=False)

        # Create optimization op.
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.05, beta1=0.9, beta2=0.999)
            train_op = optimizer.minimize(loss, global_step=global_step)

        # For saving/restoring the model.
        # Save important ops (which can be required later!) by adding them into
        # the collection. We will use them in order to evaluate our model on the test
        # data after training.
        # See tf.get_collection for details.
        tf.add_to_collection('predictions', predictions)
        tf.add_to_collection('input_samples_op', input_samples_op)
        tf.add_to_collection('mode', mode)

        # Create session object
        sess = tf.Session()
        # Add the ops to initialize variables.
        init_op = tf.global_variables_initializer()
        # Actually intialize the variables
        sess.run(init_op)

        # Register summary ops.
        train_summary_dir = os.path.join(model_dir, "summary", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        valid_summary_dir = os.path.join(model_dir, "summary", "validation")
        valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(max_to_keep=20)
        
        import matplotlib.pyplot as plt
        plt.figure()
        accurracy_list = []
        accurracy_loss = []
        validation_list = []
        validation_loss = []

        #index that counts which data is currently in memory
        data_index = 0

        # Define counters in order to accumulate measurements.
        counter_correct_predictions_training = 0.0
        counter_loss_training = 0.0
        for epoch in range(1, num_epochs+1):
            # Generate training batches
            training_batches = data_iterator(X_train, y_train, batch_size, 1)
            # Training loop.
            for batch_samples, batch_labels in training_batches:
                step = tf.train.global_step(sess, global_step)
                if (step%checkpoint_every_step) == 0 and not train:
                    ckpt_save_path = saver.save(sess, os.path.join(model_dir, 'model'), global_step)
                    print("Model saved in file: %s" % ckpt_save_path)

                # This dictionary maps the batch data (as a numpy array) to the
                # placeholder variables in the graph.
                feed_dict = {input_samples_op: batch_samples,
                             input_label_op: batch_labels,
                             mode: True}

                # Run the optimizer to update weights.
                # Note that "train_op" is responsible from updating network weights.
                # Only the operations that are fed are evaluated.
                # Run the optimizer to update weights.
                train_summary, correct_predictions_training, loss_training, _ = sess.run([summaries_training, num_correct_predictions, loss, train_op], feed_dict=feed_dict)
                # Update counters.
                counter_correct_predictions_training += correct_predictions_training
                counter_loss_training += loss_training
                # Write summary data.
                train_summary_writer.add_summary(train_summary, step)

                # Occasionally print status messages.
                if (step%print_every_step) == 0:
                    # Calculate average training accuracy.
                    accuracy_avg_value_training = counter_correct_predictions_training/(print_every_step*batch_size)
                    loss_avg_value_training = counter_loss_training/(print_every_step)
                    # [Epoch/Iteration]
                    
                    counter_correct_predictions_training = 0.0
                    counter_loss_training = 0.0
                    # Report
                    # Note that accuracy_avg and loss_avg placeholders are defined
                    # just to feed average results to summaries.
                    summary_report = sess.run(summaries_evaluation, feed_dict={accuracy_avg:accuracy_avg_value_training, loss_avg:loss_avg_value_training})
                    train_summary_writer.add_summary(summary_report, step)
                    
                    accurracy_list.append(accuracy_avg_value_training)
                    accurracy_loss.append(loss_avg_value_training)

                    print("[%d/%d] [Training] Accuracy: %.3f, Loss: %.3f" % (epoch, step, accuracy_avg_value_training, loss_avg_value_training))
                    

                if (step%evaluate_every_step) == 0 and train:
                    # Calculate average validation accuracy.
                    (loss_avg_value_validation, accuracy_avg_value_validation) = do_evaluation(sess, validation_data, validation_labels)
                    # Report
                    summary_report = sess.run(summaries_evaluation, feed_dict={accuracy_avg:accuracy_avg_value_validation, loss_avg:loss_avg_value_validation})
                    valid_summary_writer.add_summary(summary_report, step)
                    validation_list.append(accuracy_avg_value_validation)
                    validation_loss.append(loss_avg_value_validation)

                    print("[%d/%d] [Validation] Accuracy: %.3f, Loss: %.3f" % (epoch, step, accuracy_avg_value_validation, loss_avg_value_validation))

                if (step%switch_data_every_step) == 0:
                    data_index = data_index + 1
                    switchData(data_index)

#utils, helper functions

#######################################################################
## Helper functions.
#######################################################################
def data_iterator(data, labels, batch_size, num_epochs=1, shuffle=True):
    """
    A simple data iterator for samples and labels.
    @param data: Numpy tensor where the samples are in the first dimension.
    @param labels: Numpy array.
    @param batch_size:
    @param num_epochs:
    @param shuffle: Boolean to shuffle data before partitioning the data into batches.
    """
    data_size = data.shape[0]
    for epoch in range(num_epochs):
        # shuffle labels and features
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_samples = data[shuffle_indices]
            shuffled_labels = labels[shuffle_indices]
        else:
            shuffled_samples = data
            shuffled_labels = labels
        for batch_idx in range(0, data_size-batch_size, batch_size):
            batch_samples = shuffled_samples[batch_idx:batch_idx + batch_size]
            batch_labels = shuffled_labels[batch_idx:batch_idx + batch_size]
            yield batch_samples, batch_labels

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

#models

def conv_model_with_layers_api(input_layer, dropout_rate, mode):
    """
    Builds a model by using tf.layers API.

    Note that in mnist_fc_with_summaries.ipynb weights and biases are
    defined manually. tf.layers API follows similar steps in the background.
    (you can check the difference between tf.nn.conv2d and tf.layers.conv2d)
    """

    #input_layer = img_aug(input_layer)

    with tf.name_scope("network"):
        # Convolutional Layer #1
        # Computes 32 features using a 7x7 filter with ReLU activation.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 64, 64, 1]
        # Output Tensor Shape: [batch_size, 64, 64, 32]
        with tf.name_scope("cnn1"):
            net = tf.layers.conv2d(inputs=input_layer,filters=32,kernel_size=[7, 7],padding="same",activation=tf.nn.relu)

        # Pooling Layer #1
        # First max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 64, 64, 32]
        # Output Tensor Shape: [batch_size, 32, 32, 32]
        with tf.name_scope("pooling1"): net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2
        # Computes 64 features using a 7x7 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 32, 32, 32]
        # Output Tensor Shape: [batch_size, 32, 32, 64]
        with tf.name_scope("cnn2"):net = tf.layers.conv2d( inputs=net,filters=64, kernel_size=[7, 7],padding="same",activation=tf.nn.relu)

        # Pooling Layer #2
        # Second max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 32, 32, 64]
        # Output Tensor Shape: [batch_size, 16, 16, 64]
        with tf.name_scope("pooling2"): net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
            
        # Convolutional Layer #3
        # Computes 128 features using a 5x5 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 16, 16, 128]
        # Output Tensor Shape: [batch_size, 16, 16, 128]
        with tf.name_scope("cnn3"):net = tf.layers.conv2d(inputs=net,filters=128,kernel_size=[5, 5], padding="same",activation=tf.nn.relu)

        # Pooling Layer #3
        # Second max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 16, 16, 128]
        # Output Tensor Shape: [batch_size, 8, 8, 128]
        with tf.name_scope("pooling3"):net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)


        # Convolutional Layer #4
        # Computes 128 features using a 5x5 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 8, 8, 128]
        # Output Tensor Shape: [batch_size, 8, 8, 256]
        with tf.name_scope("cnn4"):net = tf.layers.conv2d(inputs=net,filters=256,kernel_size=[5, 5], padding="same",activation=tf.nn.relu)

        # Pooling Layer #4
        # Second max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 8, 8, 256]
        # Output Tensor Shape: [batch_size, 4, 4, 256]
        with tf.name_scope("pooling4"):net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)


        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, 4, 4, 256]
        # Output Tensor Shape: [batch_size, 4 * 4 * 256]
        with tf.name_scope("flatten"):
            net = tf.reshape(net, [-1, 4 * 4 * 256])

        # Dense Layer
        # Densely connected layer with 1024 neurons
        # Input Tensor Shape: [batch_size, 8 * 8 * 64]
        # Output Tensor Shape: [batch_size, 512]
        with tf.name_scope("dense"):
            net = tf.layers.dense(inputs=net, units=768, activation=tf.nn.sigmoid)

        # Add dropout operation
        with tf.name_scope("dropout"):
            net = tf.layers.dropout(inputs=net, rate=dropout_rate, training=mode)
            
        with tf.name_scope("logits"):
            net = tf.layers.dense(inputs=net, units=20)
        
        return net

def img_aug(batch_samples):
    import random
    from scipy.ndimage import rotate
    from PIL import Image
    import cv2
    from scipy.ndimage.interpolation import rotate

    height = batch_samples.shape[1]
    width = batch_samples.shape[2]

    ret = np.zeros(batch_samples.shape)

    #calculate augmentation for every image
    for image_index in range(0, batch_samples.shape[0]):
        image = batch_samples[image_index, :, :, :]

        #randomly rotate by degrees
        angle = random.uniform(0, 5)
        if bool(random.getrandbits(1)):
            angle = 360.0 - angle
        image = rotate(image, angle, axes=(1, 0), reshape=False)

        #randomly pick a box inside the picture
        max_offset = 4
        offset_height = random.randint(0, max_offset)
        offset_width = random.randint(0, max_offset)


        image = image[offset_height:height-max_offset, offset_width:width-max_offset, :]
        image = cv2.resize(image, (width, height))
        
        #save image to return array
        ret[image_index,:,:,:] = image

    return ret

import gc
def switchData(data_index):

    print("Loading new augmentation data...")
    global X_train
    #free full memory for X_train
    del X_train
    #collect any unnecessary data
    gc.collect()

    #import new data
    if data_index % (augmentation_factor + 1) == 0:
        X_train = np.load("X_train.npy")
        print("Successfully reloaded Xtrain0")
    else:
        if train:
            X_train = np.load("Xtrainaug4." + str(data_index % (augmentation_factor + 1)) + ".npy")
        else:
            X_train = np.load("Xaug4."  + str(data_index % (augmentation_factor + 1)) + ".npy")

        print("Successfully loaded new augmentation data " + str(data_index % (augmentation_factor + 1)))

#params

#toggle if you only train on part (True) or if you want to run on whole set (False)
train = False

learning_rate = 0.0005
batch_size = 113 #131 #113
num_epochs = 23
print_every_step = 200 #197 #100
evaluate_every_step = 200 #197 #100
checkpoint_every_step = 539 #539: data_length / batchsize -> checkpoint after 1 epoch
log_dir = './runs/'
dropout_rate = 0.75

#teiler von 60907
#  1,7,11,49,77,113,539,791,1243,5537,8701,60907

if not train:
    batch_size = 131
    print_every_step = 197  
    evaluate_every_step = 999999 #never 
    checkpoint_every_step = 591 #data_length / batchsize -> checkpoint after 1 epoch

#teiler von 77421
# 1,3,131,197,393,591,25807,77421


#decides how many augmentations are coming after X_train0 is used again
augmentation_factor = 2
switch_data_every_step = checkpoint_every_step

#prepare data

num_channels = 4
X = np.zeros((depth.shape[0], 64, 64, 4))
X[:,:,:,0] = depth
X[:,:,:,1:4] = rgb
y = y

#split train and validation set such that no person appears in both sets
train_index = []
test_index = []
#choose which group to pick as test set
testgroup_index = 1
for group_index in range(len(splits)):
    if group_index == testgroup_index:
        test_index = splits[group_index]
        if not train:
            train_index.extend(splits[group_index])
    else:
        train_index.extend(splits[group_index]) 
     
#split into training and test set
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

#remove X from the memory since it is not needed anymore
del X

#augment data
#create augmented data and save it as a npy file
for i in range(1, augmentation_factor+1):
    print("Augmenting data " + str(i) + " ... ")
    if not train:
        X_aug = img_aug(X_train)
        np.save("Xaug4." + str(i), X_aug)
        del X_aug
    else:
        X_aug = img_aug(X_train)
        np.save("Xtrainaug4." + str(i), X_aug)
        del X_aug

    gc.collect()

#save unaffected and unaugmented X_train as X_train0 to later use 

#Xaug1 = np.load("Xtaug4.1.npy")
#Xaug2 = np.load("Xtrainaug4.2.npy")
np.save('X_train', X_train)


print("Xtrain shape: " + str(X_train.shape))
print("Starting training")

#turn gpu off:
#CUDA_VISIBLE_DEVICES=""

#create a new saving directory
timestamp = str(int(time.time()))
model_dir = os.path.abspath(os.path.join(log_dir, timestamp))

#run tensorflow
tf.app.run()



