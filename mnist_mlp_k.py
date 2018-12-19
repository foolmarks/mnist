######################################################
# Multi-Layer Perceptron Classifier for MNIST dataset
# Mark Harvey
# Dec 2018
######################################################
import os
import sys
import tensorflow as tf


# Returns the directory the current script (or interpreter) is running in
def get_script_directory():
    path = os.path.realpath(sys.argv[0])
    if os.path.isdir(path):
        return path
    else:
        return os.path.dirname(path)


# hyper-parameters
LEARNRATE = 0.0001
EPOCHS = 3
BATCHSIZE = 100


#####################################################
# Set up directories
#####################################################
# create a directory for the MNIST dataset if it doesn't already exist
SCRIPT_DIR = get_script_directory()
MNIST_DIR = os.path.join(SCRIPT_DIR, 'mnist_dir')
if not (os.path.exists(MNIST_DIR)):
    os.makedirs(MNIST_DIR)
    print("Directory " , MNIST_DIR ,  "created ") 


#####################################################
# Dataset preparation
#####################################################
# download of dataset, will only run if doesn't already exist in disk
mnist_dataset = tf.keras.datasets.mnist.load_data(path=os.path.join(MNIST_DIR, 'mnist_data') )

"""
The split into training & test datasets is already done for us by
the tf.keras.datasets.mnist.load_data function which returns tuples
of Numpy arrays - 60k images in training set, 10k in test dataset
 - x_train: set of 60k training images
 - y_train: set of 60k training labels
 - x_test : set of 10k training images 
 - y_test : set of 10k training labels
"""
(x_train, y_train), (x_test, y_test) = mnist_dataset

"""
You should always know what your dataset looks like, so lets print some 
# information about it...
"""
print("The training dataset has {img} images and {lbl} labels".format(img=len(x_train), lbl=len(y_train)))
print("The test dataset has {img} images and {lbl} labels".format(img=len(x_test), lbl=len(y_test)))
print("The training dataset shape is: {shp}".format(shp=x_train.shape))
print("The shape of each member of the training data is: {shp}".format(shp=x_train[0].shape))
print("The datatype of each pixel of the images is: {dtyp}".format(dtyp=x_train[0].dtype))
print("The shape of each label is: {shp}".format(shp=y_train[0].shape))
print("The datatype of each label is: {dtyp}".format(dtyp=y_train[0].dtype))

"""
Based on this information, we know that we have some work to do on the dataset..
 - we can't input a 28x82 image to a MLP, we need to flatten the data to a 784 element vector
   where 784 = 28 x 28.
 - we should scale the pixel data from its current range of 0:255 to 0:1.
 - the labels are integers, our MLP outputs 10 probabilities, each from 0 to 1. We need to
   convert the integer labels into one-hot encoded vectors of 10 elements.
"""

# flatten the images
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# The image pixels are 8bit integers (uint8)
# scale them from range 0:255 to range 0:1
x_train = x_train / 255.0
x_test = x_test / 255.0

# one-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# calculate total number of batches
total_batches = int(len(x_train)/BATCHSIZE)



print('\nThe datasets now look like this:')
print("The training dataset shape is: {shp}".format(shp=x_train.shape))
print("The training labels shape is: {shp}".format(shp=y_train.shape))
print("The shape of each member of the training data is: {shp}".format(shp=x_train[0].shape))
print("The shape of each label is: {shp}".format(shp=y_train[0].shape))
print("The datatype of each label is: {dtyp}".format(dtyp=y_train[0].dtype))


#####################################################
# Create the Computational graph
#####################################################
# in this sections, we define the MLP, placeholders for feeding in data
# and the loss and optimizer functions


# define placeholders for the input data & labels
x = tf.placeholder('float32', [None, 784])
y = tf.placeholder('float32', [None,10])


# dense, fully-connected layer of 196 nodes, reLu activation
input_layer = tf.layers.dense(inputs=x, units=196, activation=tf.nn.relu)
# dense, fully-connected layer of 10 nodes, softmax activation
prediction = tf.layers.dense(inputs=input_layer, units=10, activation=tf.nn.softmax)


# Define a cross entropy loss function
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=prediction, onehot_labels=y))

# Define the optimizer function
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNRATE).minimize(loss)

# accuracy
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#####################################################
# Create & run the graph in a Session
#####################################################


# Launch the graph
with tf.Session() as sess:

    sess.run(tf.initializers.global_variables())

    # Training cycle with training data
    for epoch in range(EPOCHS):
        print ("Epoch:", epoch)

        # process all batches
        for i in range(total_batches):
            
            # fetch a batch from training dataset
            batch_x, batch_y = x_train[i*BATCHSIZE:i*BATCHSIZE+BATCHSIZE], y_train[i*BATCHSIZE:i*BATCHSIZE+BATCHSIZE]

            # calculate training accuracy every 100 batches
            if i % 100 == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                print (" Batch:", i, ' Accuracy: ', acc)

            # Run graph for optimization - i.e. do the training
            sess.run([optimizer, loss, accuracy], feed_dict={x: batch_x, y: batch_y})

    print("Training Finished!")

    # Evaluation with test data
    print ("Accuracy of trained network with test data:", sess.run(accuracy, feed_dict={x: x_test, y: y_test}))

print ("FINISHED!")

