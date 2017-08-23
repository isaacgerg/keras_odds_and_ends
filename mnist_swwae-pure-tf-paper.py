'''Trains a stacked what-where autoencoder built on residual blocks on the
MNIST dataset.  It exemplifies two influential methods that have been developed
in the past few years.

The first is the idea of properly 'unpooling.' During any max pool, the
exact location (the 'where') of the maximal value in a pooled receptive field
is lost, however it can be very useful in the overall reconstruction of an
input image.  Therefore, if the 'where' is handed from the encoder
to the corresponding decoder layer, features being decoded can be 'placed' in
the right location, allowing for reconstructions of much higher fidelity.

References:
[1]
'Visualizing and Understanding Convolutional Networks'
Matthew D Zeiler, Rob Fergus
https://arxiv.org/abs/1311.2901v3

[2]
'Stacked What-Where Auto-encoders'
Junbo Zhao, Michael Mathieu, Ross Goroshin, Yann LeCun
https://arxiv.org/abs/1506.02351v8

The second idea exploited here is that of residual learning.  Residual blocks
ease the training process by allowing skip connections that give the network
the ability to be as linear (or non-linear) as the data sees fit.  This allows
for much deep networks to be easily trained.  The residual element seems to
be advantageous in the context of this example as it allows a nice symmetry
between the encoder and decoder.  Normally, in the decoder, the final
projection to the space where the image is reconstructed is linear, however
this does not have to be the case for a residual block as the degree to which
its output is linear or non-linear is determined by the data it is fed.
However, in order to cap the reconstruction in this example, a hard softmax is
applied as a bias because we know the MNIST digits are mapped to [0,1].

References:
[3]
'Deep Residual Learning for Image Recognition'
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
https://arxiv.org/abs/1512.03385v1

[4]
'Identity Mappings in Deep Residual Networks'
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
https://arxiv.org/abs/1603.05027v3

'''
from __future__ import print_function
import numpy as np
import tqdm

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# idg - adapted from https://github.com/fchollet/keras/blob/master/examples/mnist_swwae.py
# Runs in tensorflow
# Removed keras

def convresblock(x, nfeats=8, ksize=3, deconv=False):
    if not deconv:
        y = tf.layers.conv2d(x, nfeats, [ksize, ksize], padding='SAME')               
    else:
        y = tf.layers.conv2d_transpose(x, nfeats, [ksize, ksize], padding='SAME')               
    y = tf.nn.elu(y)
    return y


def getwhere(x):
    ''' Calculate the 'where' mask that contains switches indicating which
    index contained the max value when MaxPool2D was applied.  Using the
    gradient of the sum is a nice trick to keep everything high level.'''
    y_prepool, y_postpool = x    
    return tf.gradients(tf.reduce_sum(y_postpool), y_prepool)

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_train = mnist.train.images
x_test = mnist.test.images


x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# The size of the kernel used for the MaxPooling2D
pool_size = 2
# The total number of feature maps at each layer
nfeats = [8, 16, 32, 64, 128]
# The sizes of the pooling kernel at each layer
pool_sizes = np.array([1, 1, 1, 1, 1]) * pool_size
# The convolution kernel size
ksize = 3
# Number of epochs to train for
epochs = 10
# Batch size during training
batch_size = 128

if pool_size == 2:
    # if using a 5 layer net of pool_size = 2
    x_train = np.pad(x_train, [[0, 0], [2, 2], [2, 2], [0, 0]],
                     mode='constant')
    x_test = np.pad(x_test, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='constant')
    nlayers = 5
elif pool_size == 3:
    # if using a 3 layer net of pool_size = 3
    x_train = x_train[:, :, :-1, :-1]
    x_test = x_test[:, :, :-1, :-1]
    nlayers = 3
else:
    import sys
    sys.exit('Script supports pool_size of 2 and 3.')

# Shape of input to train on (note that model is fully convolutional however)
input_shape = (None, x_train.shape[1], x_train.shape[2], x_train.shape[3])
# The final list of the size of axis=1 for all layers, including input
nfeats_all = [input_shape[-1]] + nfeats

# First build the encoder, all the while keeping track of the 'where' masks
img_input = tf.placeholder(tf.float32, shape=input_shape)

# We push the 'where' masks to the following list
wheres = [None] * nlayers
poolingOutputs = [None] * nlayers
y = img_input
for i in range(nlayers):
    y_prepool = convresblock(y, nfeats=nfeats_all[i + 1], ksize=ksize)
    poolingOutputs[i] = tf.layers.max_pooling2d(y_prepool, [pool_sizes[i], pool_sizes[i]], [pool_sizes[i], pool_sizes[i]])
    wheres[i] = getwhere([y_prepool, poolingOutputs[i]])
    y = poolingOutputs[i]

# Now build the decoder, and use the stored 'where' masks to place the features
unpoolingOutputs = [None] * nlayers
for i in range(nlayers):
    ind = nlayers - 1 - i    
    in_shape = y.get_shape().as_list()
    out_shape = [in_shape[1]*pool_sizes[ind], in_shape[2]*pool_sizes[ind]]
    y = tf.image.resize_images(y, out_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)    
    y = tf.multiply(y, wheres[ind][0])
    unpoolingOutputs[ind] = convresblock(y, nfeats=nfeats_all[ind], ksize=ksize, deconv=True)
    y = unpoolingOutputs[ind];

# Define the model and it's mean square error loss, and compile it with Adam
l2m0 = tf.nn.l2_loss(tf.contrib.layers.flatten(unpoolingOutputs[1]) - tf.contrib.layers.flatten(poolingOutputs[0]))
l2m1 = tf.nn.l2_loss(tf.contrib.layers.flatten(unpoolingOutputs[2]) - tf.contrib.layers.flatten(poolingOutputs[1]))
l2m2 = tf.nn.l2_loss(tf.contrib.layers.flatten(unpoolingOutputs[3]) - tf.contrib.layers.flatten(poolingOutputs[2]))
l2m3 = tf.nn.l2_loss(tf.contrib.layers.flatten(unpoolingOutputs[4]) - tf.contrib.layers.flatten(poolingOutputs[3]))
loss_l2 = tf.nn.l2_loss(tf.contrib.layers.flatten(img_input) -  tf.contrib.layers.flatten(y))
L = loss_l2 + l2m0 + l2m1 + l2m2 + l2m3
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(L)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    N = x_train.shape[0]
    numBatches = N // batch_size
    for k in range(epochs):    
        for batchNum in tqdm.tqdm(range(numBatches)):
            batch = x_train[batchNum*batch_size : (batchNum+1)*batch_size, :, : ,:]
            loss, _= sess.run([loss_l2, train], feed_dict={img_input: batch})
            #print(loss)
           
    # Plot    
    x_recon = sess.run(y, feed_dict={img_input:batch[:25]})     
    x_plot = np.concatenate((batch[:25], x_recon), axis=1)
    x_plot = x_plot.reshape((5, 10, input_shape[-3], input_shape[-2]))
    x_plot = np.vstack([np.hstack(x) for x in x_plot])
    plt.figure()
    plt.axis('off')
    plt.title('Test Samples: Originals/Reconstructions')
    plt.imshow(x_plot, interpolation='none', cmap='gray')
    plt.savefig('reconstructions.png')
    plt.close('all')
    
    # Debug
    unpool = sess.run(unpoolingOutputs[1], feed_dict={img_input:batch[:25]})   
    plt.figure()
    plt.imshow(unpool[0,:,:,2])
    plt.title('unpool')
    plt.colorbar()
    pool = sess.run(poolingOutputs[0], feed_dict={img_input:batch[:25]})   
    plt.figure()
    plt.imshow(pool[0,:,:,2])
    plt.title('pool')
    plt.colorbar()
    plt.show()

print('Done')
