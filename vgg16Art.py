########################################################################################
# Sean Mullery 2017                                                                    # 
# A Neural Algorithm of Artistic Style.                                                # 
# Implemented in Tensorflow, based on the paper by Leon A. Gatys, Alexander S. Ecker   # 
# and Matthias Bethge.  https://arxiv.org/abs/1508.06576v2                             # 
# Using the VGG network developed by Karen Simonyan, Andrew Zisserman                  # 
# https://arxiv.org/abs/1409.1556                                                      # 
# VGG16 implemenation in TensorFlow by  Davi Frossard, 2016                            #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
# using the following model                                                            #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################
from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.ndimage
from scipy.misc import imread, imresize
from imagenet_classes import class_names
from copy import deepcopy

import os
from io import BytesIO
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML


class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs #SM from an input placeholder, a 4 d tensor[none, 224,224,3]
        self.convlayers() #SM: call class method convlayers to setup the convolutional layers (and pooling layers)
                        # of the network. It also performs the preprocessing which entails zeroing the mean separately
                        # of the Red, Green and Blue
        self.fc_layers() #SM: call class method fc_layers to setup the fully connected laryers of the network.
        self.probs = tf.nn.softmax(self.fc3l) #SM: setup a softmax output layer with fc3l as input.
        if weights is not None and sess is not None:
            self.load_weights(weights, sess) #SM: call the class method load_weights to load the weights of the network
                                            # which have been taken from a pre-trained network.


    def convlayers(self):
        self.parameters = [] #SM Create a list that we will assign all the weights and biases to later.

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean
            '''SM: I think this manages to subtract the mean value from every pixel in
                     every image that is input thus putting every imput image to a zero mean.
                     This is an op in the graph.'''

        # conv1_1
        with tf.name_scope('conv1_1') as scope: #SM: This names the layer in the graph
            '''SM:truncated_normal outputs random values from a truncated normal distribution, mean is default to zero
               This is creating a convolutional kernel with 3x3 spatial size, x3 for the RGB and there are 64 of
               these kernels. It's initialising the weights with random values with a mean of zero and std of 1e-1'''
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            '''SM: Computes a 2-D convolution given 4-D input and filter tensors. images is the input. kernel is the filter
               [1,1,1,1] are the strides along each dimension, use SAME Padding so with stride as it is the input and
               output will be the same size,'''
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            '''SM: setup the biases with all zeros, there are 64 of them, one for each kernel
            They are set to trainable but I'm assuming that we set the values to what has already been trained.
            The conv kernel earlier had no mention of trainable'''
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            '''SM: add the bias to the conv layer'''
            out = tf.nn.bias_add(conv, biases)

            '''The name of the layer conv1_1 is then set to the output the kernel(+bias) put through a ReLU'''
            self.conv1_1 = tf.nn.relu(out, name=scope)
            '''SM: The kernel and the biases are then added to the overall set of parameters. Quite why this is done
               is not immediately obvious but may be later on'''
            self.parameters += [kernel, biases]


        with tf.name_scope('cont1_1') as scope:
            self.storedContent1_1 = tf.placeholder(tf.float32,shape=[224,224,64])
            self.cont1_1 = self.conv1_1 - self.storedContent1_1


        # SM: insered following scope
        with tf.name_scope('styleMap1_1') as scope:
            featureMap = self.conv1_1
            featureMap  = tf.reshape(featureMap, [50176, 64])
            gram = tf.matmul(tf.matrix_transpose(featureMap), featureMap)
            print(tf.shape(gram))
            self.styleMap1_1 = tf.reshape(gram, [64, -1])

        with tf.name_scope('styleDiff1_1') as scope:
            self.storedStyle1 = tf.placeholder(tf.float32, shape=[64, 64])
            self.styleDiff1_1 = (tf.reduce_sum(tf.square(self.styleMap1_1 - self.storedStyle1))/(4 * (64 ** 2) * (50176 ** 2)))
#(4.12489 * 1e-13)
        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')

            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)


            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('cont1_2') as scope:
            self.storedContent1_2 = tf.placeholder(tf.float32, shape=[224, 224, 64])
            self.cont1_2 = self.conv1_2 - self.storedContent1_2

        # pool1
        self.pool1 = tf.nn.avg_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')


        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('cont2_1') as scope:
            self.storedContent2_1 = tf.placeholder(tf.float32,shape=[112,112,128])
            self.cont2_1 = self.conv2_1 - self.storedContent2_1

        # SM: insered following scope
        with tf.name_scope('styleMap2_1') as scope:
            featureMap = self.conv2_1
            featureMap  = tf.reshape(featureMap, [12544, 128])
            gram = tf.matmul(tf.matrix_transpose(featureMap), featureMap)
            print(tf.shape(gram))
            self.styleMap2_1 = tf.reshape(gram, [128, -1])

        with tf.name_scope('styleDiff2_1') as scope:
            self.storedStyle2 = tf.placeholder(tf.float32, shape=[128, 128])
            self.styleDiff2_1 = (tf.reduce_sum(tf.square(self.styleMap2_1 - self.storedStyle2))/(4 * (128 ** 2) * (12544 ** 2)))
#(1.03122 * 1e-13)
        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('cont2_2') as scope:
            self.storedContent2_2 = tf.placeholder(tf.float32,shape=[112,112,128])
            self.cont2_2 = self.conv2_2 - self.storedContent2_2



        # pool2
        self.pool2 = tf.nn.avg_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('cont3_1') as scope:
            self.storedContent3_1 = tf.placeholder(tf.float32,shape=[56,56,256])
            self.cont3_1 = self.conv3_1 - self.storedContent3_1


        # SM: insered following scope
        with tf.name_scope('styleMap3_1') as scope:
            featureMap = self.conv3_1
            featureMap  = tf.reshape(featureMap, [3136, 256])
            gram = tf.matmul(tf.matrix_transpose(featureMap), featureMap)
            print(tf.shape(gram))
            self.styleMap3_1 = tf.reshape(gram, [256, -1])

        with tf.name_scope('styleDiff3_1') as scope:
            self.storedStyle3 = tf.placeholder(tf.float32, shape=[256, 256])
            self.styleDiff3_1 = (tf.reduce_sum(tf.square(self.styleMap3_1 - self.storedStyle3))/(4 * (256 ** 2) * (3136 ** 2)))
#(2.57805 * 1e-12)

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('cont3_2') as scope:
            self.storedContent3_2 = tf.placeholder(tf.float32,shape=[56,56,256])
            self.cont3_2 = self.conv3_2 - self.storedContent3_2


        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('cont3_3') as scope:
            self.storedContent3_3 = tf.placeholder(tf.float32,shape=[56,56,256])
            self.cont3_3 = self.conv3_3 - self.storedContent3_3

        # pool3
        self.pool3 = tf.nn.avg_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('cont4_1') as scope:
            self.storedContent4_1 = tf.placeholder(tf.float32,shape=[28,28,512])
            self.cont4_1 = self.conv4_1 - self.storedContent4_1


        # SM: insered following scope
        with tf.name_scope('styleMap4_1') as scope:
            featureMap = self.conv4_1
            featureMap  = tf.reshape(featureMap, [784, 512])
            gram = tf.matmul(tf.matrix_transpose(featureMap), featureMap)
            print(tf.shape(gram))
            self.styleMap4_1 = tf.reshape(gram, [512, -1])

        with tf.name_scope('styleDiff4_1') as scope:
            self.storedStyle4 = tf.placeholder(tf.float32, shape=[512, 512])
            self.styleDiff4_1 = (tf.reduce_sum(tf.square(self.styleMap4_1 - self.storedStyle4))/(4 * (512 ** 2) * (784 ** 2)))
#(6.44514 * 1e-11) #
        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            #print('shape of conv4_2 pre-relu', out.get_shape())
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('cont4_2') as scope:
            self.storedContent4_2 = tf.placeholder(tf.float32,shape=[28,28,512])
            self.cont4_2 = self.conv4_2 - self.storedContent4_2


        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('cont4_3') as scope:
            self.storedContent4_3 = tf.placeholder(tf.float32,shape=[28,28,512])
            self.cont4_3 = self.conv4_3 - self.storedContent4_3


        # pool4
        self.pool4 = tf.nn.avg_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('cont5_1') as scope:
            self.storedContent5_1 = tf.placeholder(tf.float32,shape=[14,14,512])
            self.cont5_1 = self.conv5_1 - self.storedContent5_1


        #SM: insered following scope
        with tf.name_scope('styleMap5_1') as scope:
            featureMap = self.conv5_1
            featureMap  = tf.reshape(featureMap, [196, 512])
            gram = tf.matmul(tf.matrix_transpose(featureMap), featureMap)
            print(tf.shape(gram))
            self.styleMap5_1 = tf.reshape(gram, [512, -1])

        with tf.name_scope('styleDiff5_1') as scope:
            self.storedStyle5 = tf.placeholder(tf.float32, shape=[512, 512])
            self.styleDiff5_1 = (tf.reduce_sum(tf.square(self.styleMap5_1 - self.storedStyle5))/(4 * (512 ** 2) * (196 ** 2)))
#(4.0281 * 1e-10)

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('cont5_2') as scope:
            self.storedContent5_2 = tf.placeholder(tf.float32,shape=[14,14,512])
            self.cont5_2 = self.conv5_2 - self.storedContent5_2


        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('cont5_3') as scope:
            self.storedContent5_3 = tf.placeholder(tf.float32,shape=[14,14,512])
            self.cont5_3 = self.conv5_3 - self.storedContent5_3


        # pool5
        self.pool5 = tf.nn.avg_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            #shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([25088, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, 25088])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        '''SM:If the file is a .npz file, then a dictionary-like object is returned,
        containing {filename: array} key-value pairs, one for each file in the archive.'''
        keys = sorted(weights.keys()) #SM: the default is to sort by keys so the .keys() is not really required here.
        '''SM: what is a bit confusing is how the keys are in the right order. How does conv1_1_W come before conv1_1_b?
        I'm assuming that it's because the Capital W comes before the lowercase b in the ascii character table '''

        for i, k in enumerate(keys): #SM: enumerate returns a tuple, the enumeration of the key and then the key itself
            #print (i, k, np.shape(weights[k])) #SM: k in each case is a string such as conv2_2_W
            sess.run(self.parameters[i].assign(weights[k])) #SM: this seems very simple but obviously parameters[i]
                                                            #is a pointer of sorts to the weights variable in the graph

img_noise = np.random.uniform(size=(224, 224, 3)) + 127.0
'''The following functions taken from deepdream.ipynb'''

def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 1) * 255)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    PIL.Image.fromarray(a).save('VGGOUT2', 'bmp')  # SM: appended this to output image to folder.
    #display(Image(data=f.getvalue()))

def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''

    return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5





def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    #SM placeholder is a tensor that will always be fed
    #map applies tf.placeholder to all the argtypes passed in.
    #list turns the bunch of placeholder tensors into a list
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders) #SM: I'm assuming this creates a list called 'out' from the list 'placeholders'
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=sess)
        return wrapper
    return wrap

# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0) #SM: this turns a single image into a batch of 1 [1,height, width, channel]
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)


k = np.float32([1,4,6,4,1])
k = np.outer(k, k)
k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)
#SM: the eye (3x3 identity matrix) makes up another 2 dimensions so k5x5 is a 5x5x3x3

def lap_split(img):
    '''Split the image into lo and hi frequency components'''
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1,2,2,1])
        hi = img-lo2
    return lo, hi

def lap_split_n(img, n):
    '''Build Laplacian pyramid with n splits'''
    levels = []
    for i in range(n):
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]

def lap_merge(levels):
    '''Merge Laplacian pyramid'''
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            #SM the *4 seems to be the amount of smoothing
            img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi), [1,2,2,1])+ hi
    return img

def normalize_std(img, eps=1e-10):
    '''Normalize image by making its standard deviation = 1.0'''
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img/tf.maximum(std, eps)

def lap_normalize(img, scale_n=4):
    '''Perform the Laplacian pyramid normalization.'''
    img = tf.expand_dims(img,0)
    tlevels = lap_split_n(img, scale_n)
    tlevels = list(map(normalize_std, tlevels))
    out = lap_merge(tlevels)
    return out[0,:,:,:]

# Showing the lap_normalize graph with TensorBoard
lap_graph = tf.Graph()
with lap_graph.as_default():
    lap_in = tf.placeholder(np.float32, name='lap_in')
    lap_out = lap_normalize(lap_in)
#show_graph(lap_graph)

def calc_grad_tiled(img, t_grad, tile_size=224):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over
    multiple iterations.'''
    sz = tile_size
    h, w = img.shape[:2]#SM give me dimensions 0 and 1 i.e. height, width
    #print(h,w)
    sx, sy = np.random.randint(sz, size=2)#SM: create two random ints (uniform) between 0 to sz (512)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0) #SM: rolls the array elements a certain amount along a given axis
                                                #in this case shift the entire image sy along y axis and sx along x axis
                                                #note that in a roll elements that go off one end enter at the other.
    grad = np.zeros_like(img)#SM grad is now an array of zeros of the same shape as img.
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = np.zeros((224,224,3), dtype=float)
            hsub,wsub,chsub = img_shift[y:y+sz,x:x+sz].shape
            sub[:hsub,:wsub,:] = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {vgg.imgs:[sub]})
            grad[y:y+sz,x:x+sz] = g[0,0:hsub,0:wsub]
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def render_deepart(img0=img_noise, iter_n=10, step=1.5, octave_n=1, octave_scale=1.4,lap_n=4):
    img1 = imread('Van_Gogh_-_Starry_Night_-_Google_Art_Project.bmp', mode='RGB')
    img1 = imresize(img1, (224, 224))

    img2 = imread('glasshouse.jpg', mode='RGB')
    img2 = imresize(img2, (224, 224))

    styleDict = {'style1': vgg.styleMap1_1, 'style2': vgg.styleMap2_1, 'style3': vgg.styleMap3_1,
                 'style4': vgg.styleMap4_1, 'style5': vgg.styleMap5_1}
    print(styleDict['style1'])

    styleDict = sess.run(styleDict, feed_dict={vgg.imgs: [img1]})
    # temp = sess.run(vgg.styleMap1_1, feed_dict={vgg.imgs: [img1]})
    for i in styleDict:
        print('Shape', styleDict[i].shape)

    contentDict = {'cont1': vgg.conv1_1, 'cont2': vgg.conv2_1, 'cont3': vgg.conv3_1, 'cont4': vgg.conv4_2,
                   'cont5': vgg.conv5_1}
    contentDict = sess.run(contentDict, feed_dict={vgg.imgs: [img2]})
    for i in contentDict:
        print('shape', contentDict[i].shape)

    loss = ((tf.reduce_sum(tf.square(vgg.styleMap1_1 - styleDict['style1'])) / (4 * (64 ** 2) * (50176 ** 2))) * 0.2 + \
            (tf.reduce_sum(tf.square(vgg.styleMap2_1 - styleDict['style2'])) / (4 * (128 ** 2) * (12544 ** 2))) * 0.2 + \
            (tf.reduce_sum(tf.square(vgg.styleMap3_1 - styleDict['style3'])) / (4 * (256 ** 2) * (3136 ** 2))) * 0.2 + \
            (tf.reduce_sum(tf.square(vgg.styleMap4_1 - styleDict['style4'])) / (4 * (512 ** 2) * (784 ** 2))) * 0.2 + \
            (tf.reduce_sum(tf.square(vgg.styleMap5_1 - styleDict['style5'])) / (4 * (512 ** 2) * (196 ** 2))) * 0.2) + \
           tf.reduce_sum(tf.square(vgg.conv4_2 - contentDict['cont4'])) * 0.005




    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))

    #img_noise = np.random.uniform(0, 1, size=(224, 224, 3)) + 127.0
    t_grad = tf.gradients(loss, vgg.imgs)[0]
    #print(img_noise.shape)
    #img3 = imread('glasshouse.jpg', mode='RGB')

    #img = img3.astype(float)
    img = img0

    #print(img)
    # split the image into a number of octaves
    #img = img0
    octaves = []
    for i in range(octave_n - 1):
        hw = img.shape[0:2]
        lo = resize(img, np.int32(np.float32(hw) / octave_scale))
        print(hw)
        hi = img - resize(lo, hw)
        img = lo
        octaves.append(hi)

    # generate details octave by octave
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2]) + hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            #SM: we can call lap norm here if necessary.
            #g = lap_norm_func(g)
            img -= g * (step / (np.abs(g).mean() + 1e-7))
            print('.', end=' ')
        #clear_output()
        print(octave)
        showarray(img / 255.0)

#render_lapnorm(T(layer)[:,:,:,channel])




def render_deepart3(img0=img_noise, iter_n=10, step=1.5, octave_n=1, octave_scale=1.4,lap_n=4):
    art = imread('the_taking_of_christ448.png', mode='RGB')
    #art = imread('Van_Gogh_-_Starry_Night_-_Google_Art_Project448.png', mode='RGB')

    artS = imresize(art, (224, 224))

    photo = imread('Macbeth2013_SMull-small-048.png', mode='RGB')
    #photo = imread('glasshouse448.png', mode='RGB')
    #photo = imread('salmon448.png', mode='RGB')
    photoS = imresize(photo, (224, 224))/1.0

    styleDict = {'style1': vgg.styleMap1_1, 'style2': vgg.styleMap2_1, 'style3': vgg.styleMap3_1,
                 'style4': vgg.styleMap4_1, 'style5': vgg.styleMap5_1}

    print('Shape of artS is ', artS.shape)

    styleDict = sess.run(styleDict, feed_dict={vgg.imgs: [artS]})
    contentDict = { 'cont1_1': vgg.conv1_1}
    contentDict = sess.run(contentDict, feed_dict={vgg.imgs: [photoS]})

    loss = (vgg.styleDiff1_1 + vgg.styleDiff2_1 + vgg.styleDiff3_1 + vgg.styleDiff4_1 + vgg.styleDiff5_1)/5.0 + \
        tf.reduce_sum(tf.square(vgg.cont1_1))*0.007


    t_grad = tf.gradients(loss, vgg.imgs)[0]
    img = img0
    #img = photoS


    for i in range(0):
        g, score = sess.run([t_grad, loss], {vgg.imgs:[img], vgg.storedContent1_1: contentDict['cont1_1'][0], vgg.storedStyle1: styleDict['style1'] ,
                              vgg.storedStyle2: styleDict['style2'], vgg.storedStyle3: styleDict['style3'],
                              vgg.storedStyle4: styleDict['style4'], vgg.storedStyle5: styleDict['style5']})

        img -= (g * (step / (np.abs(g).mean() + 1e-7)))[0]
        #print(score, end='\n')
    showarray(img / 255.0)
    #img = imresize(img, (448,448),interp='bicubic')/1.0
    #img = photo/1.0
    #showarray(img / 255.0)
    img = np.random.uniform(low=-10.0, high=+10.0, size=(448, 448, 3)) + 127.0

    #showarray(img / 255.0)

    styleDictTensor = {'style1': vgg.styleMap1_1, 'style2': vgg.styleMap2_1, 'style3': vgg.styleMap3_1,
                 'style4': vgg.styleMap4_1, 'style5': vgg.styleMap5_1}

    for style in styleDictTensor:
        h, w = styleDict[style].shape
        styleDictTensor[style] = np.zeros((4,h,w))

    styleDict = {'style1': vgg.styleMap1_1, 'style2': vgg.styleMap2_1, 'style3': vgg.styleMap3_1,
                 'style4': vgg.styleMap4_1, 'style5': vgg.styleMap5_1}
    contentDict = {'cont1_1': vgg.conv1_1}

    for y in range(0,2):
        for x in range(0,2):
            artL = art[y*224:y*224+224, x*224:x*224+224]
            subStyleDict = sess.run(styleDict, feed_dict={vgg.imgs: [artL]})
            for style in subStyleDict:
                styleDictTensor[style][2*y + x] = deepcopy(subStyleDict[style])

    for style in styleDictTensor:
        styleDict[style]=styleDictTensor[style].mean(axis=0)
    print(img.dtype)
    sz = 224;
    '''

    '''
    for i in range(10000):
        #if i%20 == 0:
        sx, sy = np.random.randint(sz, size=2)  # SM: create two random ints (uniform) between 0 to sz (512)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        photo_shift = np.roll(np.roll(photo, sx, 1), sy, 0)
        for y in range(0,2):
            for x in range(0,2):
                photoL = photo_shift[y * 224:y * 224 + 224, x * 224:x * 224 + 224]
                imgL = img_shift[y * 224:y * 224 + 224, x * 224:x * 224 + 224]
                contentDict = {'cont1_1': vgg.conv1_1}
                contentDict = sess.run(contentDict, feed_dict={vgg.imgs: [photoL]})

                #for i in range(5):
                g, score = sess.run([t_grad, loss], {vgg.imgs: [imgL], vgg.storedContent1_1: contentDict['cont1_1'][0],
                                  vgg.storedStyle1: styleDict['style1'],
                                  vgg.storedStyle2: styleDict['style2'], vgg.storedStyle3: styleDict['style3'],
                                  vgg.storedStyle4: styleDict['style4'], vgg.storedStyle5: styleDict['style5']})

                imgL -= (g * (step / (np.abs(g).mean() + 1e-7)))[0]
                #if i==500:
                 #  step /= 100.0
                #imgL -= g[0]
                #print('.', end=' ')
                img_shift[y * 224:y * 224 + 224, x * 224:x * 224 + 224] = imgL
        if i%100 == 0:
            step=step/2.0
            print(i,' ', score,end='\n')
            showarray(img / 255.0)
        img = np.roll(np.roll(img_shift, -sx, 1), -sy, 0)
        photo = np.roll(np.roll(photo_shift, -sx, 1), -sy, 0)

    #showarray(img / 255.0)



    #octaves = []
    '''for i in range(octave_n - 1):
        hw = img.shape[0:2]
        lo = resize(img, np.int32(np.float32(hw) / octave_scale))
        print(hw)
        hi = img - resize(lo, hw)
        img = lo
        octaves.append(hi)
    '''
    # generate details octave by octave
    '''for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2]) + hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            #SM: we can call lap norm here if necessary.
            #g = lap_norm_func(g)
            img -= g * (step / (np.abs(g).mean() + 1e-7))
            print('.', end=' ')
        #clear_output()
        print(octave)
        showarray(img / 255.0)
    '''
if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32,shape=[None, 224, 224, 3]) #SM 4D tensor with the first dimension the image number
    #imgs = tf.placeholder(tf.float32)
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)
    #SM: Make a summary writer so that the graph can be visualised.
    #t_input = tf.placeholder(np.float32, name='input')
    #img_in_tensor= tf.expand_dims(t_input , 0)

    writer = tf.summary.FileWriter( '/home/smullery/PycharmProjects/VGGTF/', sess.graph)


    render_deepart3(iter_n=500, step = 2.0)



