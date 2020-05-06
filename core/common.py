'''
@description: 
@author: liutaiting
@lastEditors: liutaiting
@Date: 2020-05-05 10:28:46
@LastEditTime: 2020-05-05 10:51:19
'''

import tensorflow as tf

class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training,self.trainable)
        return super().call(x, training)

def convolutional(input_layer, filters_shape, donwsample=False,activate=True,bn=True):
    if donwsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1,0)(1,0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        padding = 'same'
        strides = 1
    
    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)
    
    if bn: conv = BatchNormalization()(conv)
    if activate == True: conv = tf.nn.leaky_relu(conv,alpha=0.1)

    return conv

def residual_block(input_layer, input_channel, filter_num1, filter_num2):
    short_cut = input_layer
    conv = convolutional(input_layer,filters_shape=(1,1,input_channel,filter_num1))
    conv = convolutional(conv,filters_shape=(3,3,filter_num1,filter_num2))

    residual_output = short_cut + conv
    return residual_output

def upsample(input_layer):
    return tf.image.resize(input_layer,(input_layer.shape[1]*2,input_layer.shape[2]*2),method='nearest')