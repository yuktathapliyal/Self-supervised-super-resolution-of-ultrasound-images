# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 20:36:20 2021

@author: Yukta
"""
import tensorflow as tf
import keras
import tensorflow_addons as tfa
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose,\
                                    GlobalAveragePooling2D, AveragePooling2D, MaxPool2D, UpSampling2D,\
                                    BatchNormalization, Activation, ReLU, LeakyReLU, Flatten, Dense, Input,\
                                    Add, Multiply, Concatenate, Softmax
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax, sigmoid
from keras.applications.vgg19 import VGG19
tf.keras.backend.set_image_data_format('channels_last')
import keras.backend as K

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class KernelGAN(object):
    def generator():
        input = Input(shape=(None, None, 1), batch_size=3)
        c_1 = tfa.layers.SpectralNormalization(Conv2D(filters = 64, kernel_size=7, strides=1, padding='valid',use_bias=False))(input)
        c_2 = tfa.layers.SpectralNormalization(Conv2D(filters = 64, kernel_size=5, strides=1, padding='valid',use_bias=False))(c_1)
        c_3 = tfa.layers.SpectralNormalization(Conv2D(filters = 64, kernel_size=3, strides=1, padding='valid',use_bias=False))(c_2)
        c_4 = tfa.layers.SpectralNormalization(Conv2D(filters = 64, kernel_size=1, strides=1, padding='valid',use_bias=False))(c_3)
        c_5 = tfa.layers.SpectralNormalization(Conv2D(filters = 64, kernel_size=1, strides=1, padding='valid',use_bias=False))(c_4)
        c_6 = tfa.layers.SpectralNormalization(Conv2D(filters =1, kernel_size=1, strides=2, padding='valid',use_bias=False))(c_5)
        return Model(inputs = input, outputs = c_6)
    
    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)
    
    def discriminator():
        input = Input(shape = (None,None,3))
        l = tfa.layers.SpectralNormalization(Conv2D(filters=64, kernel_size=7,use_bias=True))(input)
        for _ in range(1,6):
            l = tfa.layers.SpectralNormalization(Conv2D(filters=64, kernel_size=1,use_bias=True))(l)
            l = BatchNormalization()(l)
            l = ReLU()(l)
        l = tfa.layers.SpectralNormalization(Conv2D(filters=1, kernel_size=1,use_bias=True))(l)
        out = sigmoid(l)
        return Model(inputs = input, outputs = out)

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
   



