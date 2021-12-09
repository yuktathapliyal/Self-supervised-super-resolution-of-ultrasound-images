
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 19:04:45 2021

@author: Yukta
"""
import numpy as np
import cv2
from data_process import *
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose,\
                                    GlobalAveragePooling2D, AveragePooling2D, MaxPool2D, UpSampling2D,\
                                    BatchNormalization, Activation, ReLU, Flatten, Dense, Input,\
                                    Add, Multiply, Concatenate, Softmax
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax
from keras.applications.vgg19 import VGG19
tf.keras.backend.set_image_data_format('channels_last')
import keras.backend as K

class DWT_downsampling(tf.keras.layers.Layer):
    """
    Chintan, (2021) Image Denoising using Deep Learning [Github]. 
    https://github.com/chintan1995/Image-Denoising-using-Deep-Learning/blob/main/Models/MWCNN_256x256.ipynb
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, x):
  
        x1 = x[:, 0::2, 0::2, :] #x(2i−1, 2j−1)
        x2 = x[:, 1::2, 0::2, :] #x(2i, 2j-1)
        x3 = x[:, 0::2, 1::2, :] #x(2i−1, 2j)
        x4 = x[:, 1::2, 1::2, :] #x(2i, 2j)   

        x_LL = x1 + x2 + x3 + x4
        x_LH = -x1 - x3 + x2 + x4
        x_HL = -x1 + x3 - x2 + x4
        x_HH = x1 - x3 - x2 + x4

        return Concatenate(axis=-1)([x_LL, x_LH, x_HL, x_HH])

class IWT_upsampling(tf.keras.layers.Layer):
    """
    Chintan, (2021) Image Denoising using Deep Learning [Github]. 
    https://github.com/chintan1995/Image-Denoising-using-Deep-Learning/blob/main/Models/MWCNN_256x256.ipynb
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, x):
        
        x_LL = x[:, :, :, 0:x.shape[3]//4]
        x_LH = x[:, :, :, x.shape[3]//4:x.shape[3]//4*2]
        x_HL = x[:, :, :, x.shape[3]//4*2:x.shape[3]//4*3]
        x_HH = x[:, :, :, x.shape[3]//4*3:]

        x1 = (x_LL - x_LH - x_HL + x_HH)/4
        x2 = (x_LL - x_LH + x_HL - x_HH)/4
        x3 = (x_LL + x_LH - x_HL - x_HH)/4
        x4 = (x_LL + x_LH + x_HL + x_HH)/4 

        y1 = K.stack([x1,x3], axis=2)
        y2 = K.stack([x2,x4], axis=2)
        shape = K.shape(x)
        return K.reshape(K.concatenate([y1,y2], axis=-1), K.stack([shape[0], shape[1]*2, shape[2]*2, shape[3]//4]))

class Conv_block(tf.keras.layers.Layer):
    def  __init__(self, num_filters=64, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.num_filters=num_filters
        self.kernel_size=kernel_size
        self.initializer = tf.keras.initializers.Orthogonal()
        self.conv_1 = Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, padding='same',kernel_initializer=self.initializer)
        self.conv_2 = Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, padding='same',kernel_initializer=self.initializer)
        self.conv_3 = Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, padding='same',kernel_initializer=self.initializer)
        self.conv_4 = Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, padding='same',kernel_initializer=self.initializer)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size':self.kernel_size
        })
        return config

    def call(self, X):
        X = self.conv_1(X)
        X = ReLU()(X)
        X = self.conv_2(X)
        X = ReLU()(X)
        X = self.conv_3(X)
        X = ReLU()(X)
        X = self.conv_4(X)
        X = ReLU()(X)

        return X

class model_class(object):
  
  def PM_1():
    
    input = Input(shape=(None, None, 3))                              
    cb_1 = Conv_block(num_filters = 64)(input)                        
    dwt = DWT_downsampling()(cb_1)                                    
    cb_2 = Conv_block(num_filters=64)(dwt)                            
    c_1 = Conv2D(filters = 256, kernel_size=3, strides=1, padding='same', activation='relu',kernel_initializer=tf.keras.initializers.Orthogonal())(cb_2)                                                                
    iwt = IWT_upsampling()(c_1)                                      
    cb_3 = Conv_block(num_filters=64)(Add()([iwt, cb_1]))             
    c_2 = Conv2D(filters = 3, kernel_size=3, strides=1, padding='same', activation='linear',kernel_initializer=tf.keras.initializers.Orthogonal())(cb_3)
    output = tf.keras.layers.Add()([c_2, input])
     
    return Model(inputs = input, outputs = output)
      
  def predict_image(image, scale_factor, model):

    """
    This function predicts the original image on the trained model.
    It takes the original image and interpolates it by scale_factor, expands image dimesnions to 4d, takes prediction,
    and outputs 8 bit image.
    """
    image_upscaled = cv2.resize(image, None, fx = scale_factor, fy = scale_factor, interpolation = cv2.INTER_CUBIC)
    image_upscaled = np.expand_dims(image_upscaled, axis = 0)
    super_image = model.predict(image_upscaled)
    super_image = np.squeeze(super_image, axis=0)
    super_image = cv2.convertScaleAbs(super_image)

    return super_image

  def custom_loss(y_true, y_pred):    
    mae_loss = mae_loss_object(y_true, y_pred)
    vgg_loss = K.mean(K.square(vgg_model(y_true) - vgg_model(y_pred)))
    vgg_loss_adjusted = (1 - (1/(1+vgg_loss)))*10
    
    return mae_loss + vgg_loss_adjusted
