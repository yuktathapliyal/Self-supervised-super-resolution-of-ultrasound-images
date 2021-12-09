# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 16:06:18 2021

@author: Yukta

run.py
"""

import argparse
import numpy as np
import cv2
import os
import sys
import pathlib
import numpy as np
import cv2
import tensorflow as tf
import time
import torch
from time import strftime, localtime
import timeit
from torch.nn import functional as F
import random
from proposed_model import model_class
from data_process import *
from KernelGAN import KernelGAN
from Kernel_process import *
from matplotlib import pyplot as plt
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

generator = KernelGAN.generator()
discriminator = KernelGAN.discriminator()
generator_optimizer = tf.keras.optimizers.Adam(learning_rate = 2e-4, beta_1=0.5, beta_2=0.999)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = 2e-4, beta_1=0.5, beta_2=0.999)

def train_step(input_image, epoch, args):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    original_image = input_image
    noisy_image = add_noise(input_image, args.sigma)
    cropped_img_list = image_crop([original_image,
                                   noisy_image], 
                                  args.crop_size,
                                  args.leave_as_is_probability, 
                                  args.shear_scale_prob, 
                                  args.center_crop_prob)
                                                                                  
    noisy_image_expand_axis = tf.expand_dims(cropped_img_list[1], axis =0)
    noisy_image_swapped_axis = swap_axis(noisy_image_expand_axis)
    original_image_expand_axis = tf.expand_dims(cropped_img_list[0], axis = 0)
    gen_output = generator(noisy_image_swapped_axis, training=True)
    gen_output = swap_axis(gen_output)
    disc_real_output = discriminator(original_image_expand_axis, training = True)
    disc_generated_output = discriminator(gen_output, training = True)
    gen_loss = KernelGAN.generator_loss(disc_generated_output)
    disc_loss = KernelGAN.discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_loss,
                                          generator.trainable_variables)
  
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)
  
  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
  
  # with summary_writer.as_default():
    # tf.summary.scalar('gen_total_loss', gen_loss, step=epoch)
    # tf.summary.scalar('disc_loss', disc_loss, step=epoch)

def fit(input_image, epochs, args):
  for epoch in range(epochs):
    start = time.time()

    # Training step
    train_step(input_image, epoch, args)
    
    # Saving (checkpointing) the model every 20 epochs
    # if (epoch + 1) % 100 == 0:
    # checkpoint.save(file_prefix=checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  # checkpoint.save(file_prefix=checkpoint_prefix)
  
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type = str, help = 'Directory containing (test) images')
    parser.add_argument('--output_path', type = str, help = 'Directory to store output images')
    parser.add_argument('--drop_lr', type = int, default = 0.5, help = 'Factor that reduces the learning rate as new_lr = lr * factor')
    parser.add_argument('--num_epochs', type = int, default = 1500, help = 'Number of epochs to run the model per image')
    parser.add_argument('--gradual_increase_value', type = int, default = 2, help = 'Value with which the images are gradually super-resolved. This gradual increase factor is inspired by Shocher, Assaf & Cohen, Nadav & Irani, Michal. (2018). Zero-Shot Super-Resolution Using Deep Internal Learning. 3118-3126. 10.1109/CVPR.2018.00329.')
    parser.add_argument('--sigma', type = int, default = 30, help = 'Standard deviation (spread or “width”) of the normal distribution, used in introducing random noise')
    parser.add_argument('--leave_as_is_probability', type = int, default = 0.2, help = 'This is the probability associated with augmentation of hr parent. A higher leave_as_is_probability reduces probability of random augmentation in hr parent.')
    parser.add_argument('--shear_scale_prob', type = int, default = 0, help = 'This the prabability associated with random shearing & scaling of HR parent during augmentations. A lower shear_scale_prob value prompts the model to increase the probability of random shearing & scaling, and vice-versa.')
    parser.add_argument('--crop_size', type = int, default = 96, help = 'This is the initial crop size to be considered. ')
    parser.add_argument('--SR_factor', type = int, default = 4, help = 'This is te super-resolution factor.')
    parser.add_argument('--center_crop_prob', type = int, default = 1, help = 'If center_crop_prob is small, more crops are taken from the center of the image. Else, if center_crop_prob is large, crops are taken randomly from the image, regardless of location.')
    parser.add_argument('--KernelGAN', type = bool, default = False, help = 'If true, KernelGAN first finds a image specific downsampling kernel and then performs super-resolution. If False, then cubic interpolation is used for obtaining lr_child from the hr_parent')
    parser.add_argument('--downsample_prob', type = int, default = 0.2, help = 'A lower downsample_prob creates higher probability taking cubic interpolation when KernelGAN is False. On the other hand, when this value is high, the lr_child is directly equated to hr_parent without interpolation.')

    args = parser.parse_args()
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    callback_list = [tf.keras.callbacks.ReduceLROnPlateau(monitor = 'loss',
                                                      factor = args.drop_lr,
                                                      patience = 100,
                                                      verbose = 1,
                                                      mode = 'min',
                                                      min_delta = 0.001,
                                                      cooldown = 20,
                                                      min_lr = 0.00000001),
                  tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                                                   min_delta = 0.0001,
                                                   patience = 350,
                                                   verbose = 1,
                                                   mode = 'min')
                      ]
    
    
    gradual_SR_list = get_gradual_factors(args.SR_factor, args.gradual_increase_value)
    print('Scaling gradually in order:', gradual_SR_list)    
    scale_factor = args.gradual_increase_value
    
    date_time = strftime('_%b_%d_%H_%M_%S', localtime())
    super_dir = args.output_path + '/' + date_time + '/' + 'result_images'
    kernel_dir = args.output_path + '/' + date_time + '/' + '_kernel_dir'
    os.makedirs(super_dir)
    if args.KernelGAN:
      os.makedirs(kernel_dir)
    start = timeit.default_timer()
    
    for file in os.listdir(args.input_path):

        image_path = os.path.join(args.input_path, '%s' %file)
        image = load_img(image_path)

        if args.KernelGAN:
          generator = KernelGAN.generator()
          discriminator = KernelGAN.discriminator()
          generator_optimizer = tf.keras.optimizers.Adam(learning_rate = 2e-4, beta_1=0.5, beta_2=0.999)
          discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = 2e-4, beta_1=0.5, beta_2=0.999)
          image_tf = tf.convert_to_tensor(image, dtype = tf.float32)
        
          tf.keras.backend.clear_session()
          fit(image_tf, 3000, args)
        
          # Convoluting the layers of learned Generator to extract the downsampling kernel
          delta = torch.Tensor([1.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
          for layer_idx, layer in enumerate(generator.layers):
              if layer_idx == 0:
                pass
              else:
                for weight_idx, weight in enumerate(layer.get_weights()):
                    if weight_idx == 0:
                        weight = np.transpose(weight, axes=[3,2,0,1])
                        weight = torch.from_numpy(weight)
                        curr_k = F.conv2d(delta, weight, padding = 13-1) if layer_idx == 1 else F.conv2d(curr_k, weight)
       
          curr_k = curr_k.squeeze().flip([0,1])
          final_kernel = post_process_k(curr_k,n = 40)
          tf.keras.backend.clear_session()

        print('Starting training for', file)
        model = model_class.PM_1()
        model.compile(loss = model_class.custom_loss, optimizer = Adam(learning_rate = 0.001))
        
        # Predicting the super-resolved image
        for i in range(len(gradual_SR_list)):
            img_name = str(file).split(sep = '.') 
            img_name = img_name[0]
            
            if args.KernelGAN:
              if i == 0:
                k = final_kernel
              else:
                k = analytic_kernel(final_kernel)
                save_final_kernel(k,img_name,i,kernel_dir)
            else:
              k = None

            if len(image.shape) == 4:
                image = np.squeeze(image, axis = 0)
            else:
                image = image

            model.fit(hr_lr_generator(image,
                                      scale_factor = scale_factor,
                                      final_kernel = k,
                                      shear_scale_prob = args.shear_scale_prob,
                                      leave_as_prob = args.leave_as_is_probability,
                                      center_crop_prob = args.center_crop_prob,
                                      crop_size = args.crop_size,
                                      sigma = args.sigma,
                                      downsample_prob = args.downsample_prob, 
                                      KernelGAN = args.KernelGAN),
                      batch_size = 1,
                      epochs = args.num_epochs,
                      verbose = 1,
                      callbacks = callback_list,
                      steps_per_epoch = 1)
            super_image = model_class.predict_image(image = image, scale_factor = scale_factor, model = model)
            image = super_image
        plt.imsave(os.path.join(super_dir,'%s' %file), super_image, format = 'png')
        tf.keras.backend.clear_session()
    stop = timeit.default_timer()
    print('Done!!!')
    print('Time take for all images is', stop-start, 'seconds')


if __name__ == '__main__':
    main()