# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 19:14:59 2021

@author: Yukta
"""
import cv2
import numpy as np
import random
import tensorflow as tf
import pathlib
from Kernel_process import *
from tensorflow.keras.models import Model
from keras.applications.vgg19 import VGG19

mae_loss_object = tf.keras.losses.MeanAbsoluteError()
vgg19 = VGG19(include_top=False, weights='imagenet')
vgg19.trainable = False
for l in vgg19.layers:
    l.trainable = False
vgg_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block1_conv2').output)
vgg_model.trainable = False

def load_img(img_path, return_data_type = 'float32'):
    """
    Takes input image and returns a return_data_type array
    """
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)                # Read image                                                                                                                                      
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                # BGR -> RGB
    else:
        image = np.stack((image,) * 3, axis=-1)                       # Grayscale, channel 1 -> channel 3
    if image.shape[0] == 1 or image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)

    image = image.astype(return_data_type)       
    return image

def add_noise(image, sigma):
    """
    Adding noise normal noise to the image.
    """
    shape = image.shape
    noise = np.random.normal(0, sigma, (shape))                      # random.normal(loc(mean)=0.0, scale(std)=1.0,
                                                                     # size(output shape)=None)
    noise = noise.astype('float32')                                  # Check image dtype before adding
    noisy = np.clip((image + noise), 0, 255)                         # We clip negative values and set them to zero 
                                                                     # and values over 255 are clipped to 255.
    return noisy


def image_crop(img_list, crop_size, leave_as_prob, shear_scale_prob, center_crop_prob):
  """
  Creating random crops from the images in a list
  """
  num_imgs = len(img_list)
  img_h, img_w, _ = img_list[0].shape
  for i in range(num_imgs):
    img_list[i] = np.array(img_list[i], dtype=np.float32)
  
  # Choosing random samples from a uniform distribution over [0, 1)
  random_chooser = np.random.rand()
  random_augment = np.random.rand()

  if random_chooser < leave_as_prob:
    # In leave_as_is mode, no augmentation are made on the input image
    mode = 'leave_as_is'
  else:
    # In random_augment mode, random augmentations such as shearing, scaling, random crops are performed
    mode = 'random_augment'

  if mode == 'leave_as_is':
    for i in range(num_imgs):
      img_list[i] = img_list[i]
  else:

    # Shearing and scaling on input image
    if random_augment > shear_scale_prob:
      shear_x = np.random.randn()*0.25
      shear_y = np.random.randn()*0.25
      scale_x = np.random.randn()*0.15
      scale_y = np.random.randn()*0.15
      transform_matrix = np.array([[1+scale_x, shear_x, 0.0],[shear_y, 1+scale_y, 0.0]])
      transform_matrix = transform_matrix.astype(img_list[0].dtype)
      for i in range(num_imgs):
        img_list[i] = cv2.warpAffine(img_list[i], transform_matrix, (img_w,img_h))      
    else:
      # Taking crops of crop_size from the center of the image
      if random_chooser > center_crop_prob:
        if img_h > crop_size:
          start_h = int((img_h - crop_size)/2)
          end_h = int(start_h + crop_size)
          for i in range(num_imgs):
            img_list[i] = img_list[i][start_h:end_h, :, :]
        if img_w > crop_size:
          start_w = int((img_w-crop_size)/2)
          end_w = int(start_w + crop_size)
          for i in range(num_imgs):
            img_list[i] = img_list[i][:, start_w: end_w, :]
      else:
          # Taking crops of crop_size randomly from the image
          while (img_h - 1 < crop_size) or (img_w - 1 < crop_size):
            crop_size -=4

          w_crop_diff = img_w - crop_size
          h_crop_diff = img_h - crop_size

          top_left_x_coordinate = np.random.randint(0, w_crop_diff)
          top_left_y_coordinate = np.random.randint(0, h_crop_diff)
          
          for i in range(num_imgs):
            img_list[i] = img_list[i][top_left_y_coordinate:top_left_y_coordinate+crop_size, top_left_x_coordinate:top_left_x_coordinate+crop_size,:]

    # Random 45 degree rotations
    random_rot = random.randint(0,7)
    for i in range(num_imgs):
      img_list[i] = np.rot90(img_list[i], random_rot, axes = (0,1))
    if random_rot > 3 :
      for i in range(num_imgs):
        img_list[i] = np.fliplr(img_list[i])

  for i in range(num_imgs):
    if img_list[i].shape[0]%2 != 0:
      img_list[i] = img_list[i][:-1,:,:]
    if img_list[i].shape[1]%2 !=0:
      img_list[i] = img_list[i][:,:-1,:]

  return img_list

def swap_axis(image):

  return np.transpose(image, axes=[3,1,2,0]) if type(image) == np.ndarray else tf.transpose(image, perm=[3,1,2,0])
  
def parent_to_child(hr_parent, scale_factor, kernel, sigma, downsample_prob, KernelGAN):
  """
  This function takes the hr_parent and first downsamples it to create lr_child and 
  then adds noise if noise_flag is True.
  Finally, the image is upsampled to feed into the network.
  """
  scale_down = 1/scale_factor
  random_chooser = np.random.rand()

  if len(hr_parent.shape) == 4:
    hr_parent = np.squeeze(hr_parent, axis=0)
  else:
    hr_parent = hr_parent
  input_shape = hr_parent.shape
  
  if KernelGAN:
    output_shape = np.uint(np.ceil(np.array(input_shape))*np.array(scale_down))
    lr_child = numeric_kernel(hr_parent, kernel, scale_down, output_shape)
    lr_child = add_noise(lr_child, sigma)
  else:
    if random_chooser < downsample_prob:
      lr_child = hr_parent
    else:
      lr_child = cv2.resize(hr_parent, None, fx = scale_down, fy = scale_down, interpolation = cv2.INTER_CUBIC)
  
  lr_child = cv2.resize(lr_child, (hr_parent.shape[1], hr_parent.shape[0]), interpolation = cv2.INTER_CUBIC)
 
  return np.expand_dims(lr_child, axis=0)

def hr_lr_generator(image, 
                    scale_factor, 
                    final_kernel,
                    shear_scale_prob,
                    leave_as_prob,
                    center_crop_prob,
                    crop_size,
                    sigma, 
                    downsample_prob, 
                    KernelGAN):

  """
  Generator to simply return hr_parent and lr_child as a pair
  """
  while True:
    
    hr_parent_list = image_crop([image],
                               crop_size,
                               leave_as_prob,
                               shear_scale_prob,
                               center_crop_prob)
    hr_parent = hr_parent_list[0]
    lr_child = parent_to_child(hr_parent, scale_factor, final_kernel, sigma, downsample_prob, KernelGAN)
  
    if len(lr_child.shape) == 4:
      x = lr_child
    else:
      x = np.expand_dims(lr_child, axis=0)

    if len(hr_parent.shape) == 4:
      y = hr_parent
    else:
      y = np.expand_dims(hr_parent, axis = 0)
    yield x, y

def get_gradual_factors(SR_factor, gradual_increase_value):
  """
  Depending on final SR_factor and gradual_increase_value, it determines the intermediate SR factors.
  For example, 
  Input: SR_factor = 8, gradual_increase_value = 2 
  Output: [2,4,8]
  """
  gradual_SR_list = [SR_factor]
  sr_fact = SR_factor/gradual_increase_value
  while (sr_fact) != 1:
    gradual_SR_list.append(int(sr_fact))
    sr_fact = sr_fact/gradual_increase_value
  gradual_SR_list.reverse()
  return gradual_SR_list

def get_images_paths(input_pd):
  """
  Get paths of images from input directory
  """
  root = pathlib.Path(input_pd)
  img_paths = list(sorted(root.rglob('*.png')))
  img_paths_list = [str(path) for path in img_paths]
  
  return img_paths_list
