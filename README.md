# Self-supervised-super-resolution-of-ultrasound-images

This repository contains proposes deep learning models for single image super resolution (SISR) of ultrasound (US) images. The proposed models, PM1 & PM2, are self-supervised and aim to perform blind SR. Self-supervision is a class of methods under unsupervised learning where any necessary supervision to the model is provided from input data only. Self-supervision allows the model to predict previously unseen images while avoiding the need for an explicit training phase. Blind SR in images refers to the task of performing super resolution on input images without any prior knowledge or assumptionabout the source of degradation (if present) and methods utilised in obtaining the input images. The proposed modelscombine 2D wavelet packet decomposition/transformation (WPD) with convolutional neural networks. WPD dividesthe input into different sub-bands for analysis.  These sub-bands are synthesized through IWPD (Inverse WPD) to reconstruct the signal. Based on extensive experimentation, the models give a fair performance on the different test cases. In ideal cases, where the source of the image is known, the benchmarking models used in this project perform better. One of theproposed models, which uses the simplest architecture, gives good performance in blind SR settings and is comparablein ideal cases.  The second proposed model, which performs super resolution in two parts, gives a slightly lower performance than Model 1 according to evaluation metrics. However, visually, the predicted images by both models are very close.

*****Running the code*********

To run model without KernelGAN estimation, i.e., PM1:
```
python main.py --input_path <include directory of input images> --output_path <include directory where you wish to see the predicted images>
```

To run model without KernelGAN estimation, i.e., PM2:
```
python main.py --KernelGAN True --input_path <include directory of input images> --output_path <include directory where you wish to see the predicted images> 
```
## Optional Arguments
```
-h --help			show this help message and exit

-- drop_lr			Factor that reduces the learning rate as new_lr = lr * factor

--num_epochs			Number of epochs to run the model per image

--gradual_increase_value	Value with which the images are gradually super-resolved. This gradual increase factor is inspired by Shocher, Assaf & Cohen, Nadav & Irani, Michal. (2018). Zero-Shot Super-Resolution Using Deep Internal Learning. 3118-3126. 10.1109/CVPR.2018.00329.

--sigma				Standard deviation (spread or “width”) of the normal distribution, used in introducing random noise

--leave_as_is_probability	This is the probability associated with augmentation of hr parent. A higher leave_as_is_probability reduces probability of random augmentation in hr parent.

--shear_scale_prob		This the prabability associated with random shearing & scaling of HR parent during augmentations. A lower shear_scale_prob value prompts the model to increase the probability of random shearing & scaling, and vice-versa.

--crop_size			This is the initial crop size to be considered.

--SR_factor			This is te super-resolution factor.

--center_crop_prob		If center_crop_prob is small, more crops are taken from the center of the image. Else, if center_crop_prob is large, crops are taken randomly from the image, regardless of location.

--KernelGAN			If true, KernelGAN first finds a image specific downsampling kernel and then performs super-resolution. If False, then cubic interpolation is used for obtaining lr_child from the hr_parent

--downsample_prob		A lower downsample_prob creates higher probability taking cubic interpolation when KernelGAN is False. On the other hand, when this value is high, the lr_child is directly equated to hr_parent without interpolation.

```
