# Self-supervised-super-resolution-of-ultrasound-images

This repository contains proposes deep learning models for single image super resolution (SISR) of ultrasound (US) images. The proposed models, PM1 & PM2, are self-supervised and aim to perform blind SR. Self-supervision is a class of methods under unsupervised learning where any necessary supervision to the model is provided from input data only. Self-supervision allows the model to predict previously unseen images while avoiding the need for an explicit training phase. Blind SR in images refers to the task of performing super resolution on input images without any prior knowledge or assumptionabout the source of degradation (if present) and methods utilised in obtaining the input images. The proposed modelscombine 2D wavelet packet decomposition/transformation (WPD) with convolutional neural networks. WPD dividesthe input into different sub-bands for analysis.  These sub-bands are synthesized through IWPD (Inverse WPD) to reconstruct the signal. Based on extensive experimentation, the models give a fair performance on the different test cases. In ideal cases, where the source of the image is known, the benchmarking models used in this project perform better. One of theproposed models, which uses the simplest architecture, gives good performance in blind SR settings and is comparablein ideal cases.  The second proposed model, which performs super resolution in two parts, gives a slightly lower performance than Model 1 according to evaluation metrics. However, visually, the predicted images by both models are very close.

*****STEPS TO RUN THE CODE*********

All of the required code blocks are included in the notebook and are generally sorted in a way to be run 
sequentially. This readme explains the flow of the code in the notebook:

Please Note: The path provided fo training and testing sets in the respective notebooks should be replaced as per the location in the local Google Drives.


	* Import necessary libraries [Cell 1]
	* Upload the dataset on which to perform super-resolution. The models were designed to be originally run on medical images, however, good results are acheieved on natural images as well.
  	* Change values of hyper-parameters under Section Run.
