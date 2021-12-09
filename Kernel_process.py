# -*- coding: utf-8 -*-
import numpy as np
import os
import scipy.io as sio
from scipy.ndimage import filters, measurements, interpolation
"""
The post-processing here has been used as in the official code of KernelGAN, URL: https://github.com/sefibk/KernelGAN

Paper: S. B. Kligler, A. Shocher, M. Irani, "Blind Super-Resolution Kernel Estimation using an Internal-GAN" in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, 8-14 December 2019, Vancouver, BC, Canada. pages 284-293, 2019.
"""

def numeric_kernel(hr_parent, kernel, scale_down, output_shape):
  # See kernel_shift function to understand what this is
  # First run a correlation (convolution with flipped kernel)
  out_im = np.zeros_like(hr_parent)
  for channel in range(hr_parent.ndim):
    out_im[:,:,channel] = filters.correlate(hr_parent[:,:,channel], kernel)
  # Then subsample and return
  return out_im[np.round(np.linspace(0, hr_parent.shape[0] - 1 / scale_down, output_shape[0])).astype(int)[:, None],
           np.round(np.linspace(0, hr_parent.shape[1] - 1 / scale_down, output_shape[1])).astype(int), :]

def zeroize_negligible(k,n = 40):
  """Zeroize values that are negligible w.r.t to values in k"""
  # Sort K's values in order to find the n-th largest
  k_sorted = np.sort(k.flatten())
  # Define the minimum value as the 0.75 * the n-th largest value
  k_n_min = 0.75 * k_sorted[-n - 1]
  # Clip values lower than the minimum value
  filtered_k = np.clip(k - k_n_min, a_min=0, a_max=100)
  # Normalize to sum to 1
  return filtered_k / filtered_k.sum()

def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel :
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between odd and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second term ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) // 2 + 0.5 * (np.array(sf) - (np.array(kernel.shape) % 2))
    
    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass
    
    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    kernel = np.pad(kernel, np.int(np.ceil(np.max(np.abs(shift_vec)))) + 1, 'constant')
    
    # Finally shift the kernel and return
    kernel = interpolation.shift(kernel, shift_vec)
    return kernel

def post_process_k(k, n):
    """Move the kernel to the CPU, eliminate negligible values, and centralize k"""
    k = k.detach().cpu().float().numpy()
    # Zeroize negligible values
    significant_k = zeroize_negligible(k, n)
    # Force centralization on the kernel
    centralized_k = kernel_shift(significant_k, sf=2)
    # return shave_a2b(centralized_k, k)
    return centralized_k

def analytic_kernel(k):
    """Calculate the X4 kernel from the X2 kernel (for proof see appendix in paper)"""
    k_size = k.shape[0]
    # Calculate the big kernels size
    big_k = np.zeros((3 * k_size - 2, 3 * k_size - 2))
    # Loop over the small kernel to fill the big one
    for r in range(k_size):
        for c in range(k_size):
            big_k[2 * r:2 * r + k_size, 2 * c:2 * c + k_size] += k[r, c] * k
    # Crop the edges of the big kernel to ignore very small values and increase run time of SR
    crop = k_size // 2
    cropped_big_k = big_k[crop:-crop, crop:-crop]
    # Normalize to 1
    return cropped_big_k / cropped_big_k.sum()

def save_final_kernel(k_2, img_name, i, kernel_pd):
    """saves the final kernel and the analytic kernel to the results folder"""
    if i == 0:
      sio.savemat(os.path.join(kernel_pd, '%s_kernel_x2.mat' % img_name), {'Kernel': k_2})
    else:
      k_4 = analytic_kernel(k_2)
      sio.savemat(os.path.join(kernel_pd, '%s_kernel_x4.mat' % img_name), {'Kernel': k_4})