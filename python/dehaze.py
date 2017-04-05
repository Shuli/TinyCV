# -*- coding: utf-8 -*-
"""
========================================================================================================================
*** Dark Channel Prior Processsing for Haze Removal ***
------------------------------------------------------------------------------------------------------------------------
	This module performs fog(haze) removal in images based on the Dark Channel Prior.
	There are other "1) CLAHE", "2) Bilateral Filter" and "3) Tone Map combined" for fog removal processing.
	This process provides "Dark Channel Prior". This process also includes the Gaussian Filter.
	This process is generally very slow. Includes early version of this.
------------------------------------------------------------------------------------------------------------------------
Reference by the IEEE to http://mmlab.ie.cuhk.edu.hk/archive/2011/Haze.pdf
========================================================================================================================
Operating conditions necessary {UTF-8/CrLf/Python2.7/numpy/pillow}
@author: Hisashi Ikari
"""
import sys, logging, time, math, numpy as np
import cv2
from PIL import Image
from PIL import ImageOps

# ======================================================================================================================
# Static Variables
# ======================================================================================================================
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(funcName)20s %(message)s")

# ======================================================================================================================
# create_dark_channel
#    This process extracts the lowest brightness value in a range that can be a candidate for fog.
# ----------------------------------------------------------------------------------------------------------------------
# *** Define the Dark Channel ***
# 	J^{dark}(x) = min_{c \in {r, g, b}} (min_{y \in \omega(x) (J^{C}(y))})
#		-> C stands for color channel.
#		-> Omega represents a local patch.
#		-> x is X represents the position of the image.
# 		-> y represents the position in the local patch.
# ======================================================================================================================
def create_dark_channel(image, winsize):
	start_time = time.time()
	image_size = image.shape
	padding_size = int(math.floor(winsize/2))
	padding_image = np.ones((image_size[0] + padding_size + padding_size, image_size[1] + padding_size + padding_size, 3), dtype="float32")
	padding_image[padding_size:image_size[0]+padding_size,padding_size:image_size[1]+padding_size] = image
	dark_channel = np.zeros((image_size[0], image_size[1]), dtype="float32")
	logging.debug("image_size(0): %d, image_size(1): %d" % (image_size[0], image_size[1]))

	for j in range(image_size[0]):
		for i in range(image_size[1]):
			local_patch = padding_image[j:(j+(winsize+0)), i:(i+(winsize+0)), :]
			dark_channel[j,i] = np.min(local_patch)

	logging.debug("dark_channel:\n%s" % str(dark_channel))

	logging.info("elapsed_time: %s" % (time.time() - start_time))
	return dark_channel

# ======================================================================================================================
# create_atmosphere(A)
#     The top 1% of the fog candidates has high brightness. For that reason it may be a fog.
# ======================================================================================================================
def create_atmosphere(image, dark_channel):
	start_time = time.time()
	image_size = image.shape
	pixels_size = image_size[0] * image_size[1]
	pixels_search = int(math.floor(pixels_size * 0.01))
	logging.debug("image_size(0): %d, image_size(1): %d, pixels_size: %d, pixels_search: %d" % (image_size[0], image_size[1], pixels_size, pixels_search))

	vec_dark = np.array(dark_channel).flatten()
	vec_image = np.reshape(image, (pixels_size, 3))
	indices = np.argsort(vec_dark)[::-1]
	accumulator = np.zeros([1, 3]);

	for k in range(pixels_search):
		accumulator = accumulator + vec_image[indices[k], :]
	atmosphere = accumulator / pixels_search
	logging.debug("atmosphere: %s" % (str(atmosphere)))

	logging.info("elapsed_time: %s" % (time.time() - start_time))
	return atmosphere

# ======================================================================================================================
# estimate_transmission
# ======================================================================================================================
def estimate_transmission(image, atmosphere, omega, winsize):
	start_time = time.time()
	image_size = image.shape
	rep_atmosphere = np.tile(atmosphere, (image_size[0], image_size[1], 1))
	transmission = 1.0 - omega * create_dark_channel(image / rep_atmosphere, winsize)
	logging.debug("transmission: %s" % str(transmission))
	logging.info("elapsed_time: %s" % (time.time() - start_time))
	return transmission

# ======================================================================================================================
# adapt_guided_filter
# ----------------------------------------------------------------------------------------------------------------------
# *** Reference by the Guided Filter implementation from "Fast Guided Filter"***
#	http://arxiv.org/abs/1505.00996 / Algorithm 1
# ======================================================================================================================
def adapt_guided_filter(image, target, radius, eps):
	start_time = time.time()
	image_size = image.shape
	avg_denom = adapt_window_sum_filter(np.ones((image_size[0], image_size[1])), radius)
	mean_g = adapt_window_sum_filter(image, radius) / avg_denom
	mean_t = adapt_window_sum_filter(target, radius) / avg_denom
	corr_g = adapt_window_sum_filter(image * image, radius) / avg_denom
	corr_gt = adapt_window_sum_filter(image * target, radius) / avg_denom

	var_g = corr_g - (mean_g * mean_g)
	cov_gt = corr_gt - (mean_g * mean_t)
	a = cov_gt / (var_g + eps)
	b = mean_t - a * mean_g
	mean_a = adapt_window_sum_filter(a, radius) / avg_denom
	mean_b = adapt_window_sum_filter(b, radius) / avg_denom
	q = mean_a * image + mean_b

	logging.info("elapsed_time: %s" % (time.time() - start_time))
	return q

# ======================================================================================================================
# adapt_window_sum_filter
# ======================================================================================================================
def adapt_window_sum_filter(image, r):
	image_size = image.shape
	h = image_size[0]
	w = image_size[1]
	result = np.zeros((h, w))

	# Y-Axis
	cumsum_y = np.cumsum(image, axis=0)
	logging.debug("cumsum_y: %s" % str(cumsum_y))
	result[1-1:r+1,:] = cumsum_y[1-1+r:2*r+1,:]
	result[r+2-1:h-r,:] = cumsum_y[2*r+2-1:h,:] - cumsum_y[1-1:h-2*r-1,:]
	result[h-r+1-1:h,:] = np.tile(cumsum_y[h-1,:], (r, 1)) - cumsum_y[h-2*r-1:h-r-1,:]

	# X-Axis
	cumsum_x = np.cumsum(result, axis=1)
	logging.debug("cumsum_x: %s" % str(cumsum_x))
	result[:,1-1:r+1] = cumsum_x[:,1+r-1:2*r+1]
	result[:,r+2-1:w-r] = cumsum_x[:,2*r+2-1:w] - cumsum_x[:,1-1:w-2*r-1]
	result[:,w-r+1-1:w] = np.tile(cumsum_x[:,w-1], (r,1)).T - cumsum_x[:,w-2*r-1:w-r-1]

	return result

# ======================================================================================================================
# create_radiance
# ======================================================================================================================
def create_radiance(image, transmission, atmosphere):
	start_time = time.time()
	image_size = image.shape
	rep_atmosphere = np.tile(atmosphere, (image_size[0], image_size[1], 1))
	max_transmission = np.tile(np.maximum(transmission, 0.1), (3,1,1)).transpose((1,2,0))
	radiance = np.absolute(((image - rep_atmosphere) / max_transmission) + rep_atmosphere)

	logging.info("elapsed_time: %s" % (time.time() - start_time))
	return radiance

# ======================================================================================================================
# facade_dark_channel_prior
# ======================================================================================================================
def facade_dark_channel_prior(image, winsize=5, omega=0.95, r=15, res=0.001):
	start_time = time.time()
	image_size = image.shape
	channel = create_dark_channel(image, winsize)
	atmosphere = create_atmosphere(image, channel)
	trans_estimation = estimate_transmission(image, atmosphere, omega, winsize)
	x = adapt_guided_filter(be_grayscale(image), trans_estimation, r, res)
	save_image(x, "py_x.png")
	transmission = np.reshape(x, (image_size[0], image_size[1]))
	radiance = create_radiance(image, transmission, atmosphere)
	logging.info("elapsed_time: %s" % (time.time() - start_time))
	return radiance, x

# ======================================================================================================================
# Utility class for Image Processing
# ======================================================================================================================
def load_image(filename, shrinking=1.0):
	image = Image.open(filename)
	image.load()
	if shrinking < 1.0:
		image = image.resize((int(image.size[0] * shrinking)+1, int(image.size[1] * shrinking)+1))
	arrays = np.asarray(image,dtype="float32")/255.0
	logging.debug("image: %s" % str(arrays))
	return arrays

# ======================================================================================================================
# Utility class for Image Processing
# ======================================================================================================================
def save_image(arrays, filename):
	image = Image.fromarray(np.asarray(arrays*255.0, dtype="uint8"))
	image.save(filename)

# ======================================================================================================================
# Utility class for Image Processing
# ======================================================================================================================
def be_grayscale(image):
	result = Image.fromarray(np.asarray(image*255.0, dtype="uint8"))
	result = ImageOps.grayscale(result)
	return np.asarray(result,dtype="float32")/255.0

# ======================================================================================================================
# Entry Point
# ======================================================================================================================
if __name__ == "__main__":
	image = load_image("sample1.png", 1.0)
	logging.debug("image[0,0]: %s" % str(image[0,0,:]))
	save_image(image, "py_source_image.png")
	result, x = facade_dark_channel_prior(image)
	save_image((result), sys.argv[2])

