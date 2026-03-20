from email.mime import image

from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
from skimage.filters import threshold_otsu



#Exercise 1: Start by reading the image and inspect the histogram. Is it a bimodal histogram? Do you think it will be possible to segment it so only the bones are visible?
in_dir = "data/"

# X-ray image
im_name = "vertebra.png"

original_image = io.imread(in_dir + im_name)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
ax[0].imshow(original_image, cmap='gray')
ax[0].set_title('Original image')

ax[1].hist(original_image.ravel(), bins=256)
ax[1].set_title('Histogram of the original image')
ax[1].set_xlabel('Pixel intensity')
ax[1].set_ylabel('Frequency')   
plt.show()

# It is kind of a bimodal histogram, so it should be 
# possible to segment the image so only the bones are visible.
# However, the vertabrae themself would be hard to separate from 
# other bone structures. 

# Exercise 2: Compute the minimum and maximum values of the image. Is the full scale of the gray-scale spectrum used or can we enhance the appearance of the image?

min_val = np.min(original_image)
max_val = np.max(original_image)
print(f"Minimum pixel intensity: {min_val}")
print(f"Maximum pixel intensity: {max_val}")

# The full scale is not used becase the minimum pixel intensity is 57 and 
# the max is 235 so we can enhance the appearance of the image by rescaling the pixel intensities to use the full range of 0 to 255. 

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
ax[0].imshow(original_image, vmin = min_val, vmax = max_val, cmap='gray')
ax[0].imshow(original_image, vmin = 0, vmax = 255, cmap = 'gray')
ax[0].set_title('Visualization window: [0, 255]')
ax[1].imshow(original_image, vmin = min_val, vmax = max_val, cmap = 'gray') # Here, we change the visualization window, not the image itself!
ax[1].set_title(f'Visualization window: [{min_val}, {max_val}]')
plt.show()

#Exercise 3 
#Read the image vertebra.png and compute and show the minumum and maximum values.
#Use img_as_float to compute a new float version of your input image. Compute the
#minimum and maximum values of this float image. Can you verify that the float image
# is equal to the original image, where each pixel value is divided by 255?

float_image = img_as_float(original_image)
min_float = np.min(float_image)
max_float = np.max(float_image)
print(f"Minimum pixel intensity (float): {min_float}")
print(f"Maximum pixel intensity (float): {max_float}")

# Verification 
verify = np.allclose(float_image, original_image / 255)
print(f"Verification result: {verify}")

# Exercise 4: Use img_as_ubyte on the float image you
#  computed in the previous exercise. Compute the Compute
#  the minimum and maximum values of this image. Are they as expected?

ubyte_image = img_as_ubyte(float_image)
min_ubyte = np.min(ubyte_image)
max_ubyte = np.max(ubyte_image)
print(f"Minimum pixel intensity (uint8): {min_ubyte}")
print(f"Maximum pixel intensity (uint8): {max_ubyte}")

if min_ubyte == min_val and max_ubyte == max_val:
    print("The minimum and maximum values are as expected.")

# Exercise 5: *Implement a Python function called histogram_stretch.

def histogram_stretch(image):
    """
    Stretches the histogram of an image 
    :param img_in: Input image
    :return: Image, where the histogram is stretched so the min values is 0 and the maximum value 255
    """
    img_float = img_as_float(image)
    v_min = np.min(img_float)
    v_max = np.max(img_float)

    v_min_d = 0.0
    v_max_d = 1.0

    streched = ((v_max_d - v_min_d) / (v_max - v_min)) * (img_float - v_min) + v_min_d
    streched_ubyte = img_as_ubyte(streched)
    return streched_ubyte

# Exercise 6: Test your histogram_stretch on
#  the vertebra.png image. Show the image
#  before and after the histogram stretching. 
# What changes do you notice in the image?
#  Are the important structures more visible?

stretched_image = histogram_stretch(original_image)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
ax[0].imshow(original_image, cmap='gray', vmin = 0, vmax = 255)
ax[0].set_title('Original image')
ax[1].imshow(stretched_image, cmap='gray')
ax[1].set_title('Stretched image')
plt.show()

# The histogram stretching has enhanced the contrast of the image, making the important structures more visible.
# The bones in the vertebrae are now more distinguishable from the surrounding tissues, which can be particularly
# helpful for medical analysis and diagnosis. However, in this case the improvement seems marginal. 

# Exercise 7: Implement a function, gamma_map(img, gamma), that:
# Converts the input image to float
# Do the gamma mapping on the pixel values
# Returns the resulting image as an unsigned byte image.

def gamma_map(img, gamma):
    img_float = img_as_float(img)
    gamma_mapped = np.power(img_float, gamma)
    gamma_mapped_ubyte = img_as_ubyte(gamma_mapped)
    return gamma_mapped_ubyte

#Exercise 8: Test your gamma_map function on the vertebra image or another image of your choice. Try different values of γ
# for example 0.5 and 2.0. Show the resuling image together with the input image. Can you see the differences in the images?

gamma_05_image = gamma_map(original_image, 0.5)
gamma_20_image = gamma_map(original_image, 2.0)
gamma_50_image = gamma_map(original_image, 5.0)
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(18, 5))
ax[0].imshow(original_image, cmap='gray', vmin = 0, vmax = 255)
ax[0].set_title('Original image')   
ax[1].imshow(gamma_05_image, cmap='gray')
ax[1].set_title('Gamma = 0.5')
ax[2].imshow(gamma_20_image, cmap='gray')
ax[2].set_title('Gamma = 2.0')
ax[3].imshow(gamma_50_image, cmap='gray')
ax[3].set_title('Gamma = 5.0')
plt.show()

# Exercise 9: Implement a function, threshold_image :

def threshold_image(img, threshold):
    """
    Apply a threshold in an image and return the resulting image
    :param img_in: Input image
    :param thres: The treshold value in the range [0, 255]
    :return: Resulting image (unsigned byte) where background is 0 and foreground is 255
    """
    mask = img > threshold
    return img_as_ubyte(mask)

# Exercise 10: Test your threshold_image function on the vertebra
#  image with different thresholds. It is probably not possible to
#  find a threshold that seperates the bones from the background,
#  but can you find a threshold that seperates the human from the
#  background?

threshold_05_image = threshold_image(original_image, 100)
threshold_10_image = threshold_image(original_image, 150)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
ax[0].imshow(original_image, cmap='gray', vmin = 0, vmax = 255)
ax[0].set_title('Original image')
ax[1].imshow(threshold_05_image, cmap='gray')
ax[1].set_title('Threshold = 100')
ax[2].imshow(threshold_10_image, cmap='gray')
ax[2].set_title('Threshold = 150')
plt.show()

# Exercise 11: Read the documentation of Otsu's
#  method and use it to compute and apply a threshold 
# to the vertebra image.
# How does the threshold and the result compare to your
# manually found threshold?

otsu_threshold = threshold_otsu(original_image)
otsu_threshold_image = threshold_image(original_image, otsu_threshold)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
ax[0].imshow(original_image, cmap='gray', vmin = 0, vmax = 255)
ax[0].set_title('Original image')
ax[1].imshow(threshold_10_image, cmap='gray')
ax[1].set_title('Threshold = 150')
ax[2].imshow(otsu_threshold_image, cmap='gray')
ax[2].set_title(f"Otsu's threshold = {otsu_threshold}")
plt.show()





