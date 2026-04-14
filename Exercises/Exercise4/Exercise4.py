from xml.etree.ElementPath import find

from scipy.ndimage import correlate
import numpy as np
from skimage.filters import median
from skimage import color
from skimage import io
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.filters import prewitt_h
from skimage.filters import prewitt_v
from skimage.filters import prewitt
from skimage.filters import threshold_otsu



input_img =  np.arange(25).reshape(5, 5)
print (input_img)

weights = [[0, 1, 0],
           [1, 2, 1],
           [0, 1, 0]]

res_img =  correlate(input_img, weights)

# Exercise 1 Print the value in position (3, 3) 
# in res_img. Explain the value?
print (res_img)

print(f"Value in position (3, 3) of res_img: {res_img[3, 3]}")
# This is calculated by applying the kernel centered at row 3, column 3.
# The value sums the center pixel (18) weighted by 2, plus its four orthogonal 
# neighbors (13, 17, 19, 23) weighted by 1 each.

res_im = correlate(input_img, weights, mode='constant', cval=10)

#Exercise 2 Compare the output images when using 
# reflect and constant for the border. Where and 
# why do you see the differences.
print("Result with constant border:")
print(res_im)
res_im_reflect = correlate(input_img, weights, mode='reflect')
print("Result with reflect border:")
print(res_im_reflect)
# The constant border mode pads the input image with a constant value 
# (10 in this case) around the borders, which affects the convolution 
# results near the edges. The reflect mode, on the other hand, pads the input image by reflecting 
# it across the borders, which preserves more of the original image's 
# structure and results in different values near the edges compared to the constant mode.


#Exercise 3 Read and show the image Gaussian.png ç
# from the exercise material. Convert the image to 
# grayscale. Although it already appears to be grayscale, 
# black-and-white images are sometimes stored as 3-channel 
# RGB with identical values in each channel.

img = io.imread('data/Gaussian.png')
print(f"Original image shape: {img.shape}")
img_gray = color.rgb2gray(img)
print(f"Grayscale image shape: {img_gray.shape}")

plt.figure(figsize=(6, 6))
plt.imshow(img_gray, cmap='gray')
plt.title('Grayscale Image')
plt.show()


def apply_mean_filter(img, size):
    weights = np.ones([size, size])
    weights = weights / np.sum(weights)

    out_img = correlate(img, weights, mode='reflect')
    return out_img


#Use correlate with the Gaussian.png 
# image and the mean filter. Show the 
# resulting image together with the input image. What do you observe?
res_img_gaussian_20 = apply_mean_filter(img_gray, 20)
res_img_gaussian_10 = apply_mean_filter(img_gray, 10)
res_img_gaussian_40 = apply_mean_filter(img_gray, 40)

fig, axes = plt.subplots(1, 4, figsize=(10, 5))
axes[0].imshow(img_gray, cmap='gray')
axes[0].set_title('Original Grayscale')
axes[1].imshow(res_img_gaussian_10, cmap='gray')
axes[1].set_title('Mean Filter Size 10')
axes[2].imshow(res_img_gaussian_20, cmap='gray')
axes[2].set_title('Mean Filter Size 20')
axes[3].imshow(res_img_gaussian_40, cmap='gray')
axes[3].set_title('Mean Filter Size 40')
plt.show()

# Try to change the size of the filter to 10, 20, 40 etc.. What do you see?
# What happens to the noise and what happens to the places 
# in image where there are transitions from light to dark areas?

# As the size of the mean filter increases, the resulting image becomes more blurred.
# The noise in the image is reduced as the filter size increases, 
# but the edges and transitions between light and dark areas become 
# less sharp and more smoothed out. This is because a larger mean filter averages over a larger neighborhood of pixels, which can help to reduce 
# noise but also leads to a loss of detail in the image.

#Exercise 4 Filter the Gaussian.png image with the 
# median filter with different size (5, 10, 20...). 
# What do you observe? What happens with the noise and with the 
# lighth-dark transitions?

def apply_median_filter(img, size):
    footprint = np.ones([size, size])
    med_img = median(img,footprint)
    return med_img

res_img_median_5 = apply_median_filter(img_gray, 5)
res_img_median_10 = apply_median_filter(img_gray, 10)
res_img_median_20 = apply_median_filter(img_gray, 20)

fig, axes = plt.subplots(1, 4, figsize=(10, 5))
axes[0].imshow(img_gray, cmap='gray')
axes[0].set_title('Original Grayscale')
axes[1].imshow(res_img_median_5, cmap='gray')
axes[1].set_title('Median Filter Size 5')
axes[2].imshow(res_img_median_10, cmap='gray')
axes[2].set_title('Median Filter Size 10')
axes[3].imshow(res_img_median_20, cmap='gray')
axes[3].set_title('Median Filter Size 20')
plt.show()

# As the size of the median filter increases, the resulting 
# image becomes smoother, but it preserves edges better than 
# the mean filter. The noise in the image is reduced as the filter 
# size increases, but the transitions between light and dark areas 
# remain sharper compared to the mean filter. This is because the 
# median filter replaces each pixel value with the median of its neighborhood, 
# which is effective at removing noise while preserving edges.

# Exercise 5 Try to use your mean and median filter with 
# different filter sizes on the SaltPepper.png. What do you observe? 
# Can they remove the noise and what happens to the image?

salt_pepper_img = io.imread('data/SaltPepper.png')
salt_pepper_gray = color.rgb2gray(salt_pepper_img)
res_salt_pepper_mean_5 = apply_mean_filter(salt_pepper_gray, 5)
res_salt_pepper_median_5 = apply_median_filter(salt_pepper_gray, 5)
res_salt_pepper_mean_10 = apply_mean_filter(salt_pepper_gray, 10)
res_salt_pepper_median_10 = apply_median_filter(salt_pepper_gray, 10)

fig, axes = plt.subplots(1, 5, figsize=(15, 5))
axes[0].imshow(salt_pepper_gray, cmap='gray')
axes[0].set_title('Original Grayscale')
axes[1].imshow(res_salt_pepper_mean_5, cmap='gray')
axes[1].set_title('Mean Filter Size 5')
axes[2].imshow(res_salt_pepper_median_5, cmap='gray')
axes[2].set_title('Median Filter Size 5')
axes[3].imshow(res_salt_pepper_mean_10, cmap='gray')
axes[3].set_title('Mean Filter Size 10')
axes[4].imshow(res_salt_pepper_median_10, cmap='gray')
axes[4].set_title('Median Filter Size 10')
plt.show()

# Exercise 6 Let us try the Gaussian filter on the Gaussian.png image. 

gauss_img = io.imread('data/Gaussian.png')

gauss_img_1 = gaussian(gauss_img, sigma=1)
gauss_img_5 = gaussian(gauss_img, sigma=5)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(gauss_img, cmap='gray')
axes[0].set_title('Original Grayscale')
axes[1].imshow(gauss_img_1, cmap='gray')
axes[1].set_title('Gaussian Filter Sigma 1')
axes[2].imshow(gauss_img_5, cmap='gray')
axes[2].set_title('Gaussian Filter Sigma 5')
plt.show()

# Exercise 7 Use one of your images (or use the car.png image) to try 
# the above filters. Especially, try with large filter kernels 
# (larger than 10) with the median and the Gaussian filter. 
# Remember to transform your image into gray-scale before filtering.
# What is the visual difference between in the output? Try to 
# observe places where there is clear light-dark transition.

car_img = io.imread('data/car.png')
car_gray = color.rgb2gray(car_img)
car_gauss_20 = gaussian(car_gray, sigma=20)
car_median_20 = apply_median_filter(car_gray, 20)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(car_gray, cmap='gray')
axes[0].set_title('Original Grayscale')
axes[1].imshow(car_gauss_20, cmap='gray')
axes[1].set_title('Gaussian Filter Sigma 20')
axes[2].imshow(car_median_20, cmap='gray')
axes[2].set_title('Median Filter Size 20')
plt.show()

# Exercise 8 Try to filter the donald_1.png photo with the 
# prewitt_h and prewitt_v filters and show the output without 
# converting the output to unsigned byte. Notice that the output 
# range is [-1, 1]. Try to explain what features of the image that 
# gets high and low values when using the two filters?

donald_img = io.imread('data/donald_1.png')
donald_gray = color.rgb2gray(donald_img)

prewitt_h_img = prewitt_h(donald_gray)
prewitt_v_img = prewitt_v(donald_gray)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(donald_gray, cmap='gray')
axes[0].set_title('Original Grayscale')
axes[1].imshow(prewitt_h_img, cmap='gray')
axes[1].set_title('Prewitt Horizontal')
axes[2].imshow(prewitt_v_img, cmap='gray')
axes[2].set_title('Prewitt Vertical')
plt.show()

# We can see that the prewitt_h filter highlights horizontal edges in the image,
# while the prewitt_v filter highlights vertical edges. The values close to 1 indicate
# strong edges in the respective direction, while values close to -1 indicate edges in the opposite direction.

# Exercise 9 Use the prewitt filter on donald_1.png. What do you see?

prewitt_img = prewitt(donald_gray)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(donald_gray, cmap='gray')
axes[0].set_title('Original Grayscale')
axes[1].imshow(prewitt_img, cmap='gray')
axes[1].set_title('Prewitt Filter')
plt.show()

# With the prewitt filter, we can see that it highlights edges in both horizontal and vertical directions, 
# resulting in a more comprehensive edge detection compared to using prewitt_h or prewitt_v alone. 
# The output emphasizes the contours and features of the image, making it easier to identify shapes and structures.

# Exercise 10¶
# The goal of this exercise is to detect the edges that seperates the bone from the
# soft tissue and the edges that separates the elbow from the background. Your detection algorithm should follow this outline:
# 
# Read the CT image
# Filter the image using either a Gaussian filter or a median filter
# Compute the gradients in the filtered image using a Prewitt filter
# Use Otsu's thresholding method to compute a threshold, T, in the gradient image
# Apply the threshold, T, to the gradient image to create a binary image.
# The final binary should contain the edges we are looking for. It will probably contain noise as well. We will explore methods to remove this noise later in the course.
# 
# You should experiment and find out:
# Does the median or Gaussian filter give the best result?
# Should you use both the median and the Gaussian filter?
# What filter size gives the best result?
# What sigma in the Gaussian filter gives the best result?

def detect_edges_in_ct_image(image_path, filter_type='gaussian', filter_size=5, gaussian_sigma=1, plot_result=False):
    # Read the CT image
    ct_img = io.imread(image_path)
    ct_gray = color.rgb2gray(ct_img)

    # Filter the image
    if filter_type == 'gaussian':
        filtered_img = gaussian(ct_gray, sigma=gaussian_sigma)
    elif filter_type == 'median':
        filtered_img = apply_median_filter(ct_gray, filter_size)
    else:
        raise ValueError("Invalid filter type. Use 'gaussian' or 'median'.")

    # Compute the gradients using Prewitt filter
    prewitt_img = prewitt(filtered_img)

    # Compute Otsu's threshold
    T = threshold_otsu(prewitt_img)

    # Apply the threshold to create a binary image
    binary_edges = prewitt_img > T

    if plot_result:
        fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (10,5))
        ax[0].imshow(filtered_img, cmap = 'gray')
        ax[1].imshow(prewitt_img, cmap = 'gray')
        ax[2].imshow(binary_edges, cmap = 'gray')
        [ax_.set_axis_off() for ax_ in ax]
        plt.show()

    return binary_edges

# Example usage:
edges = detect_edges_in_ct_image('data/ElbowCTSlice.png', filter_type='gaussian', gaussian_sigma=2, plot_result=True)
edges = detect_edges_in_ct_image('data/ElbowCTSlice.png', filter_type='median', filter_size=5, plot_result=True)

# The best tresult seems to be given by the median filter with a filter size of 5. The Gaussian filter with a sigma of 2 also gives a good result, but it may introduce more blurring compared to the median filter. 
# The choice between the two filters depends on the specific characteristics of the image and the desired outcome, but in this case, the median filter appears to be more effective at preserving edges while reducing noise.
