from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

# Directory containing data and images
in_dir = "data/"

# X-ray image
im_name = "metacarpals.png"

# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org = io.imread(in_dir + im_name)

print (im_org.shape)

print (im_org.dtype)

io.imshow(im_org)
plt.title('Metacarpals')
plt.show()

io.imshow(im_org, cmap='jet')
plt.title('Metacarpals (colormap)')
plt.show()

io.imshow(im_org, vmin = 20, vmax = 170)
plt.title('Metacarpals (vmin, vmax)')
plt.show()

def automatic_contrast(im):
    im_min = np.min(im)
    im_max = np.max(im)
    io.imshow(im, vmin = im_min, vmax = im_max)
    plt.title('Metacarpals (automatic contrast)')
    plt.show()

automatic_contrast(im_org)

plt.hist(im_org.ravel(), bins=256)
plt.title('Histogram of pixel intensities')
plt.show()

h = plt.hist(im_org.ravel(), bins=256)

bin_no = 100
count = h[0][bin_no]
print(f"There are {count} pixel values in bin {bin_no}")

bin_left = h[1][bin_no]
bin_right = h[1][bin_no + 1]
print(f"Bin {bin_no} covers pixel values from {bin_left} to {bin_right}")

y, x, _ = plt.hist(im_org.ravel(), bins=256)

## Most common range of intensities 
bin_no = np.argmax(y)
bin_left = x[bin_no]
bin_right = x[bin_no + 1]
print(f"The most common pixel values are in the range {bin_left} to {bin_right}")


r = 110
c = 90 

im_vl =im_org[r, c]
print(f"The pixel value at row {r} and column {c} is {im_vl}")


im_org[:30] = 0
io.imshow(im_org)
io.show()

## In the previous code cell, the first 30 rows of the image were set to zero. This means that all pixel values in those rows were changed to zero, effectively making that portion of the image completely black. 
##This is how we create masks in images.
## A mask is a binary image that indicates which pixels belong to a certain region of interest. In this case, by setting the first 30 rows to zero, we are creating a mask that highlights the area of the image that we want to focus on, while ignoring the rest of the image. This can be useful for various image processing tasks, such as segmentation or feature extraction.

mask = im_org > 150
io.imshow(mask)
plt.title('Mask of pixels with values greater than 150')
io.show()

## Exercise 12: Where are the values 1 and where are they 0?
## The values of 1 in the mask correspond to the pixels in the original image that have intensity values greater than 150. These pixels are highlighted in the mask, indicating that they belong to the region of interest. On the other hand, the values of 0 in the mask correspond to the pixels in the original image that have intensity values less than or equal to 150. These pixels are not highlighted in the mask, indicating that they do not belong to the region of interest.

im_org[mask] = 255
io.imshow(im_org)
io.show()

## We just set the pixels in the mask to 255 meaning white. 

##Color images

# Let's read a color image

im_color = io.imread(in_dir + "ardeche.jpg")
print(im_color.shape)
print(im_color.dtype)
io.imshow(im_color)
plt.title('Color image')
io.show()

r = 110
c = 90

pixel_value = im_color[r, c]
print(f"The pixel value at row {r} and column {c} is {pixel_value}")

# Exercise 16 coloring the upper half of the image green

im_color[:im_color.shape[0]//2, :] = [0, 255, 0]
io.imshow(im_color)
plt.title('Color image with upper half colored green')
io.show()

##Own imageg work 

my_image = io.imread(in_dir + "eye.jpg")

image_rescaled = rescale(my_image, 0.25, anti_aliasing=True, channel_axis=2)

io.imshow(image_rescaled)
plt.title('Rescaled image')
io.show()

# Inspecting pixel value range and data type of the rescaled image

print(f"Original image data type: {my_image.dtype}")
print(f"Rescaled image data type: {image_rescaled.dtype}")
print(f"Original pixel value range: {my_image.min()} to {my_image.max()}")
print(f"Rescaled pixel value range: {image_rescaled.min()} to {image_rescaled.max()}")

# Sample some specific pixel values
print(f"\nSample pixel values from rescaled image:")
print(f"Pixel at (10, 10): {image_rescaled[10, 10]}")
print(f"Pixel at (50, 50): {image_rescaled[50, 50]}")

# Check if values are in range [0, 255]
print(f"\nAre pixel values in range [0, 255]? NO, they are in range [0, 1]")
print(f"The rescale function converts to float64 with normalized values between 0 and 1")

# Exercise 19: Try to find a way to automatically scale your image so the resulting width (number of columns) is always equal to 400, no matter the size of the input image?

def resize_to_width(image, target_width=400):
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_height / original_width
    target_height = int(target_width * aspect_ratio)
    resized_image = resize(image, (target_height, target_width), anti_aliasing=True)
    return resized_image

resized_image = resize_to_width(my_image, target_width=400)
io.imshow(resized_image)
plt.title('Resized image with target width of 400')
io.show()

my_image_gray = color.rgb2gray(my_image) 

io.imshow(my_image_gray, cmap='gray')
plt.title('Grayscale image')

# Howing histogram of the grayscale image
plt.hist(my_image_gray.ravel(), bins=256)
plt.title('Histogram of grayscale image')

#Exercise 20: Take an image that is very dark and another very light image. Compute and visualise the histograms for the two images. Explain the difference between the two.

dark_image = io.imread(in_dir + "darkImage.jpg")
light_image = io.imread(in_dir + "lightImage.jpeg")

light_image_gray = color.rgb2gray(light_image)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(dark_image.ravel(), bins=256)
plt.title('Histogram of dark image')

plt.subplot(1, 2, 2)
plt.hist(light_image_gray.ravel(), bins=256)
plt.title('Histogram of light image')

io.show()

# Explanation:
# The histogram of the dark image shows most pixel values concentrated in the lower intensity range (left side of the histogram), indicating that most pixels are dark.
# The histogram of the light image shows most pixel values concentrated in the higher intensity range (right side of the histogram), indicating that most pixels are bright.
# This difference reflects the overall brightness levels in each image.

# Exercise  22

dtu_sign = io.imread(in_dir + "DTUsign1.jpg")

r_comp = dtu_sign[:, :, 0]
io.imshow(r_comp)
plt.title('DTU sign image (Red)')
io.show()

# Exercise 23.Exercise 23: Visualize the R, G, and B components individually. Why does the DTU Compute sign look bright on the R channel image and dark on the G and B channels? Why do the walls of the building look bright in all channels?

g_comp = dtu_sign[:, :, 1]
io.imshow(g_comp)
plt.title('DTU sign image (Green)')
io.show()

b_comp = dtu_sign[:, :, 2]
io.imshow(b_comp)
plt.title('DTU sign image (Blue)')
io.show()

# Explanation:
# The DTU Compute sign looks bright on the R channel image because it likely contains a significant amount of red color, which contributes to its brightness in the red channel. In contrast, it appears dark on the G and B channels because it has less green and blue color information, resulting in lower intensity values in those channels.


#Exercise 24: Start by reading and showing the DTUSign1.jpg image.

dtu_sign = io.imread(in_dir + "DTUsign1.jpg")
io.imshow(dtu_sign)
plt.title('DTU sign image')
io.show()   

dtu_sign[500:1000, 800:1500, :] = 0
io.imshow(dtu_sign)

#Exercise 25: Show the image again and save it to disk as DTUSign1-marked.jpg using the io.imsave function. Try to save the image using different image formats like for example PNG.

io.imsave(in_dir + "DTUSign1-marked.jpg", dtu_sign)
io.imsave(in_dir + "DTUSign1-marked.png", dtu_sign)

#Exercise 26: Try to create a blue rectangle around the DTU Compute sign and save the resulting image.
dtu_sign = io.imread(in_dir + "DTUsign1.jpg")  # Read the original image again to avoid modifying the previous one

dtu_sign[500:1000, 800:1500, :] = [0, 0, 255]  # Set the region to blue
io.imshow(dtu_sign)
plt.title('DTU sign image with blue rectangle')
io.show()

io.imsave(in_dir + "DTUSign1-marked-blue.jpg", dtu_sign)
io.imsave(in_dir + "DTUSign1-marked-blue.png", dtu_sign)

# Exercise 27: Try to automatically create an image based on metacarpals.png where the bones are colored blue. You should use color.gray2rgb and pixel masks.

metacarpals = io.imread(in_dir + "metacarpals.png")
metacarpals_rgb = color.gray2rgb(metacarpals)
bone_mask = metacarpals > 150  # Create a mask for the bones
metacarpals_rgb[bone_mask] = [0, 0, 255]  # Set the bone pixels to blue
io.imshow(metacarpals_rgb)
plt.title('Metacarpals with bones colored blue')
io.show()


p = profile_line(metacarpals, (342, 77), (320, 160))
plt.plot(p)
plt.title('Intensity profile along the line')
plt.xlabel('Pixel position along the line')
plt.ylabel('Pixel intensity')
plt.show()

in_dir = "data/"
im_name = "road.png"
im_org = io.imread(in_dir + im_name)
im_gray = color.rgb2gray(im_org)
l1 = 200
im_crop = im_gray[40:40 + l1, 150:150 + l1]
xx, yy = np.mgrid[0:im_crop.shape[0], 0:im_crop.shape[1]]
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
surf = ax.plot_surface(xx, yy, im_crop, rstride = 1, cstride = 1, cmap=plt.cm.jet, linewidth=0)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


in_dir = "data/"
im_name = "1-442.dcm"
ds = dicom.dcmread(in_dir + im_name)
print(ds)

#Exercise 29: What is the size (number of rows and columns) of the DICOM slice?

im = ds.pixel_array
print(f"The size of the DICOM slice is {im.shape[0]} rows and {im.shape[1]} columns.")

#Exercise 30: Try to find the shape of this image and the pixel type? Does the shape match the size of the image found by inspecting the image header information?

print(f"The shape of the DICOM image is {im.shape}.")
print(f"The pixel type of the DICOM image is {im.dtype}.")

io.imshow(im, vmin=-1000, vmax=1000, cmap='gray')
io.show()

