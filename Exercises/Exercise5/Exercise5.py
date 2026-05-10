from skimage import io, color, morphology
from skimage.util import img_as_float, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.filters import threshold_otsu
from skimage import segmentation
from skimage import measure
from skimage.color import label2rgb
import skimage.io as io

def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(modified)
    ax2.set_title(modified_name)
    ax2.axis('off')
    io.show()

# Exercise 1: Binary image from original image¶
# Read the image, convert it to grayscale and use Otsus method to compute and apply a threshold.
# Show the binary image together with the original image.

lego_img = io.imread('data/lego_4_small.png')
lego_img_gray = color.rgb2gray(lego_img)
T = threshold_otsu(lego_img_gray)

# Exercise 2: Remove border BLOBs¶
# Use segmentation.clear_border to remove border pixels
#  from the binary image.

binary_lego_img = lego_img_gray < T
segmentation.clear_border(binary_lego_img)
show_comparison(lego_img, binary_lego_img, 'Binary image')

# Exercise 3: Cleaning using morphological operations¶
# In order to remove remove noise and close holes, you 
# should do a morphological closing followed by a morphological 
# opening with a disk shaped structuring element with radius 5.
selem = morphology.disk(5)
closed_lego_img = morphology.closing(binary_lego_img, selem)
opened_lego_img = morphology.opening(closed_lego_img, selem)
show_comparison(lego_img, opened_lego_img, 'Closed and opened image')


#Exercise 4: Find labels
# The actual connected component analysis / BLOB analysis is performed using measure.label :
labels = measure.label(opened_lego_img)
n__labels = labels.max()
print(f'Number of labels: {n__labels}')

# Exercise 5: Visualize found labels¶
# We can use the function label2rbg to create a visualization of the found BLOBS. Show this together with the original image.
labelled_img = label2rgb(labels, image=lego_img, bg_label=0)
show_comparison(lego_img, labelled_img, 'Labelled image')

# Exercise 6: Compute BLOB features¶
# It is possible to compute a wide variety of BLOB features using the measure.regionprops function
region_props = measure.regionprops(labels)
areas = np.array([prop.area for prop in region_props])
plt.hist(areas, bins=50)
plt.title('Histogram of BLOB areas')   
plt.xlabel('Area')
plt.ylabel('Frequency')
plt.show()

### Exercise 7: Interactive BLOB analysis¶


in_dir = "data/"
img_original =io.imread(in_dir + 'Sample E2 - U2OS DAPI channel.tiff')

img_small = img_original[700:1200, 900:1400]
img_gray = img_as_ubyte(img_small)
io.imshow(img_gray, vmin=0, vmax=150)
plt.title('DAPI Stained U2OS cell nuclei')
io.show()



