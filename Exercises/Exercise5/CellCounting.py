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

in_dir = "data/"
img_org = io.imread(in_dir + 'Sample E2 - U2OS DAPI channel.tiff')
# slice to extract smaller image
img_small = img_org[700:1200, 900:1400]
img_gray = img_as_ubyte(img_small) 
io.imshow(img_gray, vmin=0, vmax=100)
plt.title('DAPI Stained U2OS cell nuclei')
io.show()

# avoid bin with value 0 due to the very large number of background pixels
plt.hist(img_gray.ravel(), bins=256, range=(1, 100))
io.show()

# Exercise 8: Threshold selection
# Select an appropriate threshold, that seperates nuclei from the background. You can set it manually or use Otsus method.
# Show the binary image together with the original image and evaluate if you got the information you wanted in the binary image.
T = threshold_otsu(img_gray)
print(f'Threshold value: {T}')
binary_img = img_gray > T
show_comparison(img_small, binary_img, 'Binary image')


# It can be seen that there is some noise (non-nuclei) present and that some nuclei are connected. Nuclei that are overlapping very much should be discarded in the analysis. However, if they are only touching each other a little we can try to separate them. More on this later.

# To make the following analysis easier the objects that touches the border should be removed.

binary_img = morphology.remove_small_objects(binary_img, min_size=50)
img_c_b = segmentation.clear_border(binary_img)
label_img = measure.label(img_c_b)
image_label_overlay = label2rgb(label_img)
show_comparison(img_small, image_label_overlay, 'Found blobs')

#Exercise 10: BLOB features¶
# The task is now to find some object features that identify the cell nuclei and let us remove noise and connected nuclei. We use the function regionprops to compute a set of features for each object:

region_props = measure.regionprops(label_img)
print(region_props[0].area)
areas = np.array([prop.area for prop in region_props])

#We can try if the area of the objects is enough to remove invalid object. Plot a histogram of all the areas and see if it can be used to identify well separated nuclei from overlapping nuclei and noise. You should probably play around with the number of bins in your histogram plotting function.

plt.hist(areas, bins=100)
plt.show()

# Form the hsitogram we can see that most of the nuclea have an area between 60 and 110 , including some outliears up to the 300s.

min_area = 60
max_area = 110

label_img_filter = label_img
for region in region_props:
    if region.area < min_area or region.area > max_area:
        for coord in region.coords:
            label_img_filter[coord[0], coord[1]] = 0

i_area = label_img_filter > 0
show_comparison(img_small, i_area, 'Area filtered image')


#Exercise 12: Feature space

#Extract all the perimeters of the BLOBS:

perimeters = np.array([prop.perimeter for prop in region_props])

#Plot areas vs perimeters

plt.scatter(areas, perimeters)
plt.xlabel('Area')
plt.ylabel('Perimeter')
plt.show()

# Exercise 13: BLOB Circularity

#Compute the circularity for all objects and plot a histogram.
# 

circularity = 4 * math.pi * areas / (perimeters ** 2)
plt.hist(circularity, bins=50)
plt.xlabel('Circularity')
plt.ylabel('Frequency')
plt.show()

# Select some appropriate ranges of accepted circularity. Use these ranges to select only the cells with acceptable areas and circularity and show them in an image.
label_img_filter_circ = label_img
min_circ = 0.7
for region in region_props:
    if region.area < min_area or region.area > max_area:
        for coord in region.coords:
            label_img_filter_circ[coord[0], coord[1]] = 0
    else:
        circ = 4 * math.pi * region.area / (region.perimeter ** 2)
        if circ < min_circ:
            for coord in region.coords:
                label_img_filter_circ[coord[0], coord[1]] = 0

i_circ = label_img_filter_circ > 0
show_comparison(img_small, i_circ, 'Area and circularity filtered image')

# Some of the circularities can be higher than 1: Area is counted as full pixels.
# Perimeter is estimated from a jagged pixel boundary.
# For small/compact objects, perimeter may be underestimated.
# If perimeter is too small, circularity becomes artificially high.

# Extend your method to return the number (the count) of well-formed nuclei in the image.

def cell_counting_function(img_gray, min_area=60, max_area=110, min_circ=0.7):
    binary_img = img_gray > threshold_otsu(img_gray)
    binary_img = morphology.remove_small_objects(binary_img, min_size=50)
    img_c_b = segmentation.clear_border(binary_img)
    label_img = measure.label(img_c_b)
    region_props = measure.regionprops(label_img)

    n_nuclei = region_props.__len__()

    label_img_filter_circ = label_img
    for region in region_props:
        if region.area < min_area or region.area > max_area:
            n_nuclei -= 1
            for coord in region.coords:
                label_img_filter_circ[coord[0], coord[1]] = 0
        else:
            circ = 4 * math.pi * region.area / (region.perimeter ** 2)
            if circ < min_circ:
                n_nuclei -= 1
                for coord in region.coords:
                    label_img_filter_circ[coord[0], coord[1]] = 0
    
    i_area_circ = label_img_filter_circ > 0
    return n_nuclei, i_area_circ

n_nuclei, i_area_circ = cell_counting_function(img_gray, min_area=60, max_area=110, min_circ=0.7)
print(f'Number of nuclei: {n_nuclei}')
show_comparison(img_small, i_area_circ, 'Area and circularity filtered image')

# Exercise 15: large scale testing¶
# Try to test the method on a larger set of training images. Use slicing to select the different regions from the raw image

img_org = io.imread(in_dir + 'Sample E2 - U2OS DAPI channel.tiff')
img_gray = img_as_ubyte(img_org)
n_nuclei, i_area_circ = cell_counting_function(img_gray, min_area=60, max_area=110, min_circ=0.7)
print(f'Number of nuclei: {n_nuclei}')
show_comparison(img_org, i_area_circ, 'Area and circularity filtered image')

# Exercise 16: COS7 cell classification¶
# Try your method on the Sample G1 - COS7 cells DAPI channel.tiff image. COS7 cells are African Green Monkey Fibroblast-like Kidney Cells used for a variety of research purposes.
img_org = io.imread(in_dir + 'Sample G1 - COS7 cells DAPI channel.tiff')
img_gray = img_as_ubyte(img_org)
n_nuclei, i_area_circ = cell_counting_function(img_gray, min_area=60, max_area=110, min_circ=0.7)
print(f'Number of nuclei: {n_nuclei}')
show_comparison(img_org, i_area_circ, 'Area and circularity filtered image')

#Exercise 17: Handling overlap¶
# In certain cases cell nuclei are touching and are therefore being treated as one object. It can sometimes be solved using for example the morphological operation opening before the object labelling. The operation erosion can also be used but it changes the object area.

def cell_counting_function_with_opening(img_gray, min_area=60, max_area=110, min_circ=0.7, opening_radius=1):
    binary_img = img_gray > threshold_otsu(img_gray)
    binary_img = morphology.remove_small_objects(binary_img, min_size=50)
    img_c_b = segmentation.clear_border(binary_img)
    selem = morphology.disk(opening_radius)
    img_opened = morphology.opening(img_c_b, selem)
    label_img = measure.label(img_opened)
    region_props = measure.regionprops(label_img)

    n_nuclei = region_props.__len__()

    label_img_filter_circ = label_img
    for region in region_props:
        if region.area < min_area or region.area > max_area:
            n_nuclei -= 1
            for coord in region.coords:
                label_img_filter_circ[coord[0], coord[1]] = 0
        else:
            circ = 4 * math.pi * region.area / (region.perimeter ** 2)
            if circ < min_circ:
                n_nuclei -= 1
                for coord in region.coords:
                    label_img_filter_circ[coord[0], coord[1]] = 0
    
    i_area_circ = label_img_filter_circ > 0
    return n_nuclei, i_area_circ

n_nuclei, i_area_circ = cell_counting_function_with_opening(img_gray, min_area=60, max_area=110, min_circ=0.7, opening_radius=1)
print(f'Number of nuclei: {n_nuclei}')
show_comparison(img_org, i_area_circ, 'Area and circularity filtered image with opening')

