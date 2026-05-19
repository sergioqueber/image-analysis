from skimage import io, color
from skimage.morphology import binary_closing, binary_opening
from skimage.morphology import disk
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage.color import label2rgb
import pydicom as dicom
from scipy.stats import norm
from scipy.spatial import distance


def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap="gray", vmin=-200, vmax=500)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(modified)
    ax2.set_title(modified_name)
    ax2.axis('off')
    io.show()

in_dir = "data/"
ct = dicom.dcmread(in_dir + 'Training.dcm')
img = ct.pixel_array
print(img.shape)
print(img.dtype)

io.imshow(img, cmap="gray", vmin=0, vmax=150)
io.show()

spleen_roi = io.imread(in_dir + 'SpleenROI.png')
# convert to boolean image
spleen_mask = spleen_roi > 0
spleen_values = img[spleen_mask]

#Show the masked image together with the original image
masked_img = np.zeros_like(img)
masked_img[spleen_mask] = img[spleen_mask]

io.imshow(masked_img, cmap="gray", vmin=0, vmax=150)
io.show()

#Exercise 2: Compute the average and standard deviation of the
# Hounsfield units found in the spleen in the training image. Do they correspond to the values found in the above figure?
mean_spleen = np.mean(spleen_values)
std_spleen = np.std(spleen_values)
print(f'Mean spleen HU: {mean_spleen:.2f}')
print(f'Standard deviation spleen HU: {std_spleen:.2f}')

#Mean spleen HU: 49.48
#Standard deviation spleen HU: 15.00

#This results are expected as the mean value is around 50 HU and the standard deviation is around 15 HU, which corresponds to
# the indicated values between 0 and 150 HU indicted in the exercise. 

# Exercise 3: Plot a histogram of the pixel values of the spleen. Does it look like they are Gaussian distributed?

plt.hist(spleen_values, bins=30, density=True, alpha=0.6, color='g') 
plt.xlabel('Hounsfield Units')
plt.ylabel('Density')
plt.title('Histogram of Spleen Pixel Values')
plt.show()

# The histogram of the spleen pixel values looks similar to a Gaussian distribution, with a single peak around the mean value of approximately 50 HU.
#  The distribution appears to be somewhat symmetric around the mean, which is characteristic of a Gaussian distribution.
#  However, there may be some skewness or outliers present, so it may not be a perfect Gaussian distribution.

n, bins, patches = plt.hist(spleen_values, 30, density=True)
pdf_spleen = norm.pdf(bins, mean_spleen, std_spleen)
plt.plot(bins, pdf_spleen)
plt.xlabel('Hounsfield unit')
plt.ylabel('Frequency')
plt.title('Spleen values in CT scan')
plt.show()

# Exercise 4: Plot histograms and their fitted Gaussians
#  of several of the tissues types. Do they all look like they are Gaussian distributed?

def load_mask(mask_path):
    roi = io.imread(mask_path)
    mask = roi > 0
    return mask

def get_values_in_mask(img, mask_path):
    mask = load_mask(mask_path)
    return img[mask]

paths = ['BoneROI.png', 'FatROI.png', 'KidneyROI.png', 'LiverROI.png', 'SpleenROI.png']
values = [get_values_in_mask(img, in_dir + path) for path in paths]

def get_gaussian_distributions(values, min_hu=-200, max_hu=500, bins=60):
    plt.figure(figsize=(12, 8))
    for i, tissue_values in enumerate(values):
        mean = np.mean(tissue_values)
        std = np.std(tissue_values)
        n, bins, patches = plt.hist(tissue_values, bins=bins, density=True, alpha=0.6, label=f'Tissue {i+1}')
        pdf = norm.pdf(bins, mean, std)
        plt.plot(bins, pdf, label=f'Gaussian Fit {i+1}')
    plt.xlim(min_hu, max_hu)
    plt.xlabel('Hounsfield Units')
    plt.ylabel('Density')
    plt.title('Histograms and Gaussian Fits of Tissue Types')
    plt.legend()
    plt.show()
    
def get_gaussian_distributionsV2(values, min_hu = -200, max_hu = 1000):
    hu_range = np.arange(min_hu, max_hu, 1.0)
    mu = np.mean(values)
    std = np.std(values)
    pdf = norm.pdf(hu_range, mu, std)
    return pdf

for value, name in zip(values, paths):
    n, bins, patches = plt.hist(value, 60, density=1)
    mu = np.mean(value)
    std = np.std(value)
    pdf = norm.pdf(bins, mu, std)
    plt.plot(bins, pdf)
    plt.xlabel('Hounsfield unit')
    plt.ylabel('Frequency')
    plt.title(name)
    plt.show()


# Most distributions fit the gaussian except for the bones 

#Exercise 5: Plot the fitted Gaussians of bone, fat, kidneys, liver and spleen. What classes are easy to seperate and which classes are hard to seperate?

pdfs = [get_gaussian_distributionsV2(value) for value in values]
hu_range = np.arange(-200, 1000, 1.0)
plt.figure(figsize=(12, 8))
for pdf, path in zip(pdfs, paths):
    plt.plot(hu_range, pdf, label=f'{path[:-7]} Gaussian Fit')
plt.xlabel('Hounsfield Units')
plt.ylabel('Density')
plt.title('Fitted Gaussians of Tissue Types')
plt.legend()
plt.show()

# Exercise 6: Define the classes that we aim at classifying. Perhaps some classes should be combined into one class?
#We can differenciate intensity-wise the fat, soft tissues (kidney, liver, spleen) and bone.

#In the minimum distance classifier the pixel value class ranges are defined using the average values of the training values. If you have two classes, the threshold between them is defined as the mid-point between the two class value averages.
#In the following, we will define four classes: background, fat, soft tissue and bone, where soft-tissue is a combination of the values of the spleen, liver and kidneys. We manually set the threshold for background to -200. So all pixels below -200 are set to background.

#Exercise 7: Compute the class ranges defining fat, soft tissue and bone.
vals_soft_tissue = np.concatenate(values[-3:])
soft_tissue_mean = np.mean(vals_soft_tissue)
bone_mean = np.mean(values[0])
fat_mean = np.mean(values[1])
print(f'Soft tissue mean: {soft_tissue_mean:.2f}')
print(f'Bone mean: {bone_mean:.2f}')
print(f'Fat mean: {fat_mean:.2f}')

t_fat_soft = (soft_tissue_mean + fat_mean) / 2
t_soft_bone = (soft_tissue_mean + bone_mean) / 2
print(f'Threshold between fat and soft tissue: {t_fat_soft:.2f}')
print(f'Threshold between soft tissue and bone: {t_soft_bone:.2f}')

#Exercise 8: Create class images: fat_img, soft_img and bone_img representing the fat, soft tissue and bone found in the image.

t_background = -200
fat_img = (img > t_background) & (img <= t_fat_soft)
soft_img = (img > t_fat_soft) & (img <= t_soft_bone)
bone_img = img > t_soft_bone

label_img = fat_img + 2 * soft_img + 3 * bone_img
image_label_overlay = label2rgb(label_img)
show_comparison(img, image_label_overlay, 'Classification result')

#Exercise 9: Visualize your classification result and compare it to the anatomical image in the start of the exercise. Does your results look plausible?

#It does not really fully match they are not well segmented, bone is barely represented, vertebra are still blue, while soft tissue and fat are
# a bit better separated. 

#Parametric pixel classification¶
# In the parametric classifier, the standard deviation of the training pixel values is also used when determinin the class ranges. In the following, we are also trying to classify background, fat, soft tissue and bone.
# We start by finding the class ranges by manually inspecting the fitted Gaussians from each class.
# As in the last exercise, we can still se the background-fat threshold to be -200.
# Exercise 10: Plot the fitted Gaussians of the training values and manually find the intersection between the curves.

t_fat_soft = -45
t_soft_bone = 140
print(f'Threshold between fat and soft tissue: {t_fat_soft:.2f}')
print(f'Threshold between soft tissue and bone: {t_soft_bone:.2f}')

#Exercise 8: Create class images: fat_img, soft_img and bone_img representing the fat, soft tissue and bone found in the image.
#Exercise 11: Use the same technique as in exercise 7, 8 and 9 to visualize your classification results. Did it change compared to the minimum distance classifier?

t_background = -200
fat_img = (img > t_background) & (img <= t_fat_soft)
soft_img = (img > t_fat_soft) & (img <= t_soft_bone)
bone_img = img > t_soft_bone

label_img = fat_img + 2 * soft_img + 3 * bone_img
image_label_overlay = label2rgb(label_img)
show_comparison(img, image_label_overlay, 'Classification result')

#Exercise 12: Use norm.pdf to find the optimal class ranges between fat, soft tissue and bone.

#Automatically finding the value 
mu_fat = np.mean(values[1])
std_fat = np.std(values[1])
mu_soft = np.mean(vals_soft_tissue)
std_soft = np.std(vals_soft_tissue)

for test_value in np.linspace(mu_fat, mu_soft, 1000):
    pdf_fat = norm.pdf(test_value, mu_fat, std_fat) 
    pdf_soft = norm.pdf(test_value, mu_soft, std_soft)
    if pdf_soft > pdf_fat:
        t_fat_soft_auto = test_value
        print(f'Optimal threshold between fat and soft tissue: {t_fat_soft_auto:.2f}')
        break

for test_value in np.linspace(mu_soft, bone_mean, 1000):
    pdf_soft = norm.pdf(test_value, mu_soft, std_soft) 
    pdf_bone = norm.pdf(test_value, bone_mean, np.std(values[0]))
    if pdf_bone > pdf_soft:
        t_soft_bone_auto = test_value
        print(f'Optimal threshold between soft tissue and bone: {t_soft_bone_auto:.2f}')
        break

# What we are doing here is checking the curves which give us the probability of each of the classes, and when one of the probabilities becomes higher than the other, we set that value as the threshold. 
# The optimal thresholds are: 
# Optimal threshold between fat and soft tissue: -44.33
# Optimal threshold between soft tissue and bone: 140.84

#Object segmentation - The spleen finder¶
# The goal of this part of the exercise, is to create a program that can automatically segment the spleen in CT images.
# We start by using the Training.dcm image and the expert provided annotations.

# Exercise 13: Inspect the values of the spleen as in exercise 3 and select a lower and upper threshold to create a spleen class range.

t_1, t_2 = 22, 77

spleen_estimate = (img > t_1) & (img <= t_2)
spleen_label_color = color.label2rgb(spleen_estimate)
io.imshow(spleen_label_color)
plt.title('Spleen estimate')
io.show()

# This clearly shows that just using the treshold values is not enough to segment the spleen, as there are many other structures
#  in the image that have similar intensity values. We can see that
#  the vertebrae and the liver are also included in the spleen estimate, which is not correct.

#Exercise 14: Use morphological operations to seperate the spleen from other organs and close holes.
#  Change the values where there are question marks to change the size of the used structuring elements.

for r in [1, 3, 5, 7, 10]:
    footprint = disk(r)
    footprintOpen = disk(r+1)
    closed = binary_closing(spleen_estimate, footprint)
    opened = binary_opening(closed, footprint)

    plt.figure()
    plt.imshow(opened, cmap="gray")
    plt.title(f"Radius = {r}")
    plt.show()

footprint = disk(2)
closed = binary_closing(spleen_estimate, footprint)
opened = binary_opening(closed, disk(4))

#seems that radious 2 and 4 is enought to show the spleen disconnected. 
# Now we can use BLOB analysis to do a feature based classification of the spleen.
# Exercise 15: Use the methods from BLOB analysis to compute BLOB features for every seperated BLOB in the image.

label_img = measure.label(opened)
im_blob = label2rgb(label_img)
plt.imshow(im_blob)
plt.title('BLOB analysis')  
plt.show()

#Exercise 16: Inspect the labeled image and validate the success of separating the spleen from the other objects. If it is connected (have the same color) to another organ, you should experiment with the kernel sizes in the morphological operations.
# To be able to keep only the spleen we need to find out which BLOB features,
#  that is special for the spleen. By using measure.regionprops many different BLOB
#  features can be computed, including area and perimeter. You can find the catalog of available features from here.
# Exercise 17: Using a combination of features and feature value limits, filter the image such that only the spleen remains in the output image.
region = measure.regionprops(label_img)
areas = [r.area for r in region]
plt.hist(areas, bins=20)
plt.xlabel('Area')
plt.ylabel('Frequency')
plt.show()

perimeters = [r.perimeter for r in region]
plt.hist(perimeters, bins=20)
plt.xlabel('Perimeter')
plt.ylabel('Frequency')
plt.show()

min_area = 2000
max_area = 5000
min_perimeter = 200
max_perimeter = 350
lavel_img_filter = label_img.copy()
for region in measure.regionprops(label_img):
    if region.area < min_area or region.area > max_area:
        for coord in region.coords:
            lavel_img_filter[coord[0], coord[1]] = 0
    elif region.perimeter < min_perimeter or region.perimeter > max_perimeter:
        for coord in region.coords:
            lavel_img_filter[coord[0], coord[1]] = 0

i_area = lavel_img_filter > 0
show_comparison(img, i_area, 'Spleen segmentation')

#Exercise 18: Create a function spleen_finder(img) that takes as input a CT image and returns a binary image,
#  where the pixels with value 1 represent the spleen and the pixels with value 0 everything else.

def spleen_finder(img):
    t_1, t_2 = 22, 77
    spleen_estimate = (img > t_1) & (img <= t_2)
    closed = binary_closing(spleen_estimate, disk(2))
    opened = binary_opening(closed, disk(4))
    label_img = measure.label(opened)
    
    min_area = 2000
    max_area = 7000
    min_perimeter = 100
    max_perimeter = 350
    lavel_img_filter = label_img.copy()
    for region in measure.regionprops(label_img):
        if region.area < min_area or region.area > max_area:
            for coord in region.coords:
                lavel_img_filter[coord[0], coord[1]] = 0
        elif region.perimeter < min_perimeter or region.perimeter > max_perimeter:
            for coord in region.coords:
                lavel_img_filter[coord[0], coord[1]] = 0

    return lavel_img_filter > 0

spleen_estimate = spleen_finder(img)
show_comparison(img, spleen_estimate, 'Spleen segmentation')

#Exercise 19: Test your function on the images called Validation1.dcm, Validation2.dcm and Validation3.dcm. Do you succeed in finding the spleen in all the validation images?

validation_1 = dicom.dcmread(in_dir + 'Validation1.dcm').pixel_array
validation_2 = dicom.dcmread(in_dir + 'Validation2.dcm').pixel_array
validation_3 = dicom.dcmread(in_dir + 'Validation3.dcm').pixel_array

spleen_estimate_1 = spleen_finder(validation_1)
spleen_estimate_2 = spleen_finder(validation_2)
spleen_estimate_3 = spleen_finder(validation_3)
show_comparison(validation_1, spleen_estimate_1, 'Spleen segmentation Validation 1')
show_comparison(validation_2, spleen_estimate_2, 'Spleen segmentation Validation 2')
show_comparison(validation_3, spleen_estimate_3, 'Spleen segmentation Validation 3')

#DICE Score
#We would like evaluate how good we are at finding the spleen
#  by comparing our found spleen with ground truth annotations
#  of the spleen. The DICE score (also called the DICE coefficient
#  or the DICE distance) is a standard method of comparing one segmentation with another segmentation.

ground_truth_1 = io.imread(in_dir + 'Validation1_spleen.png')
ground_truth_2 = io.imread(in_dir + 'Validation2_spleen.png')
ground_truth_3 = io.imread(in_dir + 'Validation3_spleen.png')
gt_bin1 = ground_truth_1 > 0
gt_bin2 = ground_truth_2 > 0
gt_bin3 = ground_truth_3 > 0
dice_score_1 = 1- distance.dice(spleen_estimate_1.ravel(), gt_bin1.ravel())
dice_score_2 = 1- distance.dice(spleen_estimate_2.ravel(), gt_bin2.ravel())
dice_score_3 = 1- distance.dice(spleen_estimate_3.ravel(), gt_bin3.ravel())
print(f'DICE score Validation 1: {dice_score_1:.4f}')
print(f'DICE score Validation 2: {dice_score_2:.4f}')
print(f'DICE score Validation 3: {dice_score_3:.4f}')

#scores are 
#DICE score Validation 1: 0.6293
#DICE score Validation 2: 0.9615
#DICE score Validation 3: 0.9709

#This shows that the spleen finder works well but in the first validation image it detects something besides the spleen. 

# Exercise 21: Use your spleen finder program to find the spleen on the three test images and compute the DICE score. What is the result of your independent test?

def compute_dice_score(estimation, ground_truth):
    return 1 - distance.dice(estimation.ravel(), ground_truth.ravel())

test_1 = dicom.dcmread(in_dir + 'Test1.dcm').pixel_array
test_2 = dicom.dcmread(in_dir + 'Test2.dcm').pixel_array
test_3 = dicom.dcmread(in_dir + 'Test3.dcm').pixel_array

for test_img, gt_path in zip([test_1, test_2, test_3], ['Test1_spleen.png', 'Test2_spleen.png', 'Test3_spleen.png']):
    spleen_estimate = spleen_finder(test_img)
    ground_truth = io.imread(in_dir + gt_path) > 0
    dice_score = compute_dice_score(spleen_estimate, ground_truth)
    print(f'DICE score for {gt_path}: {dice_score:.4f}')

#This test indicates again that the algorithm works well for some images but not for others, which is
#  expected as the algorithm is quite simple and relies on intensity values and basic morphological operations.
