import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte


#You start your analysis by converting the image to gray scale using rgb2gray from scikit-image. Secondly, you use a threshold of 0.6 and set all pixels with a value below to foreground (1) and the rest of the pixel to background (0). How many foreground pixels are there in this binary image?
img_path = 'data/candy/candy.jpg'
img = io.imread(img_path)
gray_img = rgb2gray(img)
binary_img = gray_img < 0.6 
foreground_pixels = np.sum(binary_img)
print(f"Number of foreground pixels: {foreground_pixels}")

#use outsu method 
from skimage.filters import threshold_otsu
threshold_value = threshold_otsu(gray_img)
print (f"Otsu's threshold value: {threshold_value}")

#After creating the binary image (with threshold 0.6) you clean the image by first doing a morphological closing with a disk-shaped structuring element with radius 3 followed by a morphological erosion with a disk-shaped structuring element with radius 6. 
# After cleaning, you do a BLOB analysis of the binary image. How many BLOBs do you find?
from skimage.morphology import closing, erosion, disk
from skimage.measure import label, regionprops
# Create structuring elements
selem_closing = disk(3)
selem_erosion = disk(6)
# Perform morphological closing followed by erosion
closed_img = closing(binary_img, selem_closing)
cleaned_img = erosion(closed_img, selem_erosion)

# Perform BLOB analysis
labeled_img = label(cleaned_img)
blobs = regionprops(labeled_img)

print(f"Number of BLOBs: {len(blobs)}")

#To be able to classify the candy, you compute a series of values per BLOB. They are (per BLOB):
#The average B value of the pixels inside the BLOB sampled from the original image values

average_b_values = []
for region in blobs:

    # Coordinates of pixels inside the blob
    coords = region.coords

    # Extract B channel values from original RGB image
    b_values = img[coords[:, 0], coords[:, 1], 2]

    # Mean B value
    mean_b = np.mean(b_values)

    average_b_values.append(mean_b)

threshold = np.quantile(average_b_values, 0.75)
print(f"Threshold for classification (75th percentile): {threshold}")

#You manually select a B threshold of 65 and classify all BLOBs with an average B value above this value to be "blue M&M". How many false positives do you get?
blue_candidates = []

for i, region in enumerate(blobs):

    coords = region.coords

    b_values = img[coords[:, 0], coords[:, 1], 2]
    mean_b = np.mean(b_values)

    if mean_b > 65:
        blue_candidates.append(i)

print(f"Predicted blue blobs: {len(blue_candidates)}")

#To use the combined set of features, you want to do a principal component analysis (PCA). To do that you start by gathering the nine measured features for each BLOB into a data matrix, where one row is the features from on BLOB. Secondly, you subtract the mean from each feature and divide by the standard deviation of the feature.
# ou use the functions cov and linalg.eig from Numpy to compute the Eigenvectors and the Eigenvalues of the data matrix. 
# How much of the total variation is explained by the first three principal components?
from sklearn.decomposition import PCA
#create feature matrix

regions = regionprops(labeled_img)

feature_list = []

for region in regions:

    coords = region.coords

    # RGB values inside blob
    r = img[coords[:, 0], coords[:, 1], 0]
    g = img[coords[:, 0], coords[:, 1], 1]
    b = img[coords[:, 0], coords[:, 1], 2]

    # Features
    mean_r = np.mean(r)
    mean_g = np.mean(g)
    mean_b = np.mean(b)

    std_r = np.std(r)
    std_g = np.std(g)
    std_b = np.std(b)

    area = region.area
    perimeter = region.perimeter

    circularity = (2 * np.sqrt(np.pi * area)) / perimeter

    features = [
        mean_r,
        mean_g,
        mean_b,
        std_r,
        std_g,
        std_b,
        area,
        perimeter,
        circularity
    ]

    feature_list.append(features)

# Convert to matrix
X = np.array(feature_list)

print(X.shape)
X_standardized = (
    X - np.mean(X, axis=0)
) / np.std(X, axis=0)

# Covariance matrix
cov_matrix = np.cov(X_standardized, rowvar=False)

# Eigen decomposition
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

# Sort eigenvalues descending
eig_vals = np.sort(eig_vals)[::-1]

# Explained variance ratio
explained = np.sum(eig_vals[:3]) / np.sum(eig_vals)

print("Explained variation:", explained)
print("Percent:", explained * 100)

#Project original data onto top 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_standardized)
print("Shape of PCA-transformed data:", X_pca.shape)
#print the plot of original data projected onto the first two principal components
import matplotlib.pyplot as plt
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of BLOB features')
plt.show()
#In your first attempt to classify using the PCA space, you examine the PCA coordinates of each sample. You use the first two coordinates and take the absolute value of both of them. You select all BLOBs, where the absolute value of the first component is larger than 1.5 and the absolute value of the second component is larger than 2. Finally, you show the selected BLOBs. It looks like:

pc1 = X_pca[:, 0]
pc2 = X_pca[:, 1]

# Selection rule
selected = (np.abs(pc1) > 1.5) & (np.abs(pc2) > 2)
# Indices of selected BLOBs
selected_indices = np.where(selected)[0]

print("Selected BLOB indices:", selected_indices)
print("Number of selected BLOBs:", len(selected_indices))

selected_mask = np.zeros_like(cleaned_img, dtype=bool)

for i in selected_indices:
    selected_mask[labeled_img == regions[i].label] = True

plt.imshow(selected_mask, cmap="gray")
plt.title("Selected BLOBs")
plt.show()

#To get an idea of the spread of the candy in the PCA space, the two BLOBs that are furthest away from each other  when only using the two first principal components are identified. By creating a binary BLOB image with only the two BLOBs, you mask the original RGB image and show the result. You start by filling the masked image with white pixels. How does this masked image look?
from skimage.color import gray2rgb
# Get coordinates of the two furthest BLOBs in PCA space

X_pca_2 = X_pca[:, :2]

# Pairwise distances between all BLOBs
from scipy.spatial.distance import pdist, squareform
dist_matrix = squareform(pdist(X_pca_2))

# Find pair with maximum distance
i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)

print("Farthest BLOB indices:", i, j)

# Create binary mask with only the two farthest BLOBs
farthest_mask = np.zeros_like(cleaned_img, dtype=bool)

farthest_mask[labeled_img == regions[i].label] = True
farthest_mask[labeled_img == regions[j].label] = True

# Start with a white RGB image
masked_img = np.ones_like(img) * 255

# Copy original pixels only inside the two selected BLOBs
masked_img[farthest_mask] = img[farthest_mask]

# Show result
plt.imshow(masked_img)
plt.title("Two farthest BLOBs in PCA space")
plt.axis("off")
plt.show()



all_images = ["1-020.dcm", "1-021.dcm", "1-022.dcm", "1-023.dcm", "1-024.dcm", "1-025.dcm", "1-026.dcm", "1-027.dcm", "1-028.dcm", "1-029.dcm", "1-030.dcm"]
import pydicom as dicom
data_dir = "data/ct/"
ct = dicom.dcmread(data_dir + "1-020.dcm")
background_img = ct.pixel_array.astype(float)

alpha = 0.85

for img_name in all_images[1:]:
    ct = dicom.dcmread(data_dir + img_name)
    current_img = ct.pixel_array.astype(float)

    background_img = alpha * background_img + (1 - alpha) * current_img

test_ct = dicom.dcmread(data_dir + "1-040.dcm")
test_img = test_ct.pixel_array.astype(float)

diff_img = np.abs(test_img - background_img)

binary_img = diff_img > 400

foreground_pixels = np.sum(binary_img)

print("Max background:", np.max(background_img))
print("Max diff:", np.max(diff_img))
print("Foreground pixels:", foreground_pixels)

#To try another approach, you start by thresholding the estimated background image with a threshold of 400, where pixels above threshold is classified as foreground (1) and the rest as background (0). You do the same with the test image 1-040.dcm. Finally, you compare the two binary images using the DICE score. What is the DICE score?
from scipy.spatial import distance

background_binary = background_img > 400
test_binary = test_img > 400

dice_score_2 = 1 - distance.dice(
    background_binary.ravel(),
    test_binary.ravel()
)

print("DICE score:", dice_score_2)

#Overlaping pixels between the two binary images
overlap_img = background_binary & test_binary

# BLOB analysis
label_overlap = label(overlap_img)

regions_overlap = regionprops(label_overlap)

# Find largest BLOB
largest_blob = max(regions_overlap, key=lambda r: r.area)

print("Largest BLOB area:", largest_blob.area)

mm_img =io.imread("data/candy/mm.jpg")
# converting the image to gray scale using rgb2gray from scikit-image
gray_mm = rgb2gray(mm_img)
#threshold of 0.65 and set all pixels with a value below to foreground (1) and the rest of the pixel to background (0).
binary_mm = gray_mm < 0.65
#morphological closing with a disk-shaped structuring element with radius 3, followed by a morphological erosion with disk-shaped structuring element with radius 1. 

selem_closing_mm = disk(3)
selem_erosion_mm = disk(1)
closed_mm = closing(binary_mm, selem_closing_mm)
cleaned_mm = erosion(closed_mm, selem_erosion_mm)
#BLOB analysis of the binary image
labeled_mm = label(cleaned_mm)
regions_mm = regionprops(labeled_mm)
print(f"Number of BLOBs in MM image: {len(regions_mm)}")

#circularity is computed as  (2 * sqrt(pi * area)) / perimeter). You keep all BLOBs with a circularity larger than 0.7 and an area larger than 100 pixels
selected_indices = []
for i, region in enumerate(regions_mm):
    area = region.area
    perimeter = region.perimeter
    circularity = (2 * np.sqrt(np.pi * area)) / perimeter

    if circularity > 0.7 and area > 100:
        selected_indices.append(i)

print(f"Number of selected BLOBs in MM image: {len(selected_indices)}")

#compute the center of mass of each BLOB that is classified as a candy. For each center of mass you compute, the Euclidean distance to the center of the image (in pixels). The spread of the candy can be measured by the average and standard deviation of these distances.
from scipy.spatial import distance
image_center = np.array(gray_mm.shape) / 2
distances = []
for i in selected_indices:
    region = regions_mm[i]
    center_of_mass = region.centroid
    dist = distance.euclidean(center_of_mass, image_center)
    distances.append(dist)

print(f"Average distance: {np.mean(distances)}")
print(f"Standard deviation: {np.std(distances)}")

# computing the center of mass of each BLOB that is classified as a candy. This position is used as a center of a rectangular crop that has a side length of 100 pixels.
crops = []
for i in selected_indices:
    region = regions_mm[i]
    center_of_mass = region.centroid
    y0 = int(center_of_mass[0] - 50)
    y1 = int(center_of_mass[0] + 50)
    x0 = int(center_of_mass[1] - 50)
    x1 = int(center_of_mass[1] + 50)

    crop = mm_img[y0:y1, x0:x1]
    crops.append(crop)

#Using these crop coordinates a crop is extracted from the original color image for each BLOB that is classified as candy.
#  The result of this is a list of rectangular crops each containing a color image of a candy in the middle.
crops = [crop for crop in crops if crop.shape == (100, 100, 3)]
crop_array = np.array(crops)

# Average image
avg_img = np.mean(crop_array, axis=0)

# Show average candy
plt.imshow(avg_img.astype(np.uint8))
plt.axis("off")
plt.title("Average candy image")
plt.show()

max_ssd = -1
most_different_pair = (None, None)
num_crops = len(crops)


for i in range(num_crops):
    for j in range(i + 1, num_crops):

        crop1 = crops[i].astype(float)
        crop2 = crops[j].astype(float)

        # Sum of squared differences
        ssd = np.sum((crop1 - crop2) ** 2)

        if ssd > max_ssd:
            max_ssd = ssd
            most_different_pair = (i, j)

print("Most different pair:", most_different_pair)
print("Largest SSD:", max_ssd)

i, j = most_different_pair

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].imshow(crops[i])
ax[0].set_title(f"Crop {i}")

ax[1].imshow(crops[j])
ax[1].set_title(f"Crop {j}")

for a in ax:
    a.axis("off")

plt.show()

training_tiger = io.imread("data/tiger/Tiger.png")
roi = io.imread("data/tiger/ROI_Tiger.png")

# Convert tiger image to grayscale byte image
gray_tiger = img_as_ubyte(training_tiger)

# If ROI has RGB channels, take one channel
if roi.ndim == 3:
    roi = roi[:, :, 0]

# Extract training pixels
class1_pixels = gray_tiger[roi == 90]    # black stripes
class2_pixels = gray_tiger[roi == 165]   # not black stripes

# Estimate Gaussian parameters
mu1 = np.mean(class1_pixels)
std1 = np.std(class1_pixels)

mu2 = np.mean(class2_pixels)
std2 = np.std(class2_pixels)

print("Class 1 mean/std:", mu1, std1)
print("Class 2 mean/std:", mu2, std2)
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from scipy.stats import norm
# Compute probability of each pixel belonging to each class
prob1 = norm.pdf(gray_tiger, mu1, std1)
prob2 = norm.pdf(gray_tiger, mu2, std2)

# Classify pixels
class1_result = prob1 > prob2

# Show class 1 pixels as white
plt.imshow(class1_result, cmap="gray")
plt.title("Detected black tiger stripes")
plt.axis("off")
plt.show()

# Training pixels
class1_pixels = gray_tiger[roi == 90]    # black stripes
class2_pixels = gray_tiger[roi == 165]   # not black stripes

# Means and stds
mu1 = np.mean(class1_pixels)
std1 = np.std(class1_pixels)

mu2 = np.mean(class2_pixels)
std2 = np.std(class2_pixels)

# Pixel position
r, c = 347, 247
x = gray_tiger[r, c]

# Equal priors
prior1 = 0.5
prior2 = 0.5

# Likelihoods
p_x_given_c1 = norm.pdf(x, mu1, std1)
p_x_given_c2 = norm.pdf(x, mu2, std2)

# Posterior probability for class 1
p_class1 = (p_x_given_c1 * prior1) / (
    p_x_given_c1 * prior1 + p_x_given_c2 * prior2
)

print("Pixel value:", x)
print("Probability of class 1:", p_class1)

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte

def imshow_orthogonal_view(sitkImage, origin = None, title=None):
    """
    Display the orthogonal views of a 3D volume from the middle of the volume.

    Parameters
    ----------
    sitkImage : SimpleITK image
        Image to display.
    origin : array_like, optional
        Origin of the orthogonal views, represented by a point [x,y,z].
        If None, the middle of the volume is used.
    title : str, optional
        Super title of the figure.

    Note:
    On the axial and coronal views, patient's left is on the right
    On the sagittal view, patient's anterior is on the left
    """
    data = sitk.GetArrayFromImage(sitkImage)

    if origin is None:
        origin = np.array(data.shape) // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    data = data.astype(float)
    data[data < 0] = 0
    data = data / np.max(data)
    data = np.clip(data, 0, 1)
    data = img_as_ubyte(data)    
    axes[0].imshow(data[origin[0], ::-1, ::-1], cmap='gray')
    axes[0].set_title('Axial')

    axes[1].imshow(data[::-1, origin[1], ::-1], cmap='gray')
    axes[1].set_title('Coronal')

    axes[2].imshow(data[::-1, ::-1, origin[2]], cmap='gray')
    axes[2].set_title('Sagittal')

    [ax.set_axis_off() for ax in axes]

    if title is not None:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.show()

def overlay_slices(sitkImage0, sitkImage1, origin = None, title=None):
    """
    Overlay the orthogonal views of a two 3D volume from the middle of the volume.
    The two volumes must have the same shape. The first volume is displayed in red,
    the second in green.

    Parameters
    ----------
    sitkImage0 : SimpleITK image
        Image to display in red.
    sitkImage1 : SimpleITK image
        Image to display in green.
    origin : array_like, optional
        Origin of the orthogonal views, represented by a point [x,y,z].
        If None, the middle of the volume is used.
    title : str, optional
        Super title of the figure.

    Note:
    On the axial and coronal views, patient's left is on the right
    On the sagittal view, patient's anterior is on the left
    """
    vol0 = sitk.GetArrayFromImage(sitkImage0)
    vol1 = sitk.GetArrayFromImage(sitkImage1)

    if vol0.shape != vol1.shape:
        raise ValueError('The two volumes must have the same shape.')
    if np.min(vol0) < 0 or np.min(vol1) < 0: # Remove negative values - Relevant for the noisy images
        vol0[vol0 < 0] = 0
        vol1[vol1 < 0] = 0
    if origin is None:
        origin = np.array(vol0.shape) // 2

    sh = vol0.shape
    R = img_as_ubyte(vol0/np.max(vol0))
    G = img_as_ubyte(vol1/np.max(vol1))

    vol_rgb = np.zeros(shape=(sh[0], sh[1], sh[2], 3), dtype=np.uint8)
    vol_rgb[:, :, :, 0] = R
    vol_rgb[:, :, :, 1] = G

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(vol_rgb[origin[0], ::-1, ::-1, :])
    axes[0].set_title('Axial')

    axes[1].imshow(vol_rgb[::-1, origin[1], ::-1, :])
    axes[1].set_title('Coronal')

    axes[2].imshow(vol_rgb[::-1, ::-1, origin[2], :])
    axes[2].set_title('Sagittal')

    [ax.set_axis_off() for ax in axes]

    if title is not None:
        fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def composite2affine(composite_transform, result_center=None):
    """
    Combine all of the composite transformation's contents to form an equivalent affine transformation.
    Args:
        composite_transform (SimpleITK.CompositeTransform): Input composite transform which contains only
                                                            global transformations, possibly nested.
        result_center (tuple,list): The desired center parameter for the resulting affine transformation.
                                    If None, then set to [0,...]. This can be any arbitrary value, as it is
                                    possible to change the transform center without changing the transformation
                                    effect.
    Returns:
        SimpleITK.AffineTransform: Affine transformation that has the same effect as the input composite_transform.

    Source:
        https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/22_Transforms.ipynb
    """
    # Flatten the copy of the composite transform, so no nested composites.
    flattened_composite_transform = sitk.CompositeTransform(composite_transform)
    flattened_composite_transform.FlattenTransform()
    tx_dim = flattened_composite_transform.GetDimension()
    A = np.eye(tx_dim)
    c = np.zeros(tx_dim) if result_center is None else result_center
    t = np.zeros(tx_dim)
    for i in range(flattened_composite_transform.GetNumberOfTransforms() - 1, -1, -1):
        curr_tx = flattened_composite_transform.GetNthTransform(i).Downcast()
        # The TranslationTransform interface is different from other
        # global transformations.
        if curr_tx.GetTransformEnum() == sitk.sitkTranslation:
            A_curr = np.eye(tx_dim)
            t_curr = np.asarray(curr_tx.GetOffset())
            c_curr = np.zeros(tx_dim)
        else:
            A_curr = np.asarray(curr_tx.GetMatrix()).reshape(tx_dim, tx_dim)
            c_curr = np.asarray(curr_tx.GetCenter())
            # Some global transformations do not have a translation
            # (e.g. ScaleTransform, VersorTransform)
            get_translation = getattr(curr_tx, "GetTranslation", None)
            if get_translation is not None:
                t_curr = np.asarray(get_translation())
            else:
                t_curr = np.zeros(tx_dim)
        A = np.dot(A_curr, A)
        t = np.dot(A_curr, t + c - c_curr) + t_curr + c_curr - c

    return sitk.AffineTransform(A.flatten(), t, c)

imgA = sitk.ReadImage("data/volumes/VolA.nii")
imgB = sitk.ReadImage("data/volumes/VolB.nii")

# Load forward matrix
T = np.loadtxt("data/volumes/matrix.txt")

# Translation
Tt = [-2, -10, 5]

centre_image = np.array(imgA.GetSize()) / 2 - 0.5
centre_world = imgA.TransformContinuousIndexToPhysicalPoint(centre_image)



# Create forward affine transform
forward_transform = sitk.AffineTransform(3)
forward_transform.SetCenter(centre_world)
forward_transform.SetMatrix(T.flatten())
forward_transform.SetTranslation(Tt)

# Use backward transform for resampling
backward_transform = forward_transform.GetInverse()

# Resample B into A's coordinate system
imgBB = sitk.Resample(
    imgB,                 # moving image
    imgA,                 # reference image
    backward_transform,   # backward transform
    sitk.sitkLinear,      # linear interpolation
    0.0,                  # default value outside image
    imgB.GetPixelID()
)

A = sitk.GetArrayFromImage(imgA).astype(float)
BB = sitk.GetArrayFromImage(imgBB).astype(float)

overlap = (A > 0) & (BB > 0)

stitched = A.copy()

# Use BB where A is empty
stitched[(A == 0) & (BB > 0)] = BB[(A == 0) & (BB > 0)]

# Blend overlap: 80% BB, 20% A
stitched[overlap] = 0.8 * BB[overlap] + 0.2 * A[overlap]

stitched_img = sitk.GetImageFromArray(stitched)
stitched_img.CopyInformation(imgA)
overlay_slices(imgA, imgBB, origin=np.array([59, 100, 100]))
imshow_orthogonal_view(
    stitched_img,
    origin=np.array([59, 100, 100]),
    title="Stitched image at [59, 100, 100]"
)