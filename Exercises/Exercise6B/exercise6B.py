import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from skimage.morphology import disk, disk, opening, closing
from Data.LDA import LDA

#Exercise 1 Exercise 1¶
# Display both the T1 and T2 images, their 1 and 2D histograms and scatter plots.
#  Tips: Use the plt.imshow(), plt.hist(), plt.hist2d() and plt.scatter() functions
#  Add relevant title and label for each axis. One can use plt.subplots() to show more
#  subfigures in the same figure. Remove intensities from background voxels for 1D and 2D histograms.


in_dir = 'data/'
in_file = 'ex6_ImagData2Load.mat'

data = sio.loadmat(in_dir + in_file)

ImgT1 = data['ImgT1']
ImgT2 = data['ImgT2']
ROI_GM = data['ROI_GM'].astype(bool)
ROI_WM = data['ROI_WM'].astype(bool)

# Common brain mask: remove background from both images
mask_init = (ImgT1 > 10) & (ImgT2 > 10)

mask = opening(mask_init, disk(3))
mask = closing(mask, disk(3))

# Extract paired intensities from the same voxels
t1_values = ImgT1[mask].flatten()
t2_values = ImgT2[mask].flatten()

plt.figure(figsize=(12, 8))

# T1 image
plt.subplot(2, 3, 1)
plt.imshow(ImgT1, cmap='gray')
plt.title('T1 Image')
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')

# T2 image
plt.subplot(2, 3, 2)
plt.imshow(ImgT2, cmap='gray')
plt.title('T2 Image')
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')

# 1D histograms
plt.subplot(2, 3, 3)
plt.hist(t1_values, bins=50, alpha=0.5, label='T1')
plt.hist(t2_values, bins=50, alpha=0.5, label='T2')
plt.title('1D Histograms without Background')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.legend()

# 2D histogram
plt.subplot(2, 3, 4)
plt.hist2d(t1_values, t2_values, bins=50, cmap='Blues')
plt.title('2D Histogram')
plt.xlabel('T1 Intensity')
plt.ylabel('T2 Intensity')
plt.colorbar(label='Frequency')

# Scatter plot
plt.subplot(2, 3, 5)
plt.scatter(t1_values, t2_values, alpha=0.2, s=2)
plt.title('Scatter Plot')
plt.xlabel('T1 Intensity')
plt.ylabel('T2 Intensity')

# Show mask
plt.subplot(2, 3, 6)
plt.imshow(mask, cmap='gray')
plt.title('Brain Mask')
plt.axis('off')

plt.tight_layout()
plt.show()

#Q1: What is the intensity threshold that can separate the GM and WM classes (roughly) from the 1D histograms? A1: For T1:about 450 and for T2: about 160

#Q2: Can the GM and WM intensity classes be observed in the 2D histogram and scatter plot?
#Yes there appear clear two clusters with maxium intensitised in the 2D histogram that represent the two classes distributions

#Exercise 2¶
# Place trainings examples i.e. ROI_WM and ROI_GM into variables C1 and C2 representing class 1 and class 2 respectively. Show in a figure the manually expert drawings of the C1 and C2 training examples.
C1 = np.column_stack((ImgT1[ROI_WM], ImgT2[ROI_WM]))
C2 = np.column_stack((ImgT1[ROI_GM], ImgT2[ROI_GM]))
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(ROI_WM, cmap='gray')
plt.title('ROI WM')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(ROI_GM, cmap='gray')
plt.title('ROI GM')
plt.axis('off')
plt.tight_layout()
plt.show()

#It looks like some random stripes. 

# Exercise 3
# For each binary training ROI find the corresponding training examples in ImgT1 and ImgT2. 
# Later these will be extracted for LDA training.


trainWM_T1 = ImgT1[ROI_WM]
trainGM_T1 = ImgT1[ROI_GM]

trainWM_T2 = ImgT2[ROI_WM]
trainGM_T2 = ImgT2[ROI_GM]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#Show histograms of the training examples for T1 and T2
axes[0, 0].hist(trainWM_T1, bins=50, alpha=0.5, label='WM T1')
axes[0, 0].hist(trainGM_T1, bins=50, alpha=0.5, label='GM T1')
axes[0, 0].set_title('T1 Training Examples')    
axes[0, 0].set_xlabel('Intensity')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 1].hist(trainWM_T2, bins=50, alpha=0.5, label='WM T2')
axes[0, 1].hist(trainGM_T2, bins=50, alpha=0.5, label='GM T2')
axes[0, 1].set_title('T2 Training Examples')
axes[0, 1].set_xlabel('Intensity')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
#Histogram of the whole image for T1 and T2
axes[1, 0].hist(t1_values, bins=50, alpha=0.5, label='T1')
axes[1, 0].hist(t2_values, bins=50, alpha=0.5, label='T2')
axes[1, 0].set_title('Whole Image Histograms')
axes[1, 0].set_xlabel('Intensity')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
plt.tight_layout()
plt.show()

#Q4: What is the difference between the 1D histogram of the training examples and the 1D histogram of the whole image? Is the difference expected?

# HIstograms looks relatively similar with better separation in the training examples. This is expected as the training examples are manually drawn to represent the two classes and thus should have better separation than the whole image which contains also background and other tissues.


#Exercise 4
# Make a training data vector (X) and target class vector (T) as input for the LDA() function. T and X should have the same length of data points.
# X: Training data vector should first include all data points for class 1 and then the data points for class 2. Data points are the two input features ImgT1, ImgT2
# T: Target class identifier for X where '0' are Class 1 and a '1' is Class 2.
X = np.vstack((C1, C2))
T = np.hstack((np.zeros(C1.shape[0]), np.ones(C2.shape[0])))

print("Shape of X:", X.shape)
print("Shape of T:", T.shape)   

#Exercise 5¶
# Make a scatter plot of the training points of the two input features for class 1 and class 2 as green and black circles, respectively. Add relevant title and labels to axis
# Q5: How does the class separation appear in the 2D scatter plot compared with 1D histogram. Is it better?
plt.figure(figsize=(8, 6))
plt.scatter(C1[:, 0], C1[:, 1], color='green', alpha=0.5, label='Class 1 (WM)')
plt.scatter(C2[:, 0], C2[:, 1], color='black', alpha=0.5, label='Class 2 (GM)')
plt.title('Scatter Plot of Training Points')
plt.xlabel('T1 Intensity')
plt.ylabel('T2 Intensity')
plt.legend()
plt.grid()
plt.show()

#Separation is much clearer in the scatter plot as we can see the two clusters representing the two classes. In the 1D histogram, the distributions of the two classes overlap more, making it harder to distinguish them. The 2D scatter plot provides a better visualization of the class separation.

#Exercise 6
# Train the linear discriminant classifier using the Fisher discriminant function and estimate the weight-vector coefficient W (i.e. 
# w for classification given X and T by using the W=LDA() function. The LDA function outputs W=[[w01, w1]; [w02, w2]] for class 1 and 2 respectively.

W = LDA(X, T)
print("Weight vector W:\n", W)
#Exercise 7¶
# Apply the linear discriminant classifier i.e. perform multi-modal classification using the trained weight-vector 
# W
# W for each class: It calculates the linear score 
# Y
# Y for all image data points within the brain slice i.e. 

Xall = np.c_[ImgT1[mask].ravel(), ImgT2[mask].ravel()]
Y = np.c_[np.ones(Xall.shape[0]), Xall] @ W.T

#Exercise 8
# Perform multi-modal classification: Calculate the posterior probability i.e. of a data point belonging to class 1

PosteriorProb = np.clip(np.exp(Y) / np.sum(np.exp(Y),1)[:, np.newaxis], 0, 1)

#Exercise 9¶
# Apply segmentation: Find all voxles in the T1w and T2w image with 
# P(C1 | X) > 0.5 as belonging to Class 1. You may use the np.where() function. Similarly, find all voxels belonging to class 2.

posteriorC1 = np.zeros(ImgT1.shape)
posteriorC2 = np.zeros(ImgT1.shape)

posteriorC1[mask] = PosteriorProb[:,0]
posteriorC2[mask] = PosteriorProb[:,1]
mask_WM = posteriorC1 >= 0.5
mask_GM = posteriorC2 > 0.5

fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (10, 5))
rgb = np.stack((posteriorC1, posteriorC2, np.zeros_like(posteriorC1)), axis = -1).astype(np.double)
axs[0].imshow(rgb)
axs[0].set_title('Posteriors (R-WM, G-GM)')
axs[1].imshow(mask_WM, cmap = 'gray', interpolation = 'none')
axs[1].set_title('WM Mask')
axs[2].imshow(mask_GM, cmap='gray', interpolation = 'none')
axs[2].set_title('GM Mask')
plt.show()

# Exercise 10
# Show scatter plot of segmentation results as in 5.

Xall_WM = Xall[PosteriorProb[:,0] > 0.5, :]
Xall_GM = Xall[PosteriorProb[:,1] > 0.5, :]

fig, ax = plt.subplots(1,1, figsize = (10,5))
ax.scatter(Xall_GM[:,0], Xall_GM[:,1], c = 'gray', label = 'GM')
ax.scatter(Xall_WM[:,0], Xall_WM[:,1], c = 'greenyellow', label = 'WM')
ax.set_xlabel('T1w intensities')
ax.set_ylabel('T2w intensities')
ax.set_title('Scatter plot of segmentation results')
plt.legend()
plt.show()

# Q6 Can you identify where the hyperplane is placed i.e. y(x)=0?
#Yes there is a clear line separating them 

# Q7 Is the linear hyper plane positioned as you expected or would a non-linear hyper plane perform better?

# Q8 Would segmentation be as good as using a single image modality using thresholding?
#In this case no because it is not orthogonal to the axes and thus a single modality would not be able to separate the two classes as well as the combination of the two modalities. The linear hyperplane is able to separate the two classes better than a single modality thresholding would do.
# Q9 From the scatter plot does the segmentation results make sense? Are the two tissue types segmented correctly.
#Potentially 

#Exercise 11¶
# Q10 Are the training examples representative for the segmentation results? Are you surprised that so few training examples perform so well? Do you need to be an anatomical expert to draw these?
# 
# Q11 Compare the segmentation results with the original image. Are the segmentation results satisfactory? Why not?
#Yes they do seem representative. It is surprising that so few training examples perform well, but it could be because the classes are well separated in the feature space. It may not be necessary to be an anatomical expert to draw these, as long as the training examples are representative of the classes.
# Q12 Is one class completely wrong segmented? What is the problem?
# They are properly segmented. 



tmp = np.zeros(mask_WM.shape)
red_WM = np.stack((mask_WM, tmp, tmp), axis = -1)
green_GM = np.stack((tmp, mask_GM, tmp), axis = -1)

def overlay_segmentation(I, M):
    I = I/I.max()
    I_aux, I_red = I.copy(), I.copy()
    I_aux[M] = 0
    I_red[M] = 1
    tmp = np.stack((I_red, I_aux, I_aux), axis = -1)
    return tmp

fig, axs = plt.subplots(2, 3, figsize = (15, 10))
axs[0,0].imshow(ImgT1, cmap = 'gray')
axs[0,0].set_title('T1w')
axs[0,1].imshow(overlay_segmentation(ImgT1, mask_WM))
axs[0,1].set_title('WM')
axs[0,2].imshow(overlay_segmentation(ImgT1, mask_GM))
axs[0,2].set_title('GM')

axs[1,0].imshow(ImgT2, cmap = 'gray')
axs[1,0].set_title('T2w')
axs[1,1].imshow(overlay_segmentation(ImgT2, mask_WM))
axs[1,1].set_title('WM')
axs[1,2].imshow(overlay_segmentation(ImgT2, mask_GM))
axs[1,2].set_title('GM')
plt.show()



