import cv2
import os 
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

feat_dir = 'data/haar_features'
img_path = os.path.join(feat_dir, 'stage_3.png')
stage_img = io.imread(img_path)
ref_path = 'data/annotation_img.jpg'
ref_img = io.imread(ref_path)
ref_img = img_as_ubyte(rgb2gray(ref_img)) # Want the image to be grayscaled


sliced_img = ref_img[14-1:17+2,4-1:7+2]

# Integral image options
# integral_img = cv2.integral(sliced_img)[1:,1:] # Pads boundary - hence we may remove the first row of y,x respectfully
integral_img = np.cumsum(np.cumsum(sliced_img, axis=0), axis=1)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(sliced_img, cmap='gray')
axes[0].set_title('Real image slice')
axes[0].set_aspect('equal')
img = axes[1].imshow(integral_img, cmap='viridis')
axes[1].set_title('Cumulative sum image of image slice')
axes[1].set_aspect('equal')

# Write numbers on each pixel of cumsum image to gain extra insights
for (i, j), val in np.ndenumerate(integral_img):
    axes[1].text(j, i, f"{val}", ha='center', va='center', color='white', fontsize=10)

plt.tight_layout()
fig.colorbar(img, label='Cumulative pixel value', ax=axes, shrink=0.38)
plt.show()

# Retrieve w, h of the Haar regions
w, h = integral_img.shape[1], integral_img.shape[0]
w_rec, h_rec = w//2, h//2

# As regions are standard uniform, we may retrieve start positions of each region like:
start_positions = np.array([
    [(y, x) for x in np.linspace(start=1, stop=w_rec, num=2)]
    for y in np.linspace(start=1, stop=h_rec, num=2)]
).reshape(-1, 2)

print(start_positions)

def compute_integral_image(integral_img, y0, x0, w, h):
    """
        Computes the integral image within the bounding box represented by (y0,x0,w,h).

        :param numpy.ndarray integral_img:      The integral image input. Dimensions need to supercede those of the bounding box.
        :param int y0:                          Initial starting y position of bounding box.
        :param int x0:                          Initial starting x position of bounding box.
        :param int w:                           Width of bounding box.
        :param int h:                           Height of bounding box.
    """

    # Retrieve corners of bounding box
    pos1, pos2, pos3, pos4 = (y0-1, x0-1), (y0-1, x0-1+w-1),(y0-1+h-1, x0-1), (y0-1+h-1, x0-1+w-1)

    # Compute integral image
    A = integral_img[pos1[0], pos1[1]]
    B = integral_img[pos2[0], pos2[1]]
    C = integral_img[pos3[0], pos3[1]]
    D = integral_img[pos4[0], pos4[1]]
    return D + A - B - C

def compute_diagonal_feat(integral_img, start_positions, w_reg, h_reg):
    """
        Computes the diagonal Haar feature of a 
    """

    # Compute integral image for each square region
    regions = []
    for y,x in start_positions:
        reg_integral = compute_integral_image(integral_img, int(y), int(x), w_reg, h_reg)
        regions.append(reg_integral)

    # Compute Haar feature
    white_pixels = regions[1] + regions[2]
    black_pixels = regions[0] + regions[3]
    haar_feat =  white_pixels - black_pixels

    return haar_feat

haar_feat = compute_diagonal_feat(integral_img, start_positions, w_rec, h_rec)
print(f"The computed Haar feature of image slice = {haar_feat}")
