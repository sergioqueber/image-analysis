from skimage import io
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.decomposition import PCA
from skimage.transform import SimilarityTransform
from skimage.transform import warp
import os
import pathlib
import random

from data.Ex8_HelperFunctions import create_u_byte_image_from_vector, preprocess_all_cats, preprocess_one_cat

#Exercise 1: Preprocess all images in the training set. To do the preprocessing, you can use the code snippets supplied here. There is also a Model Cat supplied.

raw_dir = "data/training_cats/training_data/"
preprocessed_dir = "data/preprocessed_cats"
os.makedirs(preprocessed_dir, exist_ok=True)

# preprocess_all_cats(raw_dir, preprocessed_dir)

# The data matrix can be constructed by:
# Find the number of image files in the preprocessed folder using glob. Look at the preprocess_all_cats function to get an idea of how to use glob.
# Read the first photo and use that to find the height and width of the photos
# Set n_samples and n_features
# Make an empty matrix data_matrix = np.zeros((n_samples, n_features))
# Read the image files one by one and use flatten() to make each image into a 1-D vector (flat_img).
# Put the image vector (flat_img) into the data matrix by for example data_matrix[idx, :] = flat_img , where idx is the index of the current image.

# Exercise 2: Compute the data matrix.
files = glob.glob(f"{preprocessed_dir}/*.jpg")
n_samples = len(files)
first_img = io.imread(files[0])
height, width, channels = first_img.shape

n_features = height * width * channels
data_matrix = np.zeros((n_samples, n_features))
for idx, file in enumerate(files):
    img = io.imread(file)
    flat_img = img.flatten()
    data_matrix[idx, :] = flat_img

#Exercise 3: Compute the average cat.

# You can use the supplied function create_u_byte_image_from_vector to create an image from a 1-D image vector.

average_cat = np.mean(data_matrix, axis=0)

#Exercise 4: Visualize the Mean Cat
average_cat_img = create_u_byte_image_from_vector(average_cat, height, width, channels)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plt.imshow(average_cat_img)
ax.set_title("Mean cat")
ax.set_axis_off()
plt.show()

#Find a missing cat or a cat that looks like it (using image comparison)
# Exercise 5: Decide that you quickly buy a new cat that looks very much like the missing cat - so nobody notices.

#Exercise 6: Use the preprocess_one_cat function to preprocess the photo of the poor missing cat

# preprocess_one_cat()

# Exercise 7: Flatten the pixel values of the missing cat so it becomes a vector of values.
missing_cat_preprocessed  = io.imread("data/MissingCatProcessed.jpg")
missing_cat_vector = missing_cat_preprocessed.flatten()

# Exercise 8: Subtract the missing cat data from all the rows in the data_matrix and for each row compute the sum of squared differences.
diffs = data_matrix - missing_cat_vector
sum_squared_diffs = np.linalg.norm(diffs, axis=1)

#Exercise 9: Find the cat that looks most like your missing cat by finding the cat, where the SSD is smallest. You can for example use np.argmin.

closest_cat_idx = np.argmin(sum_squared_diffs)

#Exercise 10: Extract the found cat from the data_matrix and use create_u_byte_image_from_vector to create an image that can be visualized. Did you find a good replacement cat? Do you think your neighbour will notice? Even with their glasses on?

closest_cat_file = data_matrix[closest_cat_idx, :]
closest_cat_img = create_u_byte_image_from_vector(closest_cat_file, height, width, channels)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plt.imshow(closest_cat_img)
ax.set_title("Closest cat")
ax.set_axis_off()
plt.show()

#Exercise 11: You can use np.argmax to find the cat that looks the least like the missing cat.
least_like_cat_idx = np.argmax(sum_squared_diffs)
least_like_cat_file = data_matrix[least_like_cat_idx, :]
least_like_cat_img = create_u_byte_image_from_vector(least_like_cat_file, height, width, channels)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plt.imshow(least_like_cat_img)
ax.set_title("Least like cat")
ax.set_axis_off()
plt.show()


#Exercise 12: Start by computing the first 50 principal components:

print("Computing PCA...")
cats_pca = PCA(n_components=50)
cats_pca.fit(data_matrix)

#The amount of the total variation that each component explains can be found in cats_pca.explained_variance_ratio_.
#Exercise 13: Plot the amount of the total variation explained by each component as function of the component number.
plt.figure(figsize=(10, 6))
plt.plot(cats_pca.explained_variance_ratio_, marker='o')
plt.title('Explained Variance Ratio by Principal Component')
plt.xlabel('Principal Component Number')
plt.ylabel('Explained Variance Ratio')
plt.grid()
plt.show()

#Exercise 14: How much of the total variation is explained by the first component?

print("The first component explains {:.2f}% of the variance".format(cats_pca.explained_variance_ratio_[0] * 100))

#This was about 18%

#Exercise 15: Project the cat images into PCA space:

cats_pca_projection_components = cats_pca.transform(data_matrix)

#Now each cat has a position in PCA space. For each cat this position
#  is 50-dimensional vector. Each value in this vector describes how much 
# of that component is present in that cat photo.

#Exercise 16: Plot the PCA space by plotting all the cats first and second PCA coordinates in a (x, y) plot.

plt.figure(figsize=(10, 6))
plt.scatter(cats_pca_projection_components[:, 0], cats_pca_projection_components[:, 1], alpha=0.7)
plt.title('Cats in PCA Space')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

#Exercise 17: Use np.argmin and np.argmax to find the ids of the cats that have extreme
#  values in the first and second PCA coordinates. Extract the cats data from the data matrix and use create_u_byte_image_from_vector to visualize these cats. Also plot the PCA space where you plot the extreme cats with another marker and color.

def nargmax(arr, n):
    # Like np.argmax but returns the n largest values
    idx = np.argpartition(arr, -n)[-n:]
    return idx[np.argsort(arr[idx])][::-1]

def nargmin(arr, n):
    # Like np.argmin but returns the n smallest values
    idx = np.argpartition(arr, n)[:n]
    return idx[np.argsort(arr[idx])]

def plot_pca_space_and_img(pc_idx):
    _, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].plot(cats_pca_projection_components[:, 0], cats_pca_projection_components[:, 1], "o")
    ax[0].plot(cats_pca_projection_components[pc_idx, 0], cats_pca_projection_components[pc_idx, 1], "ro", markersize=10)
    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")

    img = create_u_byte_image_from_vector(data_matrix[pc_idx, :], height, width, channels)
    ax[1].imshow(img)
    ax[1].set_title("Cat")
    ax[1].set_axis_off()
    plt.show()

""" max_pc1_idx = nargmax(cats_pca_projection_components[:, 0], 4)
for idx in max_pc1_idx:
    plot_pca_space_and_img(idx)

min_pc1_idx = nargmin(cats_pca_projection_components[:, 0], 4)
for idx in min_pc1_idx:
    plot_pca_space_and_img(idx)

max_pc2_idx = nargmax(cats_pca_projection_components[:, 1], 4)
for idx in max_pc2_idx:
    plot_pca_space_and_img(idx)

min_pc2_idx = nargmin(cats_pca_projection_components[:, 1], 4)
for idx in min_pc2_idx:
    plot_pca_space_and_img(idx) """


#Exercise 18: How do these extreme cat photo look like? Are some actually of such bad quality that
#  they should be removed from the training set? If you remove images from the training set, then you
#  should run the PCA again. Do this until you are satisfied with the quality of the training data.

filtered_matrix = data_matrix.copy()

# Remove the 8 largest values of PC1, 5 smallest values of PC1,
# 5 largest values of PC2 and 5 smallest values of PC2
max_pc1s = nargmax(cats_pca_projection_components[:, 0], 8)
min_pc1s = nargmin(cats_pca_projection_components[:, 0], 5)
max_pc2s = nargmax(cats_pca_projection_components[:, 1], 5)
min_pc2s = nargmin(cats_pca_projection_components[:, 1], 5)

remove_idx = np.concatenate((max_pc1s, min_pc1s, max_pc2s, min_pc2s))
filtered_matrix = np.delete(filtered_matrix, remove_idx, axis=0)

# Recompute PCA
cats_pca = PCA(n_components=50)
cats_pca.fit(filtered_matrix)
cats_pca_projection_components = cats_pca.transform(filtered_matrix)

# Exercise 19: Create your first fake cat using the average image and the first
#  principal component. You should choose experiment with different weight values (w) 
for w in [-30000, -20000, -10000, 500, 100000, 200000, 300000]:
    synth_cat = average_cat + w * cats_pca.components_[0, :]
    synth_cat_img = create_u_byte_image_from_vector(synth_cat, height, width, channels)
    plt.figure(figsize=(6, 6))
    plt.imshow(synth_cat_img)
    plt.title(f"Synthetic cat with w={w}")
    plt.axis('off')
    plt.show()

# Exercise 21: Synthesize some cats, where you use both the first and second principal 
# components and select their individual weights based on the PCA plot.

""" for w1 in [-30000, -20000, -10000, 500, 100000, 200000, 300000]:
    for w2 in [-30000, -20000, -10000, 500, 100000, 200000, 300000]:
        synth_cat = average_cat + w1 * cats_pca.components_[0, :] + w2 * cats_pca.components_[1, :]
        synth_cat_img = create_u_byte_image_from_vector(synth_cat, height, width, channels)
        plt.figure(figsize=(6, 6))
        plt.imshow(synth_cat_img)
        plt.title(f"Synthetic cat with w1={w1}, w2={w2}")
        plt.axis('off')
        plt.show() """


# Exercise 22: Synthesize and visualize cats that demonstrate the first three major
#  modes of variation. Try show the average cat in the middle of a plot, with the negative
#  sample to the left and the positive to the right. Can you recognise some visual patterns in these modes of variation?

# Exercise 23
n_components_to_use = 10
synth_cat = average_cat
for idx in range(n_components_to_use):
    w = random.uniform(-1, 1) * 3 * np.sqrt(cats_pca.explained_variance_[idx])
    synth_cat = synth_cat + w * cats_pca.components_[idx, :]
    synth_cat_img = create_u_byte_image_from_vector(synth_cat, height, width, channels)
    plt.figure(figsize=(6, 6))
    plt.imshow(synth_cat_img)
    plt.title(f"Synthetic cat with {idx+1} components")
    plt.axis('off')
    plt.show()

#Exercise 24: Start by finding the PCA space coordinates of your missing cat:

missing_cat_pca_coordinates = cats_pca.transform(missing_cat_vector.reshape(1, -1))
pca_coor = missing_cat_pca_coordinates.flatten()

#Exercise 25: Plot all the cats in PCA space using the first two dimensions. Plot your missing
#  cat in the same plot, with another color and marker. Is it placed somewhere sensible and does it have close neighbours?

plt.figure(figsize=(10, 6))
plt.scatter(cats_pca_projection_components[:, 0], cats_pca_projection_components[:, 1], alpha=0.7, label='Cats')
plt.scatter(pca_coor[0], pca_coor[1], color='red', marker='X', s=100, label='Missing Cat')
plt.title('Cats in PCA Space with Missing Cat')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.show()

#Exercise 26: Generate synthetic versions of your cat, where you change the n_components_to_use from 1 to for example 50.

n_components_to_use = 50
synth_cat = average_cat
for idx in range(n_components_to_use):
    synth_cat = synth_cat + pca_coor[idx] * cats_pca.components_[idx, :]

fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
ax[0].imshow(create_u_byte_image_from_vector(missing_cat_vector, height, width, channels))
ax[0].set_title("Missing cat")
ax[0].set_axis_off()
ax[1].imshow(create_u_byte_image_from_vector(synth_cat, height, width, channels))
ax[1].set_title(f"Synth cat with {n_components_to_use} components")
ax[1].set_axis_off()
plt.show()

distance_from_synth_to_rest = np.linalg.norm(cats_pca_projection_components - pca_coor, axis=1)

#Exercise 27: Find the id of the cat that has the smallest and largest
#  distance in PCA space to your missing cat. Visualize these cats.
#  Are they as you expected? Do you think your neighours will notice a difference?

best_match_idx = np.argmin(distance_from_synth_to_rest)
best_twin_cat = data_matrix[best_match_idx, :]
best_twin_cat_img = create_u_byte_image_from_vector(best_twin_cat, height, width, channels)

worst_match_idx = np.argmax(distance_from_synth_to_rest)
worst_twin_cat = data_matrix[worst_match_idx, :]
worst_twin_cat_img = create_u_byte_image_from_vector(worst_twin_cat, height, width, channels)

fig, ax = plt.subplots(ncols=3, figsize=(18, 6))
ax[0].imshow(create_u_byte_image_from_vector(missing_cat_vector, height, width, channels))
ax[0].set_title("Missing cat")
ax[0].set_axis_off()
ax[1].imshow(best_twin_cat_img)
ax[1].set_title("Best match cat")
ax[1].set_axis_off()
ax[2].imshow(worst_twin_cat_img)
ax[2].set_title("Worst match cat")
ax[2].set_axis_off()
plt.show()

# Exercise 28: Find the ids of and visualize the 5 closest cats in PCA space. Do they look like your cat?

n_best = 5
best = np.argpartition(distance_from_synth_to_rest, n_best)[:n_best]
fig, ax = plt.subplots(ncols=n_best+1, figsize=(3*(n_best+1), 6))
ax[0].imshow(create_u_byte_image_from_vector(missing_cat_vector, height, width, channels))
ax[0].set_title("Missing cat")
ax[0].set_axis_off()
for idx in range(n_best):
    best_cat = data_matrix[best[idx], :]
    best_cat_img = create_u_byte_image_from_vector(best_cat, height, width, channels)
    ax[idx+1].imshow(best_cat_img)
    ax[idx+1].set_title(f"Best match {idx+1}")
    ax[idx+1].set_axis_off()
plt.show()


