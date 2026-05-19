import matplotlib.pyplot as plt
import math
from skimage.transform import rotate
from skimage.transform import EuclideanTransform
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.transform import swirl
from skimage import io

def show_comparison(original, transformed, transformed_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(transformed)
    ax2.set_title(transformed_name)
    ax2.axis('off')
    io.show()

# Exercise 1 Read the NusaPenida.png image

im_org = io.imread('data/NusaPenida.png')
rotation_angle = 10
rotated_img = rotate(im_org, rotation_angle)
show_comparison(im_org, rotated_img, f'Rotated by {rotation_angle} degrees')

rot_center = [0, 0]
rotated_img = rotate(im_org, rotation_angle, center=rot_center)
show_comparison(im_org, rotated_img, f'Rotated by {rotation_angle} degrees around {rot_center}')

#Exercise 2¶
# Experiment with different center points and notice the results.
# As seen, there are areas of the rotated image that is filled with a background value. It can be controlled how this background filling shall behave.

rotated_img = rotate(im_org, rotation_angle, center=rot_center, mode='reflect')
show_comparison(im_org, rotated_img, f'Rotated by {rotation_angle} degrees around {rot_center} with reflect mode')

rotated_img = rotate(im_org, rotation_angle, center=rot_center, mode='constant', cval=255)
show_comparison(im_org, rotated_img, f'Rotated by {rotation_angle} degrees around {rot_center} with constant mode and cval=255')

# Exercise 3
rotated_img = rotate(im_org, rotation_angle, center=rot_center, mode='wrap')
show_comparison(im_org, rotated_img, f'Rotated by {rotation_angle} degrees around {rot_center} with wrap mode')

# Exercise 4

rotated_img =rotate(im_org, rotation_angle, resize=True, mode="constant", cval=1)
show_comparison(im_org, rotated_img, f'Rotated by {rotation_angle} degrees with resize=True and cval=1')

rotated_img =rotate(im_org, rotation_angle, resize=True, mode="constant", cval=0)
show_comparison(im_org, rotated_img, f'Rotated by {rotation_angle} degrees with resize=True and cval=0')

rotated_img =rotate(im_org, rotation_angle, resize=True, mode="constant", cval=255)
show_comparison(im_org, rotated_img, f'Rotated by {rotation_angle} degrees with resize=True and cval=255')

# Exercise 5¶
# Test the use of automatic resizing:

rotated_img =rotate(im_org, rotation_angle, resize=True)
show_comparison(im_org, rotated_img, f'Rotated by {rotation_angle} degrees with resize=True')

rotated_img =rotate(im_org, rotation_angle, resize=True, mode = "reflect")
show_comparison(im_org, rotated_img, f'Rotated by {rotation_angle} degrees with resize=True and reflect mode')

rotated_img =rotate(im_org, rotation_angle, resize=True, mode = "wrap")
show_comparison(im_org, rotated_img, f'Rotated by {rotation_angle} degrees with resize=True and wrap mode')


#Exercise  6
# Start by defining the transformation:

# angle in radians - counter clockwise
rotation_angle = 10.0 * math.pi / 180.
trans = [10, 20]
tform = EuclideanTransform(rotation=rotation_angle, translation=trans)
print(tform.params)

# Exercise 7
transformed_img = warp(im_org, tform)
show_comparison(im_org, transformed_img, f'Rotated by {rotation_angle} radians and translated by {trans} using EuclideanTransform') 

#The warp function actually does an inverse transformation of the image, since it uses the transform to find the pixels values in the input image that should be placed in the output image.

transformed_img = warp(im_org, tform.inverse)
show_comparison(im_org, transformed_img, f'Rotated by {rotation_angle} radians and translated by {trans} using the inverse of the EuclideanTransform')

# Exercise 8¶
# Construct a Euclidean transformation with only rotation. Test the transformation and the invers transformation and notice the effect.
rotation_angle = 10.0 * math.pi / 180.
trans = [0, 0]
tform = EuclideanTransform(rotation=rotation_angle, translation=trans)
transformed_img = warp(im_org, tform.inverse)

show_comparison(im_org, transformed_img, f'Rotated by {rotation_angle} radians using EuclideanTransform')

#The SimilarityTransform computes a transformation consisting of a translation, rotation and a scaling.

#Exercise 9
# Define a SimilarityTransform with an angle of 15 , a translation of (40, 30) and a scaling of 0.6 and test it on the image.
rotation_angle = 15.0 * math.pi / 180.
trans = [40, 30]
scale = 0.6
tform = SimilarityTransform(rotation=rotation_angle, translation=trans, scale=scale)
transformed_img = warp(im_org, tform.inverse)
show_comparison(im_org, transformed_img, f'Rotated by {rotation_angle} radians, translated by {trans} and scaled by {scale} using SimilarityTransform') 


