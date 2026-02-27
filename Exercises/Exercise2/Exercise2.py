#Exercise 1

import math

a = 10 
b = 3 

theta = math.atan2(a, b)  # This function computes the arctangent of a/b, taking into account the signs of both arguments to determine the correct quadrant of the result. The result is returned in radians.
theta_degrees = math.degrees(theta)  # Convert radians to degrees

print('The angle theta in radians is: %f' % theta)
print('The angle theta in degrees is: %f' % theta_degrees)

# Exercise 2

def camera_b_distance(f, g):
    """
    camera_b_distance returns the distance (b) where the CCD should be placed
    when the object distance (g) and the focal length (f) are given
    :param f: Focal length
    :param g: Object distance
    :return: b, the distance where the CCD should be placed
    """
    b = (f * g) / (g - f)  # This formula is derived from the lens equation: 1/f = 1/g + 1/b
    return b

f = 15 
g = [100, 1000, 5000 , 15000] #Mm

for obj_dist in g:
    b = camera_b_distance(f, obj_dist)
    print('For object distance g = %f, the CCD should be placed at distance b = %f' % (obj_dist, b))

#Exercise 3 Thomas is 1.8 meters tall and standing 5 meters from a camera. The cameras focal length is 5 mm. The CCD in the camera can be seen in the figure below. It is a 1/2" (inches) CCD chip and the image formed by the CCD is 640x480 pixels in a (x,y) coordinate system.

# Exercise 3.1: A focused image of Thomas is formed inside the camera. At which distance from the lens?¶

f = 5  # Focal length in mm
g = 5000  # Object distance in mm (5 meters converted to millimeters)
G = 1800  # Height of Thomas in mm (1.8 meters converted to millimeters)

b = camera_b_distance(f, g)  # Calculate the distance where the CCD should be placed
print('The distance from the lens to the CCD (b) is: %f mm' % b)

# Exercise 3.2: How tall (in mm) will Thomas be on the CCD-chip?¶

B = (b / g) * G  # Calculate the height of Thomas on the CCD using similar triangles (B/b = 1800/g)
print('The height of Thomas on the CCD-chip is: %f mm' % B)

# Exercise 3.3: What is the size of a single pixel on the CCD chip? (in mm)?¶

chip_width = 6.4 #mm
chip_height = 4.8 #mm
pixel_width = chip_width / 640  # Calculate the width of a single pixel
pixel_height = chip_height / 480  # Calculate the height of a single pixel

print('The size of a single pixel on the CCD chip is: %f mm (width) x %f mm (height)' % (pixel_width, pixel_height))

# Exercise 3.4: How tall (in pixels) will Thomas be on the CCD-chip?¶

thomas_height_pixels = B / pixel_height  # Calculate the height of Thomas in pixels
print('The height of Thomas on the CCD-chip is: %f pixels' % thomas_height_pixels)

# Exercise 3.5: What is the horizontal field-of-view (in degrees)?¶

fov_horizontal = 2 * math.atan((chip_width / 2) / b)  # Calculate the horizontal field of view in radians
fov_horizontal_degrees = 180 / math.pi * fov_horizontal  # Convert to degrees
print('The horizontal field-of-view is: %f degrees' % fov_horizontal_degrees)

FOV_x = 2*math.atan2(3.2e-3, b)*180/math.pi
print('The horizontal field-of-view is: %f degrees' % FOV_x)
      
# Exercise 3.6: What is the vertical field-of-view (in degrees)?¶

fov_vertical = 2 * math.atan((chip_height / 2) / b)  # Calculate the vertical field of view in radians
fov_vertical_degrees = 180 / math.pi * fov_vertical  # Convert to

print('The vertical field-of-view is: %f degrees' % fov_vertical_degrees)
FOV_y = 2*math.atan2(2.4e-3, b)*180/math.pi
print('The vertical field-of-view is: %f degrees' % FOV_y)



