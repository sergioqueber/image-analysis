from skimage import color
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
from skimage.filters import threshold_otsu
import time
import cv2
import numpy as np

# Exercise 17: Change the gray-scale processing in the exercise material script to be for 
# example thresholding, gamma mapping or something else. Do you get the visual result that you expected?

# Exercise 18: Real time detection of DTU signs
# Change the rgb-scale processing in the exercise material 
# script so it does a color threshold in either RGB or HSV space. 
# The goal is to make a program that can see DTU street signs. 
# The output should be a binary image, where the pixels of the sign 
# is foreground. Later in the course, we will learn how to remove the 
# noise pixels.


def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name, img)


def process_gray_image(img):
    """
    Do a simple processing of an input gray scale image and return the processed image.
    # https://scikit-image.org/docs/stable/user_guide/data_types.html#image-processing-pipeline
    """
    img_float = img_as_float(img)
    img_proc = 1 - img_float
    return img_as_ubyte(img_proc)

def process_gray_image_thresholding_and_gamma(img):
    """
    Do a simple processing of an input gray scale image and return the processed image.
    # https://scikit-image.org/docs/stable/user_guide/data_types.html#image-processing-pipeline
    """
    img_float = img_as_float(img)
    # Apply gamma correction to the thresholded image
    gamma = 2.0  # Gamma value for correction
    img_gamma = np.power(img_float, gamma)
    T = threshold_otsu(img_gamma)  # Threshold for foreground pixel detection
    mask = img_gamma > T
    return img_as_ubyte(mask)


def process_rgb_image(img):
    """
    Simple processing of a color (RGB) image
    """
    # Copy the image information so we do not change the original image
    # Exercise 18: Real time detection of DTU signs
    # Change the rgb-scale processing in the exercise material 
    # script so it does a color threshold in either RGB or HSV space. 
    # The goal is to make a program that can see DTU street signs. 
    # The output should be a binary image, where the pixels of the sign 
    # is foreground. Later in the course, we will learn how to remove the 
    # noise pixels.

    proc_img = img.copy()
    r_comp = proc_img[:, :, 0]
    b_comp = proc_img[:, :, 2]
    g_comp = proc_img[:, :, 1]

    segm_red = (r_comp > 160) & (r_comp < 180) & (g_comp > 50) & (g_comp < 80) & \
            (b_comp > 50) & (b_comp < 80)
    
    return img_as_ubyte(segm_red)

def capture_from_camera_and_show_images():
    print("Starting image capture")

    print("Opening connection to camera")
    url = 0
    use_droid_cam = False
    if use_droid_cam:
        url = "http://192.168.1.120:4747/video"
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Starting camera loop")
    # To keep track of frames per second using a high-performance counter
    old_time = time.perf_counter()
    fps = 0
    stop = False
    process_rgb = True
    while not stop:
        ret, new_frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Change from OpenCV BGR to scikit image RGB
        new_image = new_frame[:, :, ::-1]
        new_image_gray = color.rgb2gray(new_image)
        if process_rgb:
            proc_img = process_rgb_image(new_image)
        else:
            proc_img = process_gray_image_thresholding_and_gamma(new_image_gray)

        # update FPS - but do it slowly to avoid fast changing number
        new_time = time.perf_counter()
        time_dif = new_time - old_time
        old_time = new_time
        fps = fps * 0.95 + 0.05 * 1 / time_dif

        # Put the FPS on the new_frame
        str_out = f"fps: {int(fps)}"
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(new_frame, str_out, (100, 100), font, 1, 255, 1)

        # Display the resulting frame
        show_in_moved_window('Input', new_frame, 0, 10)
        show_in_moved_window('Input gray', new_image_gray, 600, 10)
        show_in_moved_window('Processed image', proc_img, 1200, 10)

        if cv2.waitKey(1) == ord('q'):
            stop = True

    print("Stopping image loop")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_from_camera_and_show_images()

