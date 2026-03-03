import time
import cv2
import numpy as np
from skimage.util import img_as_float
from skimage.util import img_as_ubyte

def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name,img)

def capture_from_camera_and_show_images():
    print("Starting image capture")

    print("Opening connection to camera")
    url = 0
    use_droid_cam = False
    if use_droid_cam:
        url = "http://192.168.1.120:4747/video"
    cap = cv2.VideoCapture(url)
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Starting camera loop")
    # Get first image
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame")
        exit()

    # Transform image to gray scale and then to float, so we can do some processing
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = img_as_float(frame_gray)

    # To keep track of frames per second
    start_time = time.time()
    n_frames = 0
    stop = False
    while not stop:
        ret, new_frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Transform image to gray scale and then to float, so we can do some processing
        new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        new_frame_gray = img_as_float(new_frame_gray)

        # Compute difference image
        dif_img = np.abs(new_frame_gray - frame_gray)

        # Computes the total number of foreground, F, pixels in the foreground image.
        T=0.1  # Threshold for foreground pixel detection
        n_foreground_pixels = np.sum(dif_img > T)  # Count pixels with a difference greater than a threshold
        print(f"Number of foreground pixels: {n_foreground_pixels}")

        #Compute the percentage of foreground pixels compared to the total number of pixels in the image (F).

        total_pixels = dif_img.size
        F = (n_foreground_pixels / total_pixels) * 100
        print(f"Percentage of foreground pixels: {F:.2f}%")

        #Decides if an alarm should be raised if F is larger than an alert value, A.

        A = 0.5  # Alert threshold in percentage
        if F > A:
            print("Alarm: Foreground pixel percentage exceeds alert threshold!")    

        # If an alarm is raised, show a text on the input image. For example Change Detected!

        if F > A:
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(new_frame, "Change Detected!", (100, 150), font, 1, (0, 0, 255), 2)

        # Keep track of frames-per-second (FPS)
        n_frames = n_frames + 1
        elapsed_time = time.time() - start_time
        fps = int(n_frames / elapsed_time)

        # Put the FPS on the new_frame
        str_out = f"fps: {fps}"
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(new_frame, str_out, (100, 100), font, 1, 255, 1)

        #Exercise 6: Use putText to write some important information on the image. For example the number of changed pixel, the average, minumum and maximum value in the difference image. These values can then be used to find even better values for T and A. You can also write the text in different colors depending on the values. For example, if F is larger than A, you can write the text in red color.
        avg_diff = np.mean(dif_img)
        min_diff = np.min(dif_img)
        max_diff = np.max(dif_img)
        str_info = f"Changed pixels: {n_foreground_pixels}, Avg: {avg_diff:.2f}, Min: {min_diff:.2f}, Max: {max_diff:.2f}"
        cv2.putText(new_frame, str_info, (100, 200), font, 0.5, (255, 255, 255), 1)

        #Shows the input image, the backround image, the difference image, and the binary image. The binary image should be converted to uint8 using img_as_ubyte.

        binary_img = (dif_img > 0.1).astype(np.uint8) * 255
        input_image = img_as_ubyte(new_frame)
        dif_img = img_as_ubyte(dif_img)
        background_image = img_as_ubyte(frame_gray)

        # Display the resulting frame
        show_in_moved_window('Input', input_image, 0, 10)
        show_in_moved_window('Input gray', background_image, 600, 10)
        show_in_moved_window('Difference image', dif_img, 1200, 10)
        show_in_moved_window('Binary image', binary_img, 0, 400)

        
        # Update background image using exponential moving average
        alpha = 0.5  # Learning rate for background update
        frame_gray = alpha * frame_gray + (1 - alpha) * new_frame_gray

        if cv2.waitKey(1) == ord('q'):
            stop = True

    print("Stopping image loop")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_from_camera_and_show_images()

    #Exercise 4: Try to change α T and A What effects do it have?
    # The parameter α (alpha) controls the learning rate for updating the background model. A higher alpha means that the background model will adapt more quickly to changes in the scene, while a lower alpha will make it more stable and less responsive to changes. Adjusting alpha can help balance between sensitivity to changes and stability of the background model.

#The images are displayed using the OpenCV function imshow. The display window has several ways of zooming in the displayed image.

#Exercise 5: Try to play around with the zoom window.



