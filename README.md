# Image Analysis

A collection of Python exercises covering fundamental image processing and computer vision concepts, from basic image manipulation to real-time video analysis.

## Exercises

### Exercise 1 – Introduction to Image Analysis
An introduction to image analysis fundamentals using Python.

**Topics covered:**
- Reading and displaying grayscale and color images
- Histogram analysis and visualization
- Pixel masking and thresholding
- Image rescaling and resizing
- RGB channel decomposition
- Drawing on images (rectangles, regions of interest)
- 3D surface plotting from image intensity data
- Reading DICOM medical images

### Exercise 1b – Statistical Analysis & PCA
Statistical analysis and dimensionality reduction applied to image data.

**Topics covered:**
- Computing feature variance and covariance
- Principal Component Analysis (PCA) via eigenvalue decomposition
- Scree plot visualization
- Comparing manual vs. scikit-learn PCA implementations
- Data projection into PCA space

### Exercise 2 – Camera Geometry
Mathematical foundations of camera optics and imaging.

**Topics covered:**
- Lens equation and focal length calculation
- CCD distance and pixel dimension computation
- Object-to-image magnification
- Horizontal and vertical field-of-view angle calculation

### Exercise 2b – Real-Time Video Processing
Change detection and motion analysis from a live camera feed.

**Topics covered:**
- Capturing and displaying frames from a webcam
- Grayscale conversion and temporal image differencing
- Foreground pixel detection via thresholding
- Motion alarm generation
- Real-time FPS calculation and frame annotation
- Optional DroidCam mobile camera support

## Requirements

- Python 3
- [scikit-image](https://scikit-image.org/)
- [NumPy](https://numpy.org/)
- [OpenCV](https://opencv.org/) (`cv2`)
- [Matplotlib](https://matplotlib.org/)
- [pandas](https://pandas.pydata.org/)
- [seaborn](https://seaborn.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pydicom](https://pydicom.github.io/)

Install all dependencies with:

```bash
pip install scikit-image numpy opencv-python matplotlib pandas seaborn scikit-learn pydicom
```

## Usage

Run each exercise script from inside its directory:

```bash
cd Exercises/Exercise1
python Exercise1.py

cd Exercises/Exercise1b
python Exercise1b.py

cd Exercises/Exercise2
python Exercise2.py

cd Exercises/Exercise2b
python exercise2b.py
```
