#

```
Learning OpenCV 3 Computer Vision with Python - Second Edition
Joseph Howse, Joe Minichino
September 28, 2015
https://www.packtpub.com/application-development/learning-opencv-3-computer-vision-python-second-edition
https://github.com/techfort/pycv
```

```
1Setting Up OpenCV
2Handling Files, Cameras, and GUIs
3Processing Images with OpenCV 3
4Depth Estimation and Segmentation
5Detecting and Recognizing Faces
6Retrieving Images and Searching Using Image Descriptors
7Detecting and Recognizing Objects
8Tracking Objects
9Neural Networks with OpenCV – an Introduction
```
# 3Processing Images with OpenCV 3
```
Converting between different color spaces
The Fourier Transform
Creating modules
Edge detection
Custom kernels – getting convoluted
Modifying the application
Edge detection with Canny
Contour detection
Contours – bounding box, minimum area rectangle, and minimum enclosing circle
Contours – convex contours and the Douglas-Peucker algorithm
Line and circle detection
Detecting shapes
```
###
```
!wget https://raw.githubusercontent.com/techfort/pycv/master/images/statue_small.jpg

import cv2
import numpy as np

img = cv2.imread("statue_small.jpg", 0)
cv2.imwrite("canny.jpg", cv2.Canny(img, 200, 300))
cv2.imshow("canny", cv2.imread("canny.jpg"))
cv2.waitKey()
cv2.destroyAllWindows()

```
