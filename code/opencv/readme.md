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
### Canny edge detection algorithm
```
waitKey()函數的功能是不斷刷新圖像，頻率時間為delay，單位為ms。
返回值為當前鍵盤按鍵值。
所以顯示圖像時，如果需要在imshow(“xxxx”,image)後吐舌頭加上while（cvWaitKey(n)==key）為大於等於0的數即可，
那麼程式將會停在顯示函數處，不運行其他代碼;直到鍵盤值為key的回應之後。

delay>0時，延遲”delay”ms，在顯示視頻時這個函數是有用的，
用於設置在顯示完一幀圖像後程式等待”delay”ms再顯示下一幀視頻；
如果使用waitKey(0)則只會顯示第一幀視頻。

返回值：如果delay>0,那麼超過指定時間則返回-1；如果delay=0，將沒有返回值。
　　如果程式想回應某個按鍵，可利用if(waitKey(1)==Keyvalue)；
如果delay<0,等待時間無限長，返回值為按鍵值

經常程式裡面出現if( waitKey(10) >= 0 ) 是說10ms中按任意鍵進入此if塊。

注意：這個函數是HighGUI中唯一能夠獲取和操作事件的函數，
所以在一般的事件處理中，它需要週期地被調用，
除非HighGUI被用在某些能夠處理事件的環境中。比如在MFC環境下，這個函數不起作用。
https://blog.csdn.net/Sunshine_in_Moon/article/details/45504563
```
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
```
cv2.waitKey 函數是用來等待與讀取使用者按下的按鍵，而其參數是等待時間（單位為毫秒），
若設定為 0 就表示持續等待至使用者按下按鍵為止，
這樣當我們按下任意按鍵之後，就會呼叫 cv2.destroyAllWindows 關閉所有 OpenCV 的視窗。

https://blog.gtwang.org/programming/opencv-basic-image-read-and-write-tutorial/
```
```
DisabledFunctionError                     Traceback (most recent call last)
<ipython-input-2-5780b98d45d0> in <module>()
      4 img = cv2.imread("statue_small.jpg", 0)
      5 cv2.imwrite("canny.jpg", cv2.Canny(img, 200, 300))
----> 6 cv2.imshow("canny", cv2.imread("canny.jpg"))
      7 cv2.waitKey()
      8 cv2.destroyAllWindows()

/usr/local/lib/python3.6/dist-packages/google/colab/_import_hooks/_cv2.py in wrapped(*args, **kwargs)
     50   def wrapped(*args, **kwargs):
     51     if not os.environ.get(env_var, False):
---> 52       raise DisabledFunctionError(message, name or func.__name__)
     53     return func(*args, **kwargs)
     54 

DisabledFunctionError: cv2.imshow() is disabled in Colab, because it causes Jupyter sessions
to crash; see https://github.com/jupyter/notebook/issues/3935.
As a substitution, consider using
  from google.colab.patches import cv2_imshow
```
```
from google.colab.patches import cv2_imshow

!curl -o logo.png https://colab.research.google.com/img/colab_favicon_256px.png
import cv2
img = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)
cv2_imshow(img)
```
```
img = cv2.imread('statue_small.jpg', cv2.IMREAD_UNCHANGED)
cv2_imshow(img)

# 執行Canny並寫入到canny.jpg
cv2.imwrite("canny.jpg", cv2.Canny(img, 200, 300))

#
img = cv2.imread('canny.jpg', cv2.IMREAD_UNCHANGED)
cv2_imshow(img)

```
