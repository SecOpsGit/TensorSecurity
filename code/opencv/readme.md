#

```
Learning OpenCV 3 Computer Vision with Python - Second Edition
Joseph Howse, Joe Minichino
September 28, 2015
https://www.packtpub.com/application-development/learning-opencv-3-computer-vision-python-second-edition
https://github.com/techfort/pycv

OpenCV3計算機視覺：Python語言實現(原書第2版)
　作　　者：	(愛爾蘭)喬．米尼奇諾
　出版單位：	機械工業
　ＩＳＢＮ：	9787111539759
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
```
第1章安裝OpenCV1
1.1選擇和使用合適的安裝工具2
1.1.1在Windows上安裝2
1.1.2在OS X系統中安裝6
1.1.3在Ubuntu及其衍生版本中安裝11
1.1.4在其他類Unix系統中安裝12
1.2安裝Contrib模塊13
1.3運行示例13
1.4查找文檔、幫助及更新14

第2章處理文件、攝像頭和圖形用戶界面16
2.1基本I／O腳本16
2.1.1讀／寫圖像文件16
2.1.2圖像與原始字節之間的轉換19
2.1.3使用numpy.array訪問圖像數據20
2.1.4視頻文件的讀／寫22
2.1.5捕獲攝像頭的幀23
2.1.6在窗口顯示圖像24
2.1.7在窗口顯示攝像頭幀25
2.2Cameo項目（人臉跟蹤和圖像處理）26
2.3Cameo—面向對象的設計27
2.3.1使用managers.CaptureManager提取視頻流27
2.3.2使用managers.WindowManager抽象窗口和鍵盤32
2.3.3cameo.Cameo的強大實現33

第3章使用OpenCV 3處理圖像36
3.1不同色彩空間的轉換36
3.2傅裡葉變換37
3.2.1高通濾波器37
3.2.2低通濾波器39
3.3創建模塊39
3.4邊緣檢測40
3.5用定制內核做卷積41
3.6修改應用43
3.7Canny邊緣檢測44
3.8輪廓檢測45
3.9邊界框、最小矩形區域和最小閉圓的輪廓46
3.10凸輪廓與Douglas—Peucker算法48
3.11直線和圓檢測50
3.11.1直線檢測50
3.11.2圓檢測51
3.12檢測其他形狀52

第4章深度估計與分割53
4.1創建模塊53
4.2捕獲深度攝像頭的幀54
4.3從視差圖得到掩模56
4.4對複製操作執行掩模57
4.5使用普通攝像頭進行深度估計59
4.6使用分水嶺和GrabCut算法進行物體分割63
4.6.1用GrabCut進行前景檢測的例子64
4.6.2使用分水嶺算法進行圖像分割66

第5章人臉檢測和識別70
5.1Haar級聯的概念70
5.2獲取Haar級聯數據71
5.3使用OpenCV進行人臉檢測72
5.3.1靜態圖像中的人臉檢測72
5.3.2視頻中的人臉檢測74
5.3.3人臉識別76

第6章圖像檢索以及基於圖像描述符的搜索83
6.1特徵檢測算法83
6.1.1特徵定義84
6.1.2使用DoG和SIFT進行特徵提取與描述86
6.1.3使用快速Hessian算法和SURF來提取和檢測特徵89
6.1.4基於ORB的特徵檢測和特徵匹配91
6.1.5ORB特徵匹配93
6.1.6K—最近鄰匹配95
6.1.7FLANN匹配96
6.1.8FLANN的單應性匹配99
6.1.9基於文身取證的應用程序示例102

第7章目標檢測與識別106
7.1目標檢測與識別技術106
7.1.1HOG描述符107
7.1.2檢測人112
7.1.3創建和訓練目標檢測器113
7.2汽車檢測116
7.2.1代碼的功能118
7.2.2SVM和滑動窗口122

第8章目標跟蹤135
8.1檢測移動的目標135
8.2背景分割器：KNN、MOG2和GMG138
8.2.1均值漂移和CAMShift142
8.2.2彩色直方圖144
8.2.3返回代碼146
8.3CAMShift147
8.4卡爾曼濾波器149
8.4.1預測和更新149
8.4.2範例150
8.4.3一個基於行人跟蹤的例子153
8.4.4Pedestrian類154
8.4.5主程序157

第9章基於OpenCV的神經網絡簡介160
9.1人工神經網絡160
9.2人工神經網絡的結構161
9.2.1網絡層級示例162
9.2.2學習算法163
9.3OpenCV中的ANN164
9.3.1基於ANN的動物分類166
9.3.2訓練週期169
9.4用人工神經網絡進行手寫數字識別170
9.4.1MNIST—手寫數字數據庫170
9.4.2定制訓練數據170
9.4.3初始參數171
9.4.4迭代次數171
9.4.5其他參數171
9.4.6迷你庫172
9.4.7主文件175
9.5可能的改進和潛在的應用180
9.5.1改進180
9.5.2應用181

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
