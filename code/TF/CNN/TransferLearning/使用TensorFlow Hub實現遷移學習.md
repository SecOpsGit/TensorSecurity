# TensorFlow Hub
```
TensorFlow Hub is a library for the publication, discovery, and consumption of 
reusable parts of machine learning models. 

A module is a self-contained piece of a TensorFlow graph, 
along with its weights and assets, that can be reused across different tasks 
in a process known as transfer learning. 

Transfer learning can:
Train a model with a smaller dataset,
Improve generalization, and
Speed up training.

TensorFlow Hub是一種共用預訓練模型元件的方法。

TensorFlow Hub是一個用於促進機器學習模型的可重用部分的發佈，探索和使用的庫。
特別是，它提供經過預先訓練的TensorFlow模型，可以在新任務中重複使用。
（可以理解為做遷移學習：可以使用較小的資料集訓練模型，可以改善泛化和加快訓練。）

GitHub 地址：https://github.com/tensorflow/hub

有關預先訓練模型的可搜索清單，請參閱TensorFlow模組中心TensorFlow Module Hub
```

# 使用TensorFlow Hub實現遷移學習(TF2.0官方教程)
```
最新版本：https://www.mashangxue123.com/tensorflow/tf2-tutorials-images-hub_with_keras.html
英文版本：https://tensorflow.google.cn/beta/tutorials/images/hub_with_keras
翻譯建議PR：https://github.com/mashangxue/tensorflow2-zh/edit/master/r2/tutorials/images/hub_with_keras.md

基於Keras使用TensorFlow Hub實現遷移學習(tensorflow2.0官方教程翻譯)
https://zhuanlan.zhihu.com/p/68061929

本教程示範：
如何在tf.keras中使用TensorFlow Hub。
如何使用TensorFlow Hub進行圖像分類。
如何做簡單的遷移學習。
```

## 安裝tensorflow_hub
```
安裝命令：pip install -U tensorflow_hub
```
### 導入模組與套件
```
from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
``` 

### 2.使用根據ImageNet資料集所建置的mobilenet分類器

### 下載分類器
```
使用hub.module載入mobilenet，並使用tf.keras.layers.Lambda將其包裝為keras層。
來自tfhub.dev的任何相容tf2的圖像分類器URL都可以在這裡工作。

classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"  
#@param {type:"string"}

IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])
```
### 測試看看:在單個圖像上運行它
```
下載單個圖像以試用該模型。

import numpy as np
import PIL.Image as Image

grace_hopper = tf.keras.utils.get_file('image.jpg',
'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
grace_hopper = np.array(grace_hopper)/255.0
grace_hopper.shape

添加批量維度，並將圖像傳遞給模型。

result = classifier.predict(grace_hopper[np.newaxis, ...])
result.shape

結果是1001元素向量的logits，對圖像屬於每個類的概率進行評級。
因此，可以使用argmax找到排在最前的類別ID：

predicted_class = np.argmax(result[0], axis=-1)
predicted_class

解碼預測:我們有預測的類別ID，獲取ImageNet標籤，並解碼預測

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')

imagenet_labels = np.array(open(labels_path).read().splitlines())

plt.imshow(grace_hopper)
plt.axis('off')

predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())
```

### tf.keras.utils.get_file()
```
https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file

從給定的URL中下載檔案, 可以傳遞MD5值用於資料校驗(下載後或已經緩存的資料均可)

預設情況下檔案會被下載到~/.keras中的cache_subdir資料夾，並將其檔案名設為fname，
因此若有一個檔案example.txt最終將會被存放在`~/.keras/datasets/example.txt~

tar,tar.gz.tar.bz和zip格式的檔可以被提取，提供雜湊碼可以在下載後校驗檔。

tf.keras.utils.get_file(
    fname,  #檔案名稱:如果指定了絕對路徑/path/to/file.txt,則檔案將會保存到該位置
    origin, #文件的URL地址
    untar=False, #布林值,是否要進行解壓
    md5_hash=None,  #MD5雜湊值,用於資料校驗，支援sha256和md5雜湊
    file_hash=None,
    cache_subdir='datasets', # 用於緩存資料的資料夾，若指定絕對路徑/path/to/folder則將存放在該路徑下。
    hash_algorithm='auto', #選擇檔校驗的雜湊演算法
                           #可選項有'md5', 'sha256', 和'auto'. 
                           #預設是'auto'自動檢測使用的雜湊演算法
    extract=False, #若為True則試圖提取檔，
                   # 例如tar或zip tries extracting the file as an Archive, like tar or zip.
    archive_format='auto',# 試圖提取的檔案格式，可選為'auto', 'tar', 'zip', 和None. 
                          #'tar' 包括tar, tar.gz, tar.bz文件. 
                          # 預設是'auto'是['tar', 'zip']. 
                          # None或空列表將返回沒有匹配。
    cache_dir=None   #快取檔案存放地
)

返回值:下載後的文件地址
```
### 3.簡單的遷移學習
```
使用TF Hub可以很容易地重新訓練模型的頂層以識別資料集中的類。

Dataset:本範例將使用TensorFlow鮮花資料集：

data_root = tf.keras.utils.get_file(
  'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', untar=True)

```
### tf.keras.preprocessing預處理模組

```
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence
image module: Set of tools for real-time data augmentation on image data.
sequence module: Utilities for preprocessing sequence data.
text module: Utilities for text input preprocessing.

tf.keras.preprocessing.sequence 序列型資料預處理模組
tf.keras.preprocessing.image 圖片預處理模組
```
### 圖片預處理===tf.keras.preprocessing.image模組
```
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image
將此資料載入到我們的模型中的最簡單方法是使用 
tf.keras.preprocessing.image.ImageDataGenerator類別

此類別有許多方法:
[1]apply_transform(x,transform_parameters)
   x是一個3D tensor, single image.
   transform_parameters是一大推字典資料型態定義轉換參數
     Dictionary with string - parameter pairs describing the transformation.
   Returns:A transformed version of the input (same shape)
   
[2]fit( x,augment=False, rounds=1,seed=None)  
     Fits the data generator to some sample data.   
     計算依賴於資料的變換所需要的統計資訊(均值方差等)
     只有使用featurewise_center，featurewise_std_normalization或zca_whitening時需要此函數
     
[3]flow(.....)
     接收numpy陣列和標籤為參數,生成經過資料提升或標準化後的batch資料,
     並在一個無限迴圈中不斷的返回batch資料

[4]flow_from_dataframe(.....)
     Takes the dataframe and the path to a directory 
     generates batches of augmented/normalized data.
     更多參數說明,請參閱官方網址
[5]flow_from_directory()
    以資料夾路徑為參數,產生經過資料處理過(提升/歸一化/...)後的資料
    在一個無限迴圈中無限產生batch資料 generates batches of augmented data

flow_from_directory(
    directory,  #目的檔案夾路徑
                #對於每一個類,該資料夾都要包含一個子資料夾
                #子資料夾中任何JPG、PNG、BNP、PPM的圖片都會被生成器使用
    target_size=(256, 256), #整數tuple,預設為(256, 256). `(height, width)` 圖像將被resize成該尺寸
    color_mode='rgb',  # 顏色模式,有三種選項:"grayscale", "rgb", "rgba"   預設為"rgb"
                       # 代表這些圖片是否會被轉換為單通道或三通道的圖片 
    classes=None,
    class_mode='categorical',# 有五種選項:"categorical", "binary", "sparse","input", or None.
                             # 預設為為"categorical. 
                             # 此參數決定了返回的標籤陣列的形式
                             #"categorical"會返回2D的one-hot編碼標籤,
                             #"binary"返回1D的二值標籤.
                             #"sparse"返回1D的整數標籤,
                             #"input" will be images identical to input images 
                                       (mainly used to work with autoencoders).
                             # None則不返回任何標籤, 生成器將僅僅生成batch資料
    batch_size=32, # batch資料的大小,預設32
    shuffle=True, # 是否打亂資料,預設為True
    seed=None, # 可選參數,打亂資料和進行變換時的亂數種子
    
    save_to_dir=None, # None或字串，該參數能讓你將提升後的圖片保存起來，用以視覺化
    save_prefix='',  # 字串: 保存提升後圖片時使用的首碼, 只有當設置了save_to_dir時才會生效
    save_format='png',# 圖片存檔的格式:有倆個選項:"png"或"jpeg",預設"jpeg"
    
    follow_links=False, #是否訪問子資料夾中的軟連結
    subset=None, # Subset of data (`"training"` or `"validation"`) 
                 # if `validation_split` is set in `ImageDataGenerator`.
    interpolation='nearest' #Interpolation method used to resample the image if the
                              target size is different from that of the loaded image.
                            #Supported methods are `"nearest"`, `"bilinear"`,and `"bicubic"`.
)

Returns:A `DataFrameIterator` yielding tuples of `(x, y)`
  `x` is a numpy array containing a batch of images with shape `(batch_size, *target_size, channels)`
  `y` is a numpy array of corresponding labels.

[5]get_random_transform(img_shape,seed=None)
[6]random_transform(x,seed=None)
[7]standardize(x)

類別建搆子:
__init__(
    featurewise_center=False,  #布林值，使輸入資料集去中心化（均值為0）, 按feature執行
    samplewise_center=False,  #布林值，使輸入資料的每個樣本均值為0
    featurewise_std_normalization=False, #布林值，將輸入除以資料集的標準差以完成標準化, 按feature執行
    samplewise_std_normalization=False, # 布林值，將輸入的每個樣本除以其自身的標準差
    
    zca_whitening=False, #布林值，對輸入資料施加ZCA白化
    zca_epsilon=1e-06, #ZCA使用的eposilon，默認1e-6
    
    #資料提升(data argumment)時常用參數
    rotation_range=0, #整數Int，圖片隨機轉動的角度  
    width_shift_range=0.0, #浮點數，圖片寬度的某個比例，圖片水準偏移的幅度
    height_shift_range=0.0, #浮點數，圖片高度的某個比例，資料提升時圖片豎直偏移的幅度
    
    brightness_range=None, #Tuple or list of two floats. 
                           #Range for picking a brightness shift value from.
    shear_range=0.0,  #浮點數，剪切強度（逆時針方向的剪切變換角度
    zoom_range=0.0, #浮點數或形如[lower,upper]的清單，隨機縮放的幅度
                    #若為浮點數，則相當於[lower,upper] = [1 - zoom_range, 1+zoom_range]
    channel_shift_range=0.0, #浮點數，隨機通道偏移的幅度
    
    fill_mode='nearest', #當進行變換時超出邊界的點將根據本參數給定的方法進行處理                   
                         #有四種選項: ‘constant’，‘nearest’，‘reflect’或‘wrap’
    cval=0.0, #浮點數或整數，當fill_mode=constant時，指定要向超出邊界的點填充的值

    horizontal_flip=False, #布林值，進行隨機水準翻轉
    vertical_flip=False,   #布林值，進行隨機垂直翻轉
    
    rescale=None, #所有TensorFlow Hub的圖像模組都期望浮點輸入在“[0,1]”範圍內。
                  #使用ImageDataGenerator的rescale參數來實現這一目的。
                  #放縮因數,預設為None. 如果為None或0則不進行放縮,否則會將該數值乘到資料上
                  
    preprocessing_function=None, #將被應用於每個輸入的函數。該函數將在圖片縮放和資料提升之後運行。
                                 #該函數接受一個參數，為一張圖片（秩為3的numpy array），
                                 #並且輸出一個具有相同shape的numpy array
                                 
    data_format=None, # 代表圖像的通道維的位置。 字串，“channel_first” 或 “channel_last”
                      # "channels_last" : (samples, height, width, channels), 
                      # "channels_first" : (samples, channels, height, width). 
    
    validation_split=0.0, #Float. Fraction of images reserved for validation 
                          #(strictly between 0 and 1).
    dtype=None
)

```
```
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)

結果物件是一個返回image_batch，label_batch對的反覆運算器。

for image_batch, label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Labe batch shape: ", label_batch.shape)
  break
```
### 在一批圖像上運行分類:
```
result_batch = classifier.predict(image_batch)
result_batch.shape  # (32, 1001)

predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
predicted_class_names

# 檢查這些預測如何與圖像對齊：

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(predicted_class_names[n])
  plt.axis('off')
_ = plt.suptitle("ImageNet predictions")

#結果沒有那麼完美，但考慮到這些不是模型訓練的類（“daisy雛菊”除外），這是合理的。
```
### 4.去除頂級分類層情況下的Transfer Learning==>特徵提取器
```
TensorFlow Hub還可以在沒有頂級分類層的情況下分發模型。
這些可以用來輕鬆做遷移學習。

創建特徵提取器:來自tfhub.dev的任何Tensorflow 2相容圖像特徵向量URL都可以在此處使用。

feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2" 
                                           #@param {type:"string"}

feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224,224,3))

它為每個圖像返回一個1280長度的向量：

feature_batch = feature_extractor_layer(image_batch)

print(feature_batch.shape)

凍結特徵提取器層中的變數，以便訓練僅修改新的分類器層。

feature_extractor_layer.trainable = False

附上我們要的分類層:將中心層包裝在tf.keras.Sequential模型中，並添加新的分類層。

model = tf.keras.Sequential([
  feature_extractor_layer,
  layers.Dense(image_data.num_classes, activation='softmax')
])

model.summary()

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    keras_layer_1 (KerasLayer)   (None, 1280)              2257984   
    _________________________________________________________________
    dense (Dense)                (None, 5)                 6405      
    =================================================================
    Total params: 2,264,389
    Trainable params: 6,405
    Non-trainable params: 2,257,984
    _________________________________________________________________


predictions = model(image_batch)

predictions.shape
```
### 訓練模型
### 使用compile配置訓練過程：
```
model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss='categorical_crossentropy',
  metrics=['acc'])
```
### 使用.fit方法訓練模型:此案例只訓練兩個週期
```
若要顯示訓練進度，請使用自訂回檔單獨記錄每個批次的損失和準確性，而不是記錄週期的平均值。

class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()

steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)

batch_stats_callback = CollectBatchStats()

history = model.fit(image_data, epochs=2,
                    steps_per_epoch=steps_per_epoch,
                    callbacks = [batch_stats_callback])
```

```
現在，即使只是幾次訓練反覆運算，我們已經可以看到模型正在完成任務。

plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(batch_stats_callback.batch_losses)


plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats_callback.batch_acc)
```
檢查預測:要重做之前的圖，首先獲取有序的類名列表：
```
class_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])

class_names = np.array([key.title() for key, value in class_names])

class_names
```

通過模型運行圖像批次處理，並將索引轉換為類名。
```
predicted_batch = model.predict(image_batch)

predicted_id = np.argmax(predicted_batch, axis=-1)

predicted_label_batch = class_names[predicted_id]

```
### 繪製結果
```
label_id = np.argmax(label_batch, axis=-1)

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)

for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  color = "green" if predicted_id[n] == label_id[n] else "red"
  plt.title(predicted_label_batch[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions (green: correct, red: incorrect)")

```
### 匯出模型
```
import time
t = time.time()

export_path = "/tmp/saved_models/{}".format(int(t))
tf.keras.experimental.export_saved_model(model, export_path)
```
### 重新載入模型
```
現在確認我們可以重新載入模型，它仍然給出相同的結果：

reloaded = tf.keras.experimental.load_from_saved_model(export_path, custom_objects={'KerasLayer':hub.KerasLayer})

result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)

abs(reloaded_result_batch - result_batch).max()
```

保存的模型可以在以後載入推理，或轉換為TFLite 和 TFjs。
