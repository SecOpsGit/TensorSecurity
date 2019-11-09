#
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
```
# 下載分類器
使用hub.module載入mobilenet，並使用tf.keras.layers.Lambda將其包裝為keras層。
來自tfhub.dev的任何相容tf2的圖像分類器URL都可以在這裡工作。

classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"  
#@param {type:"string"}

IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])


2.2. 在單個圖像上運行它
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


2.3. 解碼預測
我們有預測的類別ID，獲取ImageNet標籤，並解碼預測

labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())
```

### 3.簡單的遷移學習
```
使用TF Hub可以很容易地重新訓練模型的頂層以識別資料集中的類。

3.1. Dataset
對於此示例，您將使用TensorFlow鮮花資料集：

data_root = tf.keras.utils.get_file(
  'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)


將此資料載入到我們的模型中的最簡單方法是使用 tf.keras.preprocessing.image.ImageDataGenerator,

所有TensorFlow Hub的圖像模組都期望浮點輸入在“[0,1]”範圍內。
使用ImageDataGenerator的rescale參數來實現這一目的。圖像大小將在稍後處理。

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)

結果物件是一個返回image_batch，label_batch對的反覆運算器。

for image_batch, label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Labe batch shape: ", label_batch.shape)
  break


3.2. 在一批圖像上運行分類器
現在在圖像批次處理上運行分類器。

result_batch = classifier.predict(image_batch)
result_batch.shape  # (32, 1001)

predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
predicted_class_names

現在檢查這些預測如何與圖像對齊：

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(predicted_class_names[n])
  plt.axis('off')
_ = plt.suptitle("ImageNet predictions")


有關圖像屬性，請參閱LICENSE.txt文件。

結果沒有那麼完美，但考慮到這些不是模型訓練的類（“daisy雛菊”除外），這是合理的。
```
### 3.3. 下載無頭模型
```
TensorFlow Hub還可以在沒有頂級分類層的情況下分發模型。
這些可以用來輕鬆做遷移學習。

來自tfhub.dev的任何Tensorflow 2相容圖像特徵向量URL都可以在此處使用。

feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2" #@param {type:"string"}

創建特徵提取器。

feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224,224,3))

它為每個圖像返回一個1280長度的向量：

feature_batch = feature_extractor_layer(image_batch)

print(feature_batch.shape)

凍結特徵提取器層中的變數，以便訓練僅修改新的分類器層。

feature_extractor_layer.trainable = False


3.4. 附上分類頭
現在將中心層包裝在tf.keras.Sequential模型中，並添加新的分類層。

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


3.5. 訓練模型
使用compile配置訓練過程：

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss='categorical_crossentropy',
  metrics=['acc'])


現在使用.fit方法訓練模型。

這個例子只是訓練兩個週期。
要顯示訓練進度，請使用自訂回檔單獨記錄每個批次的損失和準確性，而不是記錄週期的平均值。

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



3.6. 檢查預測
要重做之前的圖，首先獲取有序的類名列表：

class_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])
class_names


通過模型運行圖像批次處理，並將索引轉換為類名。

predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]


繪製結果

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


4. 匯出你的模型
現在您已經訓練了模型，將其匯出為已保存的模型：

import time
t = time.time()

export_path = "/tmp/saved_models/{}".format(int(t))
tf.keras.experimental.export_saved_model(model, export_path)



現在確認我們可以重新載入它，它仍然給出相同的結果：

reloaded = tf.keras.experimental.load_from_saved_model(export_path, custom_objects={'KerasLayer':hub.KerasLayer})

result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)

abs(reloaded_result_batch - result_batch).max()


這個保存的模型可以在以後載入推理，或轉換為TFLite 和 TFjs。
```
