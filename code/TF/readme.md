#
```
deeplearning.ai TensorFlow in Practice 專項課程

此專項課程包含 4 門課程

課程 1
Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

課程 2
Convolutional Neural Networks in TensorFlow

課程 3
Natural Language Processing in TensorFlow

課程 4
Sequences, Time Series and Prediction
```


# Tensorflow 範例程式
```

https://github.com/dragen1860/TensorFlow-2.x-Tutorials
```
```
tensorflow2_tutorials_chinese
tensorflow2中文教程
https://github.com/czy36mengfei/tensorflow2_tutorials_chinese
```
```
https://github.com/sjchoi86/Tensorflow-101
https://github.com/aymericdamien/TensorFlow-Examples
https://github.com/machinelearningmindset/TensorFlow-Course

https://github.com/terryum/TensorFlow_Exercises
```
```
Customization basics: tensors and operations====https://www.tensorflow.org/tutorials/customization/basics
```
### 使用tensorflow 2.x
```
開啟新的ipynb

 %tensorflow_version 2.x
```
```
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf


a = tf.constant(1.)
b = tf.constant(2.)
print(a+b)
```
```
tf.Tensor(3.0, shape=(), dtype=float32)
```
```
20191023
原先一開始開啟的ipynb 執行後結果會是 Graph Model

Tensor("add:0", shape=(), dtype=float32)
```
### Datasets內建資料集==tensorflow_datasets
```
https://www.tensorflow.org/datasets/api_docs/python/tfds

https://s0www0tensorflow0org.icopy.site/datasets/api_docs/python/tfds

Google開源機器學習資料集，可在 TensorFlow 直接調用
https://www.infoq.cn/article/w7FY0rk5ba-TgzlxyaEY
```
```
import tensorflow as tf
import tensorflow_datasets as tfds

mnist_data = tfds.load("mnist")
mnist_train, mnist_test = mnist_data["train"], mnist_data["test"]
assert isinstance(mnist_train, tf.data.Dataset)
```
```
# See all registered datasets
tfds.list_builders()

# Load a given dataset by name, along with the DatasetInfo
data, info = tfds.load("mnist", with_info=True)
train_data, test_data = data['train'], data['test']
assert isinstance(train_data, tf.data.Dataset)
assert info.features['label'].num_classes == 10
assert info.splits['train'].num_examples == 60000

# You can also access a builder directly
builder = tfds.builder("mnist")
assert builder.info.splits['train'].num_examples == 60000
builder.download_and_prepare()
datasets = builder.as_dataset()

# If you need NumPy arrays
np_datasets = tfds.as_numpy(datasets)
```
```
https://www.tensorflow.org/datasets/catalog/overview

https://tf.wiki/zh/appendix/tfds.html
```
### TensorFlow Hub 
```
TensorFlow Hub 是一個針對可重複使用的機器學習模組的類別庫，
用於發佈、發現和使用機器學習模型中可重複利用的部分。
模組是一個獨立的 TensorFlow 圖部分，其中包含權重和資源，
可以在一個進程中供不同任務重複使用（稱為遷移學習）。

遷移學習可以：
使用較小的資料集訓練模型，
改善泛化效果，以及
加快訓練速度。

Introducing TensorFlow Hub: 
A Library for Reusable Machine Learning Modules in TensorFlow
https://medium.com/tensorflow/introducing-tensorflow-hub-a-library-for-reusable-machine-learning-modules-in-tensorflow-cdee41fa18f9

https://www.tensorflow.org/hub/overview
https://tfhub.dev/

example and tutorials:
https://github.com/tensorflow/hub/blob/master/examples/README.md
```
```
  !pip install "tensorflow_hub==0.4.0"
  !pip install "tf-nightly"

  import tensorflow as tf
  import tensorflow_hub as hub

  tf.enable_eager_execution()

  module_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
  embed = hub.KerasLayer(module_url)
  embeddings = embed(["A long sentence.", "single-word",
                      "http://example.com"])
  print(embeddings.shape)  #(3,128)
```
```
# Download and use NASNet feature vector module.
module = hub.Module("https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/1")
features = module(my_images)
logits = tf.layers.dense(features, NUM_CLASSES)
probabilities = tf.nn.softmax(logits)
```
### TensorFlow tutorials
```
https://www.tensorflow.org/tutorials
```
# CNN

###
```
用 tf.data 載入圖片  https://www.tensorflow.org/tutorials/load_data/images
```

### Transfer learning
```
https://adventuresinmachinelearning.com/transfer-learning-tensorflow-2/

https://lambdalabs.com/blog/tensorflow-2-0-tutorial-02-transfer-learning/

https://github.com/lambdal/TensorFlow2-tutorial
```
# RNN
```
Text generation with an RNN: https://www.tensorflow.org/tutorials/text/text_generation

Time series forecasting:https://www.tensorflow.org/tutorials/structured_data/time_series
```

# Generative Model

### Convolutional Variational Autoencoder  
```
https://www.tensorflow.org/tutorials/generative/cvae
```
### GAN
```

CycleGAN: https://www.tensorflow.org/tutorials/generative/cyclegan
Pix2Pix:https://www.tensorflow.org/tutorials/generative/pix2pix

DCGAN:https://www.tensorflow.org/tutorials/generative/dcgan
```
