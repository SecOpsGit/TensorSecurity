# Tensorflow 系統開發

```


```
```
Customization basics: tensors and operations====https://www.tensorflow.org/tutorials/customization/basics
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
模組是一個獨立的 TensorFlow 圖部分，其中包含權重和資源，可以在一個進程中供不同任務重複使用（稱為遷移學習）。

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

###
```

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
