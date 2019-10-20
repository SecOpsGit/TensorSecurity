# Tensorflow 系統開發

```


```
```
Customization basics: tensors and operations====https://www.tensorflow.org/tutorials/customization/basics
```
### Datasets內建資料集
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
