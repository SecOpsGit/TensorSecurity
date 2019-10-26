# TensorFlow Datasets
```
TensorFlow Datasets provides many public datasets as tf.data.Datasets

```
### Installation
```

pip install tensorflow-datasets

# Requires TF 1.15+ to be installed.
# Some datasets require additional libraries; see setup.py extras_require
pip install tensorflow
# or:
pip install tensorflow-gpu
```
### 版本
```
Semantic
Every DatasetBuilder defined in TFDS comes with a version, for example:

class MNIST(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version("1.0.0")
```
```
Supported versions
A DatasetBuilder can support several versions, which can be both higher or lower than the canonical version. For example:

class Imagenet2012(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('2.0.1', 'Encoding fix. No changes from user POV')
  SUPPORTED_VERSIONS = [
      tfds.core.Version('3.0.0', 'S3: tensorflow.org/datasets/splits'),
      tfds.core.Version('2.0.1', 'Encoding fix. No changes from user POV'),
      tfds.core.Version('1.0.0'),
  ]
```
```
https://www.tensorflow.org/datasets/datasets_versioning


Loading a specific version
When loading a dataset or a DatasetBuilder, you can specify the version to use. For example:

tfds.load('imagenet2012:2.0.1')
tfds.builder('imagenet2012:2.0.1')

tfds.load('imagenet2012:2.0.0')  # Error: unsupported version.

# Resolves to 3.0.0 for now, but would resolve to 3.1.1 if when added.
tfds.load('imagenet2012:3.*.*')

```

### 使用方次
```
import tensorflow_datasets as tfds
import tensorflow as tf

# tfds works in both Eager and Graph modes
tf.compat.v1.enable_eager_execution()

# See available datasets
print(tfds.list_builders())

# Construct a tf.data.Dataset
ds_train, ds_test = tfds.load(name="mnist", split=["train", "test"])

# Build your input pipeline
ds_train = ds_train.shuffle(1000).batch(128).prefetch(10)
for features in ds_train.take(1):
  image, label = features["image"], features["label"]

```
