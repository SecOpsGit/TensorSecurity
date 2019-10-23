
# eager execution

### colab預設是使用 tensorflow 1.x

```
import tensorflow as tf
tf.__version__
```
```
'1.15.0'
```

```
from __future__ import absolute_import, division, print_function, unicode_literals
import os

import tensorflow as tf
tf.executing_eagerly()
```
```
False
```
# eager execution@ Tensorflow 1.15

# 另外新啟動一個cell 
```
%tensorflow_version 2.x
```
```
TensorFlow 2.x selected.
```


```
import tensorflow as tf
tf.__version__
```

```
'2.0.0'
```

### Tensorflow 2.0, eager execution is enabled by default

```
import tensorflow as tf
tf.executing_eagerly()
```
```
True
```

### 無須再使用 tf.Session
```
# Define constant tensors
print("Define constant tensors")
a = tf.constant(2)
print("a = %i" % a)
b = tf.constant(3)
print("b = %i" % b)

# Run the operation without the need for tf.Session
print("Running operations, without tf.Session")
c = a + b
print("a + b = %i" % c)
d = a * b
print("a * b = %i" % d)
```
# Full compatibility with Numpy
```
import numpy as np
print("Mixing operations with Tensors and Numpy Arrays")

# Define constant tensors
a = tf.constant([[2., 1.],
                 [1., 0.]], dtype=tf.float32)
print("Tensor:\n a = %s" % a)

b = np.array([[3., 0.],
              [5., 1.]], dtype=np.float32)
print("NumpyArray:\n b = %s" % b)
```
```
請參閱更多
https://www.tensorflow.org/guide/eager?hl=zh_cn
```
