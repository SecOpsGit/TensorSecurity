
# eager execution
```
Eager Execution (TensorFlow Dev Summit 2018)
https://www.youtube.com/watch?v=T8AW0fKP0Hs
```

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
```
2019 Google開發者大會於 9 月 10 日和 11 日在上海舉辦，大會將分享眾多開發經驗與工具。
在第一天的 KeyNote 中，Google發佈了很多開發工具新特性，並介紹而它們是如何構建更好的應用。
值得注意的是，TensorFlow 剛剛發佈了 2.0 RC01 版和 1.15，Google表示 1.15 是 1.x 的最後一次更新了。
```

### 早期啟用eager execution的做法
```
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
tfe.enable_eager_execution()
```
### 2019.10.23:已經不能這樣了
```
WARNING:tensorflow:
The TensorFlow contrib module will not be included in TensorFlow 2.0.
For more information, please see:
  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
  * https://github.com/tensorflow/addons
  * https://github.com/tensorflow/io (for I/O related ops)
If you depend on functionality not listed there, please file an issue.

---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-1-b9a19578cd20> in <module>()
      2 import tensorflow.contrib.eager as tfe
      3 import numpy as np
----> 4 tfe.enable_eager_execution()

AttributeError: module 'tensorflow.contrib.eager' has no attribute 'enable_eager_execution'
```

### 正確的啟用eager execution方法
```
import tensorflow as tf

tf.enable_eager_execution()

tf.executing_eagerly()
```
### 成功了! 20191023! 可以不用tf.Session
```
# Define constant tensors
print("Define constant tensors")
a = tf.constant(2)
print("a = %i" % a)
b = tf.constant(3)
print("b = %i" % b)
```

# eager execution@ Tensorflow 2.x====另外新啟動一個cell 
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
# GPU
```
import tensorflow as tf
import time

def time_matmul(x):
  start = time.time()
  for loop in range(10):
    tf.matmul(x, x)
  result = time.time()-start
  print("10 loops: {:0.2f}ms".format(1000*result))

# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
  x = tf.random.uniform([1000, 1000])
  assert x.device.endswith("CPU:0")
  time_matmul(x)

# Force execution on GPU #0 if available
if tf.test.is_gpu_available():
  print("On GPU:")
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)
```
