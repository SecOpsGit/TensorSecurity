# TensorFlow 2.0 Architecture

```
pip install tensorflow==1.15

#for GPU support: pip install tensorflow-gpu==1.15

pip install tensorflow==2.0
#for GPU support: pip install tensorflow-gpu==2.0

```
```
%tensorflow_version 2.x
```

```
Hands-On Neural Networks with TensorFlow 2.0
Paolo Galeone
September 18, 2019
```
# CH 3 TensorFlow Graph Architecture

### TF1
```
import tensorflow as tf
import numpy as np
A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
x = tf.constant([[0, 10], [0, 0.5]])
b = tf.constant([[1, -1]], dtype=tf.float32)
y = tf.add(tf.matmul(A, x), b, name="result")

writer = tf.summary.FileWriter("log/matmul", tf.get_default_graph())
writer.close()

with tf.Session() as sess:
   A_value, x_value, b_value = sess.run([A, x, b])
   y_value = sess.run(y)
# Overwrite
   y_new = sess.run(y, feed_dict={b: np.zeros((1, 2))})

print(f"A: {A_value}\nx: {x_value}\nb: {b_value}\n\ny: {y_value}")
print(f"y_new: {y_new}")
```
```
對TF1的抱怨:
A steep learning curve
Hard to debug
Counter-intuitive semantics when it comes to certain operations
Python is only used to build the graph
```

```
The release of TensorFlow 2.0 introduced several changes to the framework: 
from defaulting to eager execution to a complete cleanup of the APIs. 

The whole TensorFlow package, in fact, was full of duplicated and deprecated APIs 
that, in TensorFlow 2.0, have been finally removed. 

Moreover, by deciding to follow the Keras API specification, the TensorFlow developers decided to remove several modules 
that do not follow it: the most important removal was tf.layers (which we used in Chapter 3, TensorFlow Graph
Architecture) in favor of tf.keras.layers.

Another widely used module, tf.contrib, has been completely removed. 
The tf.contrib module contained the community-added layers/software that used TensorFlow. 
From a software engineering point of view, having a module that contains
several completely unrelated and huge projects in one package is a terrible idea. 
For this reason, they removed it from the main package and decided to move maintained and huge
modules into separate repositories, while removing unused and unmaintained modules.

The Keras framework and its models 
In contrast to what people who already familiar with Keras usually think, 
Keras is not a high-level wrapper around a machine learning framework (TensorFlow, CNTK, or Theano); 
instead, it is an API specification that's used for defining and training machine learning models.

TensorFlow implements the specification in its tf.keras module. 
In particular, TensorFlow 2.0 itself is an implementation of the specification 
and as such, many first-level submodules are nothing but aliases of the tf.keras submodules; 
for example, tf.metrics = tf.keras.metrics and 
tf.optimizers = tf.keras.optimizers.

Since TensorFlow 2.0's release, there has been no need to download a separate Python package in order to use Keras. 
The tf.keras module is already built into the tensorflow package, and it has some TensorFlow-specific enhancements.

Eager execution is a first-class citizen, just like the high-performance input pipeline module known as tf.data. 
Exporting a model that's been created using Keras is even easier than exporting a model defined in plain TensorFlow. 
Being exported in a language-agnostic format means that 
its compatibility with any production environment has already been configured, 
and so it is guaranteed to work with TensorFlow.
```
# Ch4.TensorFlow 2.0 Architecture
```


```
# 重用Keras

### 使用Sequential API 
```
%tensorflow_version 2.x
```
```
import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist

n_classes = 10
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        32, (5, 5), activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D((2, 2), (2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D((2, 2), (2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(n_classes)
])

model.summary()

(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
# Scale input in [-1, 1] range
train_x = train_x / 255. * 2 - 1
test_x = test_x / 255. * 2 - 1
train_x = tf.expand_dims(train_x, -1).numpy()
test_x = tf.expand_dims(test_x, -1).numpy()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_x, train_y, epochs=10)
model.evaluate(test_x, test_y)
```

### 使用keras Functional API 
```
The Sequential API is the simplest and most common way of defining models. 
However, it cannot be used to define arbitrary models. 
The Functional API allows you to define complex topologies without the constraints of the sequential layers.
The Functional API allows you to define multi-input, multi-output models, easily sharing
layers, defines residual connections, and in general define models with arbitrary complex
topologies.

Once built, a Keras layer is a callable object that accepts an input tensor and produces an output tensor. 
It knows that it is possible to compose the layers by treating them as functions and 
building a tf.keras.Model object just by passing the input and output layers.
```
```
import tensorflow as tf

input_shape = (100,)
inputs = tf.keras.layers.Input(input_shape)
net = tf.keras.layers.Dense(units=64, activation=tf.nn.elu, name="fc1")(inputs)
net = tf.keras.layers.Dense(units=64, activation=tf.nn.elu, name="fc2")(net)
net = tf.keras.layers.Dense(units=1, name="G")(net)
model = tf.keras.Model(inputs=inputs, outputs=net)
model.summary()
```

### 使用The subclassing method[不推薦]
```
The Sequential and Functional APIs cover almost any possible scenario. 
However, Keras offers another way of defining models that is object-oriented, more flexible, 
but error-prone and harder to debug. 
In practice, it is possible to subclass any tf.keras.Model by defining the layers in __init__ 
and the forward passing in the call method:
```


```
import tensorflow as tf


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense_1 = tf.keras.layers.Dense(
            units=64, activation=tf.nn.elu, name="fc1")
        self.dense_2 = f.keras.layers.Dense(
            units=64, activation=tf.nn.elu, name="fc2")
        self.output = f.keras.layers.Dense(units=1, name="G")

    def call(self, inputs):
        # Build the model in functional style here
        # and return the output tensor
        net = self.dense_1(inputs)
        net = self.dense_2(net)
        net = self.output(net)
        return net
```


# Eager execution
```
https://www.tensorflow.org/guide/eager

TensorFlow's eager execution is an imperative programming environment that evaluates operations immediately, without building graphs: 
operations return concrete values instead of constructing a computational graph to run later. 

This makes it easy to get started with TensorFlow and debug models, and it reduces boilerplate as well. 

To follow along with this guide, run the code samples below in an interactive python interpreter.

Eager execution is a flexible machine learning platform for research and experimentation, providing:

An intuitive interface—Structure your code naturally and use Python data structures. 
Quickly iterate on small models and small data.

Easier debugging—Call ops directly to inspect running models and test changes. 
Use standard Python debugging tools for immediate error reporting.

Natural control flow—Use Python control flow instead of graph control flow, 
simplifying the specification of dynamic models.

Eager execution supports most TensorFlow operations and GPU acceleration.
```

### 檢查是否使用Eager execution
```
import tensorflow as tf
tf.executing_eagerly()
```


### TF1 vs TF2
```
import tensorflow as tf
A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
x = tf.constant([[0, 10], [0, 0.5]])
b = tf.constant([[1, -1]], dtype=tf.float32)
y = tf.add(tf.matmul(A, x), b, name="result") #y = Ax + b

with tf.Session() as sess:
   print(sess.run(y))
```


```
import tensorflow as tf
A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
x = tf.constant([[0, 10], [0, 0.5]])
b = tf.constant([[1, -1]], dtype=tf.float32)

y = tf.add(tf.matmul(A, x), b, name="result")
print(y)
```


### 關注Functions, 不再思考sessions
```
The tf.Session object has been removed from the TensorFlow API. 

By focusing on eager execution, you no longer need the concept of a session 
because the execution of the operation is immediate—we don't build a computational graph 
before running the computation.

This opens up a new scenario, in which the source code can be organized better. 
In TensorFlow 1.x, it was tough to design software by following object-oriented programming
principles or even create modular code that used Python functions. 

However, in TensorFlow 2.0, this is natural and is highly recommended.
```


```
import tensorflow as tf


def multiply(x, y):
    """Matrix multiplication.
    Note: it requires the input shape of both input to match.
    Args:
        x: tf.Tensor a matrix
        y: tf.Tensor a matrix
    Returns:
        The matrix multiplcation x @ y
    """

    assert x.shape == y.shape
    return tf.matmul(x, y)


def add(x, y):
    """Add two tensors.
    Args:
        x: the left hand operand.
        y: the right hand operand. It should be compatible with x.
    Returns:
        x + y
    """
    return x + y


def main():
    """Main program."""
    A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    x = tf.constant([[0, 10], [0, 0.5]])
    b = tf.constant([[1, -1]], dtype=tf.float32)

    z = multiply(A, x)
    y = add(z, b)
    print(y)


if __name__ == "__main__":
    main()
```

### 更清楚的Control flow

問題:
```
1. Declare and initialize two variables: x and y.
2. Increment the value of y by 1.
3. Compute x*y.
4. Repeat this five times.
```

TF1的寫法:
```
import tensorflow as tf

x = tf.Variable(1, dtype=tf.int32)
y = tf.Variable(2, dtype=tf.int32)

for _ in range(5):
    y.assign_add(1)
    out = x * y
    print(out)
    tf.print(out)
```
```
tf.Tensor(3, shape=(), dtype=int32)
3
tf.Tensor(4, shape=(), dtype=int32)
4
tf.Tensor(5, shape=(), dtype=int32)
5
tf.Tensor(6, shape=(), dtype=int32)
6
tf.Tensor(7, shape=(), dtype=int32)
7
```

TF2的寫法:
```
import tensorflow as tf

x = tf.Variable(1, dtype=tf.int32)
y = tf.Variable(2, dtype=tf.int32)

for _ in range(5):
   y.assign_add(1)
   out = x * y
   print(out)
```
