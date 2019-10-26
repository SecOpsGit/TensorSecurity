#

```
%tensorflow_version 2.x
```
#  CIFAR10
```
https://www.tensorflow.org/tutorials/images/cnn
```
```
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# 資料下載與預先處理(pre-processing))
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 驗證資料
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
```

```
# 使用keras Sequential建立分類模型
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()
```
```
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_3 (Conv2D)            (None, 30, 30, 32)        896       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 15, 15, 32)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 13, 13, 64)        18496     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 6, 6, 64)          0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 4, 64)          36928     
_________________________________________________________________
flatten (Flatten)            (None, 1024)              0         
_________________________________________________________________
dense_4 (Dense)              (None, 64)                65600     
_________________________________________________________________
dense_5 (Dense)              (None, 10)                650       
=================================================================
Total params: 122,570
Trainable params: 122,570
Non-trainable params: 0
_________________________________________________________________

```

```
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
```
```
Train on 50000 samples, validate on 10000 samples
Epoch 1/10
50000/50000 [==============================] - 94s 2ms/sample - loss: 1.5284 - accuracy: 0.4431 - val_loss: 1.2152 - val_accuracy: 0.5645
Epoch 2/10
50000/50000 [==============================] - 94s 2ms/sample - loss: 1.1470 - accuracy: 0.5927 - val_loss: 1.1128 - val_accuracy: 0.6083
Epoch 3/10
50000/50000 [==============================] - 93s 2ms/sample - loss: 1.0016 - accuracy: 0.6469 - val_loss: 0.9802 - val_accuracy: 0.6643
Epoch 4/10
50000/50000 [==============================] - 93s 2ms/sample - loss: 0.9063 - accuracy: 0.6821 - val_loss: 0.9326 - val_accuracy: 0.6806
Epoch 5/10
50000/50000 [==============================] - 92s 2ms/sample - loss: 0.8328 - accuracy: 0.7101 - val_loss: 0.8915 - val_accuracy: 0.6938
Epoch 6/10
50000/50000 [==============================] - 93s 2ms/sample - loss: 0.7754 - accuracy: 0.7277 - val_loss: 0.8985 - val_accuracy: 0.6903
Epoch 7/10
50000/50000 [==============================] - 92s 2ms/sample - loss: 0.7260 - accuracy: 0.7460 - val_loss: 0.8650 - val_accuracy: 0.7092
Epoch 8/10
50000/50000 [==============================] - 93s 2ms/sample - loss: 0.6843 - accuracy: 0.7575 - val_loss: 0.9688 - val_accuracy: 0.6818
Epoch 9/10
50000/50000 [==============================] - 94s 2ms/sample - loss: 0.6433 - accuracy: 0.7732 - val_loss: 0.9026 - val_accuracy: 0.7019
Epoch 10/10
50000/50000 [==============================] - 92s 2ms/sample - loss: 0.6044 - accuracy: 0.7879 - val_loss: 0.8897 - val_accuracy: 0.7103
```
```
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
```

```
print(test_acc)
```

```
0.7103
```



# MNIST
```
針對專業人員的 TensorFlow 2.0 入門
https://www.tensorflow.org/tutorials/quickstart/advanced
```

```
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

#載入並預先處理MNIST 資料集。

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

#使用 tf.data 來將資料集切分為 batch 並混淆資料集(shuffle)：

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

#####################################################################
# 使用 Keras 模型子類化（model subclassing） API 構建 tf.keras 模型：
#####################################################################
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)
#####################################################################

model = MyModel()

#選擇優化器與損失函數：

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

#選擇衡量指標來度量模型的損失值（loss）和準確率（accuracy）。這些指標在 epoch 上累積值，然後列印出整體結果。

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

#使用 tf.GradientTape 來訓練模型：

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

#測試模型：

@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))

```

```

```
