# 自我練習
```

```

### Linear-Regression

```
https://github.com/dragen1860/TensorFlow-2.x-Tutorials
TensorFlow-2.x-Tutorials/04-Linear-Regression/main.py
```
```
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
import  os

#定義Regressor類別[繼承keras.layers.Layer]
class Regressor(keras.layers.Layer):

    def __init__(self):
        super(Regressor, self).__init__()

        # here must specify shape instead of tensor !
        # name here is meanless !
        # [dim_in, dim_out]
        self.w = self.add_variable('meanless-name', [13, 1])
        # [dim_out]
        self.b = self.add_variable('meanless-name', [1])

        print(self.w.shape, self.b.shape)
        print(type(self.w), tf.is_tensor(self.w), self.w.name)
        print(type(self.b), tf.is_tensor(self.b), self.b.name)


    def call(self, x):
        x = tf.matmul(x, self.w) + self.b
        return x

def main():

    tf.random.set_seed(22)
    np.random.seed(22)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')

# 資料預處理:使用boston房價預測資料集
    (x_train, y_train), (x_val, y_val) = keras.datasets.boston_housing.load_data()
    #
    x_train, x_val = x_train.astype(np.float32), x_val.astype(np.float32)
    # (404, 13) (404,) (102, 13) (102,)
    
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    
    # Here has two mis-leading issues:
    # 1. (x_train, y_train) cant be written as [x_train, y_train]
    # 2.
    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64)
    db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(102)

# 模型:
    model = Regressor()
    criteon = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    
# 訓練
    for epoch in range(200):
        for step, (x, y) in enumerate(db_train):
        
            with tf.GradientTape() as tape:
                # [b, 1]
                logits = model(x)
                # [b]
                logits = tf.squeeze(logits, axis=1)
                # [b] vs [b]
                loss = criteon(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print(epoch, 'loss:', loss.numpy())


        if epoch % 10 == 0:
            for x, y in db_val:
                # [b, 1]
                logits = model(x)
                # [b]
                logits = tf.squeeze(logits, axis=1)
                # [b] vs [b]
                loss = criteon(y, logits)

                print(epoch, 'val loss:', loss.numpy())


if __name__ == '__main__':
    main()
```
### TF2_fashion_MNIST_MLP
```
import  os
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, optimizers, datasets

def prepare_mnist_features_and_labels(x, y):
  x = tf.cast(x, tf.float32) / 255.0
  y = tf.cast(y, tf.int64)

  return x, y

def mnist_dataset():
  (x, y), (x_val, y_val) = datasets.fashion_mnist.load_data()
  print('x/y shape:', x.shape, y.shape)
  
  y = tf.one_hot(y, depth=10)
  y_val = tf.one_hot(y_val, depth=10)
  
  ds = tf.data.Dataset.from_tensor_slices((x, y))
  ds = ds.map(prepare_mnist_features_and_labels)
  ds = ds.shuffle(60000).batch(100)

  ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
  ds_val = ds_val.map(prepare_mnist_features_and_labels)
  ds_val = ds_val.shuffle(10000).batch(100)

  return ds,ds_val


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

    train_dataset, val_dataset = mnist_dataset()

    model = keras.Sequential([
        layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
        layers.Dense(200, activation='relu'),
        layers.Dense(200, activation='relu'),
        layers.Dense(200, activation='relu'),
        layers.Dense(10)])
        
    # no need to use compile if you have no loss/optimizer/metrics involved here.
    model.compile(optimizer=optimizers.Adam(0.001),
                  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_dataset.repeat(), epochs=30, steps_per_epoch=500,
              validation_data=val_dataset.repeat(),
              validation_steps=2
              )

if __name__ == '__main__':
    main()
```
