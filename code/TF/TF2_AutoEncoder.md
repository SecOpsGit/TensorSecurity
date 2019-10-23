# AutoEncoder
```
AutoEncoder是深度學習的另外一個重要內容，並且非常有意思，神經網路通過大量資料集，進行end-to-end的訓練，不斷提高其準確率，
而AutoEncoder通過設計encode和decode過程使輸入和輸出越來越接近，是一種無監督學習過程。
```
# AutoEncoder@ TensorFlow-2
```
https://github.com/dragen1860/TensorFlow-2.x-Tutorials/tree/master/11-AE

深度学习与TensorFlow入门实战-源码和PPT目錄底下lesson48-AutoEncoders
```

```
 %tensorflow_version 2.x
```
```
import  os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from    tensorflow.keras import Sequential, layers
from    PIL import Image
from    matplotlib import pyplot as plt



tf.random.set_seed(22)
np.random.seed(22) 
assert tf.__version__.startswith('2.')


def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1

    new_im.save(name)

h_dim = 20
batchsz = 512
lr = 1e-3

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.

# we do not need label
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batchsz)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# 定義模型(類別)###################################
class AE(keras.Model):

    def __init__(self):
        super(AE, self).__init__()

        # Encoders
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(h_dim)
        ])

        # Decoders
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784)
        ])


    def call(self, inputs, training=None):
        # [b, 784] => [b, 10]
        h = self.encoder(inputs)
        # [b, 10] => [b, 784]
        x_hat = self.decoder(h)

        return x_hat
###################################

model = AE()
model.build(input_shape=(None, 784))
model.summary()

optimizer = tf.optimizers.Adam(lr=lr)

# 早期版本:optimizer = tf.train.GradientDescentOptimizer(learning_rate)


for epoch in range(100):

    for step, x in enumerate(train_db):

        #[b, 28, 28] => [b, 784]
        x = tf.reshape(x, [-1, 784])

#計算中間節點的梯度
        with tf.GradientTape() as tape:
            x_rec_logits = model(x)

            rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            rec_loss = tf.reduce_mean(rec_loss)

        grads = tape.gradient(rec_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


        if step % 100 ==0:
            print(epoch, step, float(rec_loss))


        # evaluation
        x = next(iter(test_db))
        logits = model(tf.reshape(x, [-1, 784]))
        x_hat = tf.sigmoid(logits)
        # [b, 784] => [b, 28, 28]
        x_hat = tf.reshape(x_hat, [-1, 28, 28])

        # [b, 28, 28] => [2b, 28, 28]
        x_concat = tf.concat([x, x_hat], axis=0)
        x_concat = x_hat
        x_concat = x_concat.numpy() * 255.
        x_concat = x_concat.astype(np.uint8)
        save_images(x_concat, 'rec_epoch_%d.png'%epoch)
```

```
(60000, 28, 28) (60000,)
(10000, 28, 28) (10000,)
Model: "ae_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
sequential_2 (Sequential)    multiple                  236436    
_________________________________________________________________
sequential_3 (Sequential)    multiple                  237200    
=================================================================
Total params: 473,636
Trainable params: 473,636
Non-trainable params: 0
_________________________________________________________________
0 0 0.6926943063735962
0 100 0.3214707374572754
1 0 0.30643314123153687
1 100 0.3051005005836487
2 0 0.2951914072036743
2 100 0.3031919002532959

要跑很久
```
