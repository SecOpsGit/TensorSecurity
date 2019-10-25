# Resnet
```

```


#  Resnet@TF2 | cifar100
```
https://github.com/dragen1860/TensorFlow-2.x-Tutorials/tree/master/11-AE

深度学习与TensorFlow入门实战-源码和PPT目錄底下
lesson43-ResNet
```


```
%tensorflow_version 2.x
```
### 定義ResNet類別

```
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, Sequential



class BasicBlock(layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x:x



    def call(self, inputs, training=None):

        # [b, h, w, c]
        out = self.conv1(inputs)
        out = self.bn1(out,training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out,training=training)

        identity = self.downsample(inputs)

        output = layers.add([out, identity])
        output = tf.nn.relu(output)

        return output


class ResNet(keras.Model):

    def __init__(self, layer_dims, num_classes=100): # [2, 2, 2, 2]
        super(ResNet, self).__init__()

        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                ])

        self.layer1 = self.build_resblock(64,  layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        # output: [b, 512, h, w],
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)


    def call(self, inputs, training=None):

        x = self.stem(inputs,training=training)

        x = self.layer1(x,training=training)
        x = self.layer2(x,training=training)
        x = self.layer3(x,training=training)
        x = self.layer4(x,training=training)

        # [b, c]
        x = self.avgpool(x)
        # [b, 100]
        x = self.fc(x)

        return x

    def build_resblock(self, filter_num, blocks, stride=1):

        res_blocks = Sequential()
        # may down sample
        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks

def resnet18():
    return ResNet([2, 2, 2, 2])

def resnet34():
    return ResNet([3, 4, 6, 3])
```

### 主程式
```
import  os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import  tensorflow as tf
from    tensorflow.keras import layers, optimizers, datasets, Sequential 
#from    resnet import resnet18 

tf.random.set_seed(2345)


def preprocess(x, y):
    # [-1~1]
    x = tf.cast(x, dtype=tf.float32) / 255. - 0.5
    y = tf.cast(y, dtype=tf.int32)
    return x,y

# 載入cifar100並分割成train dataset+ test dataset
(x,y), (x_test, y_test) = datasets.cifar100.load_data()


y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x.shape, y.shape, x_test.shape, y_test.shape)
# 


train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(1000).map(preprocess).batch(512)

test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(512)

sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))


def main():

    # [b, 32, 32, 3] => [b, 1, 1, 512]
    model = resnet18()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    optimizer = optimizers.Adam(lr=1e-3)

    for epoch in range(500):

        for step, (x,y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 100]
                logits = model(x,training=True)
                # [b] => [b, 100]
                y_onehot = tf.one_hot(y, depth=100)
                # compute loss
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step %50 == 0:
                print(epoch, step, 'loss:', float(loss))



        total_num = 0
        total_correct = 0
        for x,y in test_db:

            logits = model(x,training=False)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print(epoch, 'acc:', acc)



if __name__ == '__main__':
    main()
```

```
Downloading data from https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
169009152/169001437 [==============================] - 11s 0us/step
(50000, 32, 32, 3) (50000,) (10000, 32, 32, 3) (10000,)
sample: (512, 32, 32, 3) (512,) tf.Tensor(-0.5, shape=(), dtype=float32) tf.Tensor(0.5, shape=(), dtype=float32)
Model: "res_net"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
sequential (Sequential)      multiple                  2048      
_________________________________________________________________
sequential_1 (Sequential)    multiple                  148736    
_________________________________________________________________
sequential_2 (Sequential)    multiple                  526976    
_________________________________________________________________
sequential_4 (Sequential)    multiple                  2102528   
_________________________________________________________________
sequential_6 (Sequential)    multiple                  8399360   
_________________________________________________________________
global_average_pooling2d (Gl multiple                  0         
_________________________________________________________________
dense (Dense)                multiple                  51300     
=================================================================
Total params: 11,230,948[1千多萬個參數!!]
Trainable params: 11,223,140
Non-trainable params: 7,808
_________________________________________________________________
0 0 loss: 5.247768402099609
....

```
