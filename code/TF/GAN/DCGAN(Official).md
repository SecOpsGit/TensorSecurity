#
```

```

```
# -*- coding: utf-8 -*-


"""# 深度卷積生成對抗網路

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://tensorflow.google.cn/tutorials/generative/dcgan">
    <img src="https://tensorflow.google.cn/images/tf_logo_32px.png" />
    在 tensorFlow.google.cn 上查看</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/zh-cn/tutorials/generative/dcgan.ipynb">
    <img src="https://tensorflow.google.cn/images/colab_logo_32px.png" />
    在 Google Colab 中運行</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/zh-cn/tutorials/generative/dcgan.ipynb">
    <img src="https://tensorflow.google.cn/images/GitHub-Mark-32px.png" />
    在 GitHub 上查看原始程式碼</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/zh-cn/tutorials/generative/dcgan.ipynb"><img src="https://tensorflow.google.cn/images/download_logo_32px.png" />下載 notebook</a>
  </td>
</table>

Note: 我們的 TensorFlow 社區翻譯了這些文檔。因為社區翻譯是盡力而為， 所以無法保證它們是最準確的，並且反映了最新的
[官方英文文檔](https://www.tensorflow.org/?hl=en)。如果您有改進此翻譯的建議， 請提交 pull request 到
[tensorflow/docs](https://github.com/tensorflow/docs) GitHub 倉庫。要志願地撰寫或者審核譯文，請加入
[docs-zh-cn@tensorflow.org Google Group](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-zh-cn)。

本教程演示了如何使用[深度卷積生成對抗網路](https://arxiv.org/pdf/1511.06434.pdf)（DCGAN）生成手寫數位圖片。該代碼是使用 [Keras Sequential API](https://tensorflow.google.cn/guide/keras) 與 `tf.GradientTape` 訓練迴圈編寫的。

## 什麼是生成對抗網路？

[生成對抗網路](https://arxiv.org/abs/1406.2661)（GANs）是當今電腦科學領域最有趣的想法之一。兩個模型通過對抗過程同時訓練。一個*生成器*（“藝術家”）學習創造看起來真實的圖像，而*判別器*（“藝術評論家”）學習區分真假圖像。

![生成器和判別器圖示](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/gan1.png?raw=1)

訓練過程中，*生成器*在生成逼真圖像方面逐漸變強，而*判別器*在辨別這些圖像的能力上逐漸變強。當*判別器*不再能夠區分真實圖片和偽造圖片時，訓練過程達到平衡。

![生成器和判別器圖示二](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/gan2.png?raw=1)

本筆記在 MNIST 資料集上演示了該過程。下方動畫展示了當訓練了 50 個epoch （全部資料集反覆運算50次） 時*生成器*所生成的一系列圖片。圖片從隨機雜訊開始，隨著時間的推移越來越像手寫數字。

![輸出樣本](https://tensorflow.google.cn/images/gan/dcgan.gif)

要瞭解關於 GANs 的更多資訊，我們建議參閱 MIT的 [深度學習入門](http://introtodeeplearning.com/) 課程。

### Import TensorFlow and other libraries
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Commented out IPython magic to ensure Python compatibility.
try:
  # %tensorflow_version 只在 Colab 中使用。
#   %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf

tf.__version__

# 用於生成 GIF 圖片
!pip install imageio

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

"""### 載入和準備資料集

您將使用 MNIST 資料集來訓練生成器和判別器。生成器將生成類似於 MNIST 資料集的手寫數字。
"""

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # 將圖片標準化到 [-1, 1] 區間內

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# 批量化和打亂數據
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

"""## 創建模型

生成器和判別器均使用 [Keras Sequential API](https://tensorflow.google.cn/guide/keras#sequential_model) 定義。

### 生成器

生成器使用 `tf.keras.layers.Conv2DTranspose` （上採樣）層來從種子（隨機雜訊）中產生圖片。
以一個使用該種子作為輸入的 `Dense` 層開始，然後多次上採樣直到達到所期望的 28x28x1 的圖片尺寸。
注意除了輸出層使用 tanh 之外，其他每層均使用 `tf.keras.layers.LeakyReLU` 作為啟動函數。
"""

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # 注意：batch size 沒有限制

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

"""使用（尚未訓練的）生成器創建一張圖片。"""

generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')

"""### 判別器

判別器是一個基於 CNN 的圖片分類器。
"""

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

"""使用（尚未訓練的）判別器來對圖片的真偽進行判斷。模型將被訓練為為真實圖片輸出正值，為偽造圖片輸出負值。"""

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

"""## 定義損失函數和優化器

為兩個模型定義損失函數和優化器。
"""

# 該方法返回計算交叉熵損失的輔助函數
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

"""### 判別器損失

該方法量化判別器從判斷真偽圖片的能力。
它將判別器對真實圖片的預測值與值全為 1 的陣列進行對比，
將判別器對偽造（生成的）圖片的預測值與值全為 0 的陣列進行對比。
"""

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

"""### 生成器損失
生成器損失量化其欺騙判別器的能力。
直觀來講，如果生成器表現良好，判別器將會把偽造圖片判斷為真實圖片（或 1）。
這裡我們將把判別器在生成圖片上的判斷結果與一個值全為 1 的陣列進行對比。
"""

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

"""由於我們需要分別訓練兩個網路，判別器和生成器的優化器是不同的。"""

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

"""### 保存檢查點

本筆記還演示了如何保存和恢復模型，這在長時間訓練任務被中斷的情況下比較有説明。
"""

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

"""## 定義訓練迴圈"""

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16


# 我們將重複使用該種子（因此在動畫 GIF 中更容易視覺化進度）
seed = tf.random.normal([num_examples_to_generate, noise_dim])

"""
訓練迴圈在生成器接收到一個隨機種子作為輸入時開始。
該種子用於生產一張圖片。判別器隨後被用於區分真實圖片（選自訓練集）和偽造圖片（由生成器生成）。
針對這裡的每一個模型都計算損失函數，並且計算梯度用於更新生成器與判別器。
"""

# 注意 `tf.function` 的使用
# 該注解使函數被“編譯”
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # 繼續進行時為 GIF 生成圖像
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # 每 15 個 epoch 保存一次模型
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # 最後一個 epoch 結束後生成圖片
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

"""**生成與保存圖片**"""

def generate_and_save_images(model, epoch, test_input):
  # 注意 training` 設定為 False
  # 因此，所有層都在推理模式下運行（batchnorm）。
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

"""## 訓練模型
調用上面定義的 `train()` 方法來同時訓練生成器和判別器。
注意，訓練 GANs 可能是棘手的。
重要的是，生成器和判別器不能夠互相壓制對方（例如，他們以相似的學習率訓練）。

在訓練之初，生成的圖片看起來像是隨機雜訊。
隨著訓練過程的進行，生成的數位將越來越真實。
在大概 50 個 epoch 之後，這些圖片看起來像是 MNIST 數位。
使用 Colab 中的預設設置可能需要大約 1 分鐘每 epoch。
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# train(train_dataset, EPOCHS)

"""恢復最新的檢查點。"""

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

"""## 創建 GIF"""

# 使用 epoch 數生成單張圖片
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)

"""使用訓練過程中生成的圖片通過 `imageio` 生成動態 gif"""

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

import IPython
if IPython.version_info > (6,2,0,''):
  display.Image(filename=anim_file)

"""如果您正在使用 Colab，您可以通過如下代碼下載動畫："""

try:
  from google.colab import files
except ImportError:
   pass
else:
  files.download(anim_file)
```
