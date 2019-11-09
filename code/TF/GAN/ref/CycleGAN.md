#
```

```

```

```

##
```
import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras



class Encoder(keras.Model):

    def __init__(self):
        super(Encoder, self).__init__()

        # Small variance in initialization helps with preventing colour inversion.
        self.conv1 = keras.layers.Conv2D(32, kernel_size=7, strides=1,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv3 = keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        
        # TODO: replace Instance Normalization for batchnorm
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()

    def call(self, inputs, training=True):
        x = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        # Implement instance norm to more closely match orig. paper (momentum=0.1)?
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)

        return x


class Residual(keras.Model):

    def __init__(self):
        super(Residual, self).__init__()

        self.conv1 = keras.layers.Conv2D(128, kernel_size=3, strides=1,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2D(128, kernel_size=3, strides=1,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        

        # TODO: replace Instance Normalization for batchnorm
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization() 

    def call(self, inputs, training=True):
        x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = tf.add(x, inputs)

        return x


class Decoder(keras.Model):

    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same',
                                                     kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same',
                                                     kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv3 = keras.layers.Conv2D(3, kernel_size=7, strides=1,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))


        # TODO: replace Instance Normalization for batchnorm
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.tanh(x)

        return x


class Generator(keras.Model):

    def __init__(self, img_size=256, skip=False):
        super(Generator, self).__init__()

        self.img_size = img_size
        self.skip = skip  # TODO: Add skip

        self.encoder = Encoder()
        if (img_size == 128):
            self.res1 = Residual()
            self.res2 = Residual()
            self.res3 = Residual()
            self.res4 = Residual()
            self.res5 = Residual()
            self.res6 = Residual()
        else:
            self.res1 = Residual()
            self.res2 = Residual()
            self.res3 = Residual()
            self.res4 = Residual()
            self.res5 = Residual()
            self.res6 = Residual()
            self.res7 = Residual()
            self.res8 = Residual()
            self.res9 = Residual()
        self.decoder = Decoder()


    def call(self, inputs, training=True):

        x = self.encoder(inputs, training)
        if (self.img_size == 128):
            x = self.res1(x, training)
            x = self.res2(x, training)
            x = self.res3(x, training)
            x = self.res4(x, training)
            x = self.res5(x, training)
            x = self.res6(x, training)
        else:
            x = self.res1(x, training)
            x = self.res2(x, training)
            x = self.res3(x, training)
            x = self.res4(x, training)
            x = self.res5(x, training)
            x = self.res6(x, training)
            x = self.res7(x, training)
            x = self.res8(x, training)
            x = self.res9(x, training)
        x = self.decoder(x, training)

        return x


class Discriminator(keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv2 = keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv3 = keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv4 = keras.layers.Conv2D(512, kernel_size=4, strides=1, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv5 = keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.leaky = keras.layers.LeakyReLU(0.2)


        # TODO: replace Instance Normalization for batchnorm
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()


    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.leaky(x)

        x = self.conv2(x)
        x = self.bn1(x, training=training)
        x = self.leaky(x)

        x = self.conv3(x)
        x = self.bn2(x, training=training)
        x = self.leaky(x)

        x = self.conv4(x)
        x = self.bn3(x, training=training)
        x = self.leaky(x)

        x = self.conv5(x)
        # x = tf.nn.sigmoid(x) # use_sigmoid = not lsgan
        return x






def discriminator_loss(disc_of_real_output, disc_of_gen_output, lsgan=True):
    if lsgan:  # Use least squares loss
        # real_loss = tf.reduce_mean(tf.squared_difference(disc_of_real_output, 1))
        real_loss = keras.losses.mean_squared_error(disc_of_real_output, tf.ones_like(disc_of_real_output))
        generated_loss = tf.reduce_mean(tf.square(disc_of_gen_output))

        total_disc_loss = (real_loss + generated_loss) * 0.5  # 0.5 slows down rate that D learns compared to G
    else:  # Use vanilla GAN loss
        raise NotImplementedError
        real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(disc_of_real_output),
                                                    logits=disc_of_real_output)
        generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(disc_of_gen_output),
                                                         logits=disc_of_gen_output)

        total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generator_loss(disc_of_gen_output, lsgan=True):
    if lsgan:  # Use least squares loss
        # gen_loss = tf.reduce_mean(tf.squared_difference(disc_of_gen_output, 1))
        gen_loss = keras.losses.mean_squared_error(disc_of_gen_output, tf.ones_like(disc_of_gen_output))
    else:  # Use vanilla GAN loss
        raise NotImplementedError
        gen_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(disc_generated_output),
                                                   logits=disc_generated_output)
        # l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) # Look up pix2pix loss
    return gen_loss


def cycle_consistency_loss(data_A, data_B, reconstructed_data_A, reconstructed_data_B, cyc_lambda=10):
    loss = tf.reduce_mean(tf.abs(data_A - reconstructed_data_A) + tf.abs(data_B - reconstructed_data_B))
    return cyc_lambda * loss
```

```
import  os
import  time
import  numpy as np
import  matplotlib.pyplot as plt
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras


#from    model import Generator, Discriminator, cycle_consistency_loss, generator_loss, discriminator_loss



tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


learning_rate = 0.0002
batch_size = 1  # Set batch size to 4 or 16 if training multigpu
img_size = 256
cyc_lambda = 10
epochs = 1000

"""### Load Datasets"""

path_to_zip = keras.utils.get_file('horse2zebra.zip',
                              cache_subdir=os.path.abspath('.'),
                              origin='https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip',
                              extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'horse2zebra/')

trainA_path = os.path.join(PATH, "trainA")
trainB_path = os.path.join(PATH, "trainB")

trainA_size = len(os.listdir(trainA_path))
trainB_size = len(os.listdir(trainB_path))
print('train A:', trainA_size)
print('train B:', trainB_size)


def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    # scale alread!
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [256, 256])
    # image = (image / 127.5) - 1
    image = image * 2 - 1

    return image


train_datasetA = tf.data.Dataset.list_files(PATH + 'trainA/*.jpg', shuffle=False)
train_datasetA = train_datasetA.shuffle(trainA_size).repeat(epochs)
train_datasetA = train_datasetA.map(lambda x: load_image(x))
train_datasetA = train_datasetA.batch(batch_size)
train_datasetA = train_datasetA.prefetch(batch_size)
train_datasetA = iter(train_datasetA)

train_datasetB = tf.data.Dataset.list_files(PATH + 'trainB/*.jpg', shuffle=False)
train_datasetB = train_datasetB.shuffle(trainB_size).repeat(epochs)
train_datasetB = train_datasetB.map(lambda x: load_image(x))
train_datasetB = train_datasetB.batch(batch_size)
train_datasetB = train_datasetB.prefetch(batch_size)
train_datasetB = iter(train_datasetB)

a = next(train_datasetA)
print('img shape:', a.shape, a.numpy().min(), a.numpy().max())

discA = Discriminator()
discB = Discriminator()
genA2B = Generator()
genB2A = Generator()

discA_optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.5)
discB_optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.5)
genA2B_optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.5)
genB2A_optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.5)


def generate_images(A, B, B2A, A2B, epoch):
    """
    :param A:
    :param B:
    :param B2A:
    :param A2B:
    :param epoch:
    :return:
    """
    plt.figure(figsize=(15, 15))
    A = tf.reshape(A, [256, 256, 3]).numpy()
    B = tf.reshape(B, [256, 256, 3]).numpy()
    B2A = tf.reshape(B2A, [256, 256, 3]).numpy()
    A2B = tf.reshape(A2B, [256, 256, 3]).numpy()
    display_list = [A, B, A2B, B2A]

    title = ['A', 'B', 'A2B', 'B2A']
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig('images/generated_%d.png'%epoch)
    plt.close()


def train(train_datasetA, train_datasetB, epochs, lsgan=True, cyc_lambda=10):

    for epoch in range(epochs):
        start = time.time()


        with tf.GradientTape() as genA2B_tape, tf.GradientTape() as genB2A_tape, \
                tf.GradientTape() as discA_tape, tf.GradientTape() as discB_tape:
            try:
                # Next training minibatches, default size 1
                trainA = next(train_datasetA)
                trainB = next(train_datasetB)
            except tf.errors.OutOfRangeError:
                print("Error, run out of data")
                break

            genA2B_output = genA2B(trainA, training=True)
            genB2A_output = genB2A(trainB, training=True)


            discA_real_output = discA(trainA, training=True)
            discB_real_output = discB(trainB, training=True)

            discA_fake_output = discA(genB2A_output, training=True)
            discB_fake_output = discB(genA2B_output, training=True)

            reconstructedA = genB2A(genA2B_output, training=True)
            reconstructedB = genA2B(genB2A_output, training=True)

            # generate_images(reconstructedA, reconstructedB)

            # Use history buffer of 50 for disc loss
            discA_loss = discriminator_loss(discA_real_output, discA_fake_output, lsgan=lsgan)
            discB_loss = discriminator_loss(discB_real_output, discB_fake_output, lsgan=lsgan)

            genA2B_loss = generator_loss(discB_fake_output, lsgan=lsgan) + \
                          cycle_consistency_loss(trainA, trainB, reconstructedA, reconstructedB,
                                                 cyc_lambda=cyc_lambda)
            genB2A_loss = generator_loss(discA_fake_output, lsgan=lsgan) + \
                          cycle_consistency_loss(trainA, trainB, reconstructedA, reconstructedB,
                                                 cyc_lambda=cyc_lambda)

        genA2B_gradients = genA2B_tape.gradient(genA2B_loss, genA2B.trainable_variables)
        genB2A_gradients = genB2A_tape.gradient(genB2A_loss, genB2A.trainable_variables)

        discA_gradients = discA_tape.gradient(discA_loss, discA.trainable_variables)
        discB_gradients = discB_tape.gradient(discB_loss, discB.trainable_variables)

        genA2B_optimizer.apply_gradients(zip(genA2B_gradients, genA2B.trainable_variables))
        genB2A_optimizer.apply_gradients(zip(genB2A_gradients, genB2A.trainable_variables))

        discA_optimizer.apply_gradients(zip(discA_gradients, discA.trainable_variables))
        discB_optimizer.apply_gradients(zip(discB_gradients, discB.trainable_variables))

        if epoch % 40 == 0:
            generate_images(trainA, trainB, genB2A_output, genA2B_output, epoch)

            print('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start))



if __name__ == '__main__':
    train(train_datasetA, train_datasetB, epochs=epochs, lsgan=True, cyc_lambda=cyc_lambda)
```
