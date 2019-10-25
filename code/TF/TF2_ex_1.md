#

```
%tensorflow_version 2.x
```
```
Hands-On Neural Networks with TensorFlow 2.0
Paolo Galeone
September 18, 2019
CH4
```
###
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


###
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


###
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
###
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

```



```


###  sequential-complete.py


```
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


def make_model(n_classes):
    return tf.keras.Sequential([
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


def load_data():
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    # Scale input in [-1, 1] range
    train_x = tf.expand_dims(train_x, -1)
    train_x = (tf.image.convert_image_dtype(train_x, tf.float32) - 0.5) * 2
    train_y = tf.expand_dims(train_y, -1)

    test_x = test_x / 255. * 2 - 1
    test_x = (tf.image.convert_image_dtype(test_x, tf.float32) - 0.5) * 2
    test_y = tf.expand_dims(test_y, -1)

    return (train_x, train_y), (test_x, test_y)


def train():
    # Define the model
    n_classes = 10
    model = make_model(n_classes)

    # Input data
    (train_x, train_y), (test_x, test_y) = load_data()

    # Training parameters
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    step = tf.Variable(1, name="global_step")
    optimizer = tf.optimizers.Adam(1e-3)

    ckpt = tf.train.Checkpoint(step=step, optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f"Restored from {manager.latest_checkpoint}")
    else:
        print("Initializing from scratch.")

    accuracy = tf.metrics.Accuracy()
    mean_loss = tf.metrics.Mean(name='loss')

    # Train step function
    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logits = model(inputs)
            loss_value = loss(labels, logits)

        gradients = tape.gradient(loss_value, model.trainable_variables)
        # TODO: apply gradient clipping here
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        step.assign_add(1)

        accuracy.update_state(labels, tf.argmax(logits, -1))
        return loss_value, accuracy.result()

    epochs = 10
    batch_size = 32
    nr_batches_train = int(train_x.shape[0] / batch_size)
    print(f"Batch size: {batch_size}")
    print(f"Number of batches per epoch: {nr_batches_train}")

    train_summary_writer = tf.summary.create_file_writer('./log/train')

    with train_summary_writer.as_default():
        for epoch in range(epochs):
            for t in range(nr_batches_train):
                start_from = t * batch_size
                to = (t + 1) * batch_size

                features, labels = train_x[start_from:to], train_y[start_from:
                                                                   to]

                loss_value, accuracy_value = train_step(features, labels)
                mean_loss.update_state(loss_value)

                if t % 10 == 0:
                    print(
                        f"{step.numpy()}: {loss_value} - accuracy: {accuracy_value}"
                    )
                    save_path = manager.save()
                    print(f"Checkpoint saved: {save_path}")
                    tf.summary.image(
                        'train_set', features, max_outputs=3, step=step.numpy())
                    tf.summary.scalar(
                        'accuracy', accuracy_value, step=step.numpy())
                    tf.summary.scalar(
                        'loss', mean_loss.result(), step=step.numpy())
                    accuracy.reset_states()
                    mean_loss.reset_states()
            print(f"Epoch {epoch} terminated")
            # Measuring accuracy on the whole training set at the end of epoch
            for t in range(nr_batches_train):
                start_from = t * batch_size
                to = (t + 1) * batch_size
                features, labels = train_x[start_from:to], train_y[start_from:
                                                                   to]
                logits = model(features)
                accuracy.update_state(labels, tf.argmax(logits, -1))
            print(f"Training accuracy: {accuracy.result()}")
            accuracy.reset_states()


if __name__ == "__main__":
    train()

```


```


```

### sequential-gradient-clip.py
```
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


def make_model(n_classes):
    return tf.keras.Sequential([
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


def load_data():
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    # Scale input in [-1, 1] range
    train_x = tf.expand_dims(train_x, -1)
    train_x = (tf.image.convert_image_dtype(train_x, tf.float32) - 0.5) * 2
    train_y = tf.expand_dims(train_y, -1)

    test_x = test_x / 255. * 2 - 1
    test_x = (tf.image.convert_image_dtype(test_x, tf.float32) - 0.5) * 2
    test_y = tf.expand_dims(test_y, -1)

    return (train_x, train_y), (test_x, test_y)


def train():
    # Define the model
    n_classes = 10
    model = make_model(n_classes)

    # Input data
    (train_x, train_y), (test_x, test_y) = load_data()

    # Training parameters
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    step = tf.Variable(1, name="global_step")
    optimizer = tf.optimizers.Adam(1e-3)
    accuracy = tf.metrics.Accuracy()

    # Train step function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logits = model(inputs)
            loss_value = loss(labels, logits)

        gradients = tape.gradient(loss_value, model.trainable_variables)
        # TODO: apply gradient clipping here
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        step.assign_add(1)

        accuracy_value = accuracy(labels, tf.argmax(logits, -1))
        return loss_value, accuracy_value

    epochs = 10
    batch_size = 32
    nr_batches_train = int(train_x.shape[0] / batch_size)
    print(f"Batch size: {batch_size}")
    print(f"Number of batches per epoch: {nr_batches_train}")

    for epoch in range(epochs):
        for t in range(nr_batches_train):
            start_from = t * batch_size
            to = (t + 1) * batch_size

            features, labels = train_x[start_from:to], train_y[start_from:to]

            loss_value, accuracy_value = train_step(features, labels)
            if t % 10 == 0:
                print(
                    f"{step.numpy()}: {loss_value} - accuracy: {accuracy_value}"
                )
        print(f"Epoch {epoch} terminated")


if __name__ == "__main__":
    train()


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


```
import tensorflow as tf

x = tf.constant(4.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.pow(x, 2)
# Will compute 8 = 2*x, x = 8
dy_dx = tape.gradient(y, x)
print(dy_dx)


```


```
import tensorflow as tf

x = tf.Variable(4.0)
y = tf.Variable(2.0)
with tf.GradientTape(persistent=True) as tape:
    z = x + y
    w = tf.pow(x, 2)
dz_dy = tape.gradient(z, y)
dz_dx = tape.gradient(z, x)
dw_dx = tape.gradient(w, x)
print(dz_dy, dz_dx, dw_dx)
# Release the resources
del tape


```

# CH5
```
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


def train_dataset(batch_size=32, num_epochs=1):
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    input_x, input_y = train_x, train_y

    def scale_fn(image, label):
        return (
            tf.image.convert_image_dtype(image, tf.float32) - 0.5) * 2.0, label

    dataset = tf.data.Dataset.from_tensor_slices((tf.expand_dims(
        input_x, -1), tf.expand_dims(input_y, -1))).map(scale_fn)

    dataset = dataset.cache().repeat(num_epochs)
    dataset = dataset.shuffle(batch_size)

    return dataset.batch(batch_size).prefetch(1)


def make_model(n_classes):
    return tf.keras.Sequential([
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


def train():
    # Define the model
    n_classes = 10
    model = make_model(n_classes)

    # Input data
    dataset = train_dataset(num_epochs=10)

    # Training parameters
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    step = tf.Variable(1, name="global_step")
    optimizer = tf.optimizers.Adam(1e-3)
    accuracy = tf.metrics.Accuracy()

    # Train step function
    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logits = model(inputs)
            loss_value = loss(labels, logits)

        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        step.assign_add(1)

        accuracy_value = accuracy(labels, tf.argmax(logits, -1))
        return loss_value, accuracy_value

    @tf.function
    def loop():
        for features, labels in dataset:
            loss_value, accuracy_value = train_step(features, labels)
            if tf.equal(tf.mod(step, 10), 0):
                tf.print(step, ": ", loss_value, " - accuracy: ",
                         accuracy_value)

    loop()


if __name__ == "__main__":
    train()


```


```
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices({
    "a":
    tf.random.uniform([4]),
    "b":
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)
})
for value in dataset:
    print(value["a"])


def noise():
    while True:
        yield tf.random.uniform((100,))


dataset = tf.data.Dataset.from_generator(noise, (tf.float32))
buffer_size = 10
batch_size = 32
dataset = dataset.map(lambda x: x + 10).shuffle(buffer_size).batch(batch_size)
for idx, noise in enumerate(dataset):
    if idx == 2:
        break
    print(idx)
    print(noise.shape)


```


```
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


def get_input_fn(mode, batch_size=32, num_epochs=1):
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    half = test_x.shape[0] // 2
    if mode == tf.estimator.ModeKeys.TRAIN:
        input_x, input_y = train_x, train_y
        train = True
    elif mode == tf.estimator.ModeKeys.EVAL:
        input_x, input_y = test_x[:half], test_y[:half]
        train = False
    elif mode == tf.estimator.ModeKeys.PREDICT:
        input_x, input_y = test_x[half:-1], test_y[half:-1]
        train = False
    else:
        raise ValueError("tf.estimator.ModeKeys required!")

    def scale_fn(image, label):
        return (
            (tf.image.convert_image_dtype(image, tf.float32) - 0.5) * 2.0,
            tf.cast(label, tf.int32),
        )

    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(
            (tf.expand_dims(input_x, -1), tf.expand_dims(input_y, -1))
        ).map(scale_fn)
        if train:
            dataset = dataset.shuffle(10).repeat(num_epochs)
        dataset = dataset.batch(batch_size).prefetch(1)
        return dataset

    return input_fn


def model_fn(features, labels, mode):
    v1 = tf.compat.v1
    model = make_model(10)
    logits = model(features)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Extract the predictions
        predictions = v1.argmax(logits, -1)
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = v1.reduce_mean(
        v1.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=v1.squeeze(labels)
        )
    )

    global_step = v1.train.get_global_step()

    # Compute evaluation metrics.
    accuracy = v1.metrics.accuracy(
        labels=labels, predictions=v1.argmax(logits, -1), name="accuracy"
    )
    # The metrics dictionary is used by the estimator during the evaluation
    metrics = {"accuracy": accuracy}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    if mode == tf.estimator.ModeKeys.TRAIN:
        opt = v1.train.AdamOptimizer(1e-4)
        train_op = opt.minimize(
            loss, var_list=model.trainable_variables, global_step=global_step
        )

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    raise NotImplementedError(f"Unknown mode {mode}")


print("Every log is on Tensorboard, please run tensorboard --logidr log")
estimator = tf.estimator.Estimator(model_fn, model_dir="log")
for epoch in range(50):
    print(f"Training for the {epoch}-th epoch")
    estimator.train(get_input_fn(tf.estimator.ModeKeys.TRAIN, num_epochs=1))
    print("Evaluating...")
    estimator.evaluate(get_input_fn(tf.estimator.ModeKeys.EVAL))

```


#### keras.py

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


def train_dataset(batch_size=32, num_epochs=1):
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    half = test_x.shape[0] // 2
    input_x, input_y = train_x, train_y

    def scale_fn(image, label):
        return (
            tf.image.convert_image_dtype(image, tf.float32) - 0.5) * 2.0, label

    dataset = tf.data.Dataset.from_tensor_slices((tf.expand_dims(
        input_x, -1), tf.expand_dims(input_y, -1))).map(scale_fn)

    dataset = dataset.cache().repeat(num_epochs)
    dataset = dataset.shuffle(batch_size)

    return dataset.batch(batch_size).prefetch(1)


model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_dataset(num_epochs=10))

```

### tfds-base.py
```
import tensorflow_datasets as tfds

# See available datasets
print(tfds.list_builders())
# Construct 2 tf.data.Dataset objects
# The training dataset and the test dataset
ds_train, ds_test = tfds.load(name="mnist", split=["train", "test"])
builder = tfds.builder("mnist")
print(builder.info)

```

# CH6
```
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


dataset, info = tfds.load("tf_flowers", with_info=True)
print(info)
dataset = dataset["train"]
tot = 3670

train_set_size = tot // 2
validation_set_size = tot - train_set_size - train_set_size // 2
test_set_size = tot - train_set_size - validation_set_size


print("train set size: ", train_set_size)
print("validation set size: ", validation_set_size)
print("test set size: ", test_set_size)

train, test, validation = (
    dataset.take(train_set_size),
    dataset.skip(train_set_size).take(validation_set_size),
    dataset.skip(train_set_size + validation_set_size).take(test_set_size),
)


def to_float_image(example):
    example["image"] = tf.image.convert_image_dtype(example["image"], tf.float32)
    return example


def resize(example):
    example["image"] = tf.image.resize(example["image"], (299, 299))
    return example


train = train.map(to_float_image).map(resize)
validation = validation.map(to_float_image).map(resize)
test = test.map(to_float_image).map(resize)

num_classes = 5

model = tf.keras.Sequential(
    [
        hub.KerasLayer(
            "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/2",
            output_shape=[2048],
            trainable=False,
        ),
        tf.keras.layers.Dense(512),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(num_classes),  # linear
    ]
)

# Training utilities
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
step = tf.Variable(1, name="global_step", trainable=False)
optimizer = tf.optimizers.Adam(1e-3)

train_summary_writer = tf.summary.create_file_writer("./log/transfer/train")
validation_summary_writer = tf.summary.create_file_writer("./log/transfer/validation")

# Metrics
accuracy = tf.metrics.Accuracy()
mean_loss = tf.metrics.Mean(name="loss")


@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        logits = model(inputs)
        loss_value = loss(labels, logits)

    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    step.assign_add(1)

    accuracy.update_state(labels, tf.argmax(logits, -1))
    return loss_value


# Configure the training set to use batches and prefetch
train = train.batch(32).prefetch(1)
validation = validation.batch(32).prefetch(1)
test = test.batch(32).prefetch(1)

num_epochs = 10
for epoch in range(num_epochs):

    for example in train:
        image, label = example["image"], example["label"]
        loss_value = train_step(image, label)
        mean_loss.update_state(loss_value)

        if tf.equal(tf.math.mod(step, 10), 0):
            tf.print(
                step, " loss: ", mean_loss.result(), " acccuracy: ", accuracy.result()
            )
            mean_loss.reset_states()
            accuracy.reset_states()

    # Epoch ended, measure performance on validation set
    tf.print("## VALIDATION - ", epoch)
    accuracy.reset_states()
    for example in validation:
        image, label = example["image"], example["label"]
        logits = model(image)
        accuracy.update_state(label, tf.argmax(logits, -1))
    tf.print("accuracy: ", accuracy.result())
    accuracy.reset_states()
```

# CH7
```
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

# Train, test, and validation are datasets for object detection: multiple objects per image.
(train, test, validation), info = tfds.load(
    "voc2007", split=["train", "test", "validation"], with_info=True
)

# Create a subset of the dataset by filtering the elements: we are interested
# in creating a dataset for object detetion and classification that is a dataset
# of images with a single object annotated.
def filter(dataset):
    return dataset.filter(lambda row: tf.equal(tf.shape(row["objects"]["label"])[0], 1))


train, test, validation = filter(train), filter(test), filter(validation)


# Input layer
inputs = tf.keras.layers.Input(shape=(299, 299, 3))

# Feature extractor
net = hub.KerasLayer(
    "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/2",
    output_shape=[2048],
    trainable=False,
)(inputs)

# Regression head
regression_head = tf.keras.layers.Dense(512)(net)
regression_head = tf.keras.layers.ReLU()(regression_head)
coordinates = tf.keras.layers.Dense(4, use_bias=False)(regression_head)

# Classification head
classification_head = tf.keras.layers.Dense(1024)(net)
classification_head = tf.keras.layers.ReLU()(classificatio_head)
classification_head = tf.keras.layers.Dense(128)(net)
classification_head = tf.keras.layers.ReLU()(classificatio_head)
num_classes = 20
classification_head = tf.keras.layers.Dense(num_classes, use_bias=False)(
    classification_head
)

model = tf.keras.Model(inputs=inputs, outputs=[coordinates, classification_head])


def prepare(dataset):
    def _fn(row):
        row["image"] = tf.image.convert_image_dtype(row["image"], tf.float32)
        row["image"] = tf.image.resize(row["image"], (299, 299))
        return row

    return dataset.map(_fn)


train, test, validation = prepare(train), prepare(test), prepare(validation)

# First option -> this requires to call the loss l2, taking care of squeezing the input
# l2 = tf.losses.MeanSquaredError()

# Second option, it is the loss function iself that squeezes the input
def l2(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - tf.squeeze(y_true, axis=[1])))


precision_metric = tf.metrics.Precision()


def iou(pred_box, gt_box, h, w):
    """
    Compute IoU between detect box and gt boxes
    Args:
        pred_box: shape (4, ):  y_min, x_min, y_max, x_max - predicted box
        gt_boxes: shape (n, 4): y_min, x_min, y_max, x_max - ground truth
        h: image height
        w: image width
    """

    # Transpose the coordinates from y_min, x_min, y_max, x_max
    # In absolute coordinates to x_min, y_min, x_max, y_max
    # in pixel coordinates
    def _swap(box):
        return tf.stack([box[1] * w, box[0] * h, box[3] * w, box[2] * h])

    pred_box = _swap(pred_box)
    gt_box = _swap(gt_box)

    box_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    xx1 = tf.maximum(pred_box[0], gt_box[0])
    yy1 = tf.maximum(pred_box[1], gt_box[1])
    xx2 = tf.minimum(pred_box[2], gt_box[2])
    yy2 = tf.minimum(pred_box[3], gt_box[3])

    # compute the width and height of the bounding box
    w = tf.maximum(0, xx2 - xx1)
    h = tf.maximum(0, yy2 - yy1)

    inter = w * h
    return inter / (box_area + area - inter)


threshold = 0.75


def draw(dataset, regressor, step):
    with tf.device("/CPU:0"):
        row = next(iter(dataset.take(3).batch(3)))
        images = row["image"]
        obj = row["objects"]
        boxes = regressor(images)

        images = tf.image.draw_bounding_boxes(
            images=images, boxes=tf.reshape(boxes, (-1, 1, 4))
        )
        images = tf.image.draw_bounding_boxes(
            images=images, boxes=tf.reshape(obj["bbox"], (-1, 1, 4))
        )
        tf.summary.image("images", images, step=step)

        true_labels, predicted_labels = [], []
        for idx, predicted_box in enumerate(boxes):
            iou_value = iou(predicted_box, tf.squeeze(obj["bbox"][idx]), 299, 299)
            true_labels.append(1)
            predicted_labels.append(1 if iou_value >= threshold else 0)

        precision_metric.update_state(true_labels, predicted_labels)
        tf.summary.scalar("precision", precision_metric.result(), step=step)
        tf.print(precision_metric.result())


optimizer = tf.optimizers.Adam()
epochs = 10
batch_size = 3

global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

train_writer, validation_writer = (
    tf.summary.create_file_writer("log/train"),
    tf.summary.create_file_writer("log/validation"),
)
with validation_writer.as_default():
    draw(validation, regressor, global_step)


@tf.function
def train_step(image, coordinates):
    with tf.GradientTape() as tape:
        loss = l2(coordinates, regressor(image))
    gradients = tape.gradient(loss, regressor.trainable_variables)
    optimizer.apply_gradients(zip(gradients, regressor.trainable_variables))
    return loss


train_batches = train.cache().batch(batch_size).prefetch(1)
with train_writer.as_default():
    for _ in tf.range(epochs):
        for batch in train_batches:
            obj = batch["objects"]
            coordinates = obj["bbox"]
            loss = train_step(batch["image"], coordinates)
            tf.summary.scalar("loss", loss, step=global_step)
            global_step.assign_add(1)
            if tf.equal(tf.mod(global_step, 10), 0):
                tf.print("step ", global_step, " loss: ", loss)
                with validation_writer.as_default():
                    draw(validation, regressor, global_step)
                with train_writer.as_default():
                    draw(train, regressor, global_step)
    # Clean the metrics at the end of every epoch
    precision_metric.reset()
```

# ch8
```
import tensorflow as tf
import tensorflow_datasets as tfds
import math
import os


def downsample(depth):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                depth, 3, strides=2, padding="same", kernel_initializer="he_normal"
            ),
            tf.keras.layers.LeakyReLU(),
        ]
    )


def upsample(depth):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2DTranspose(
                depth, 3, strides=2, padding="same", kernel_initializer="he_normal"
            ),
            tf.keras.layers.ReLU(),
        ]
    )

def get_unet(input_size=(256, 256, 3), num_classes=21):

    # Downsample from 256x256 to 4x4, while adding depth
    # using powers of 2, startin from 2**5. Cap to 512.
    encoders = []
    for i in range(2, int(math.log2(256))):
        depth = 2 ** (i + 5)
        if depth > 512:
            depth = 512
        encoders.append(downsample(depth=depth))

    # Upsample from 4x4 to 256x256, reducing the depth
    decoders = []
    for i in reversed(range(2, int(math.log2(256)))):
        depth = 2 ** (i + 5)
        if depth < 32:
            depth = 32
        if depth > 512:
            depth = 512
        decoders.append(upsample(depth=depth))

    # Build the model by invoking the encoder layers with the correct input
    inputs = tf.keras.layers.Input(input_size)
    concat = tf.keras.layers.Concatenate()

    x = inputs
    # Encoder: downsample loop
    skips = []
    for conv in encoders:
        x = conv(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Decoder: input + skip connection
    for deconv, skip in zip(decoders, skips):
        x = deconv(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    # Add the last layer on top and define the model
    last = tf.keras.layers.Conv2DTranspose(
        num_classes, 3, strides=2, padding="same", kernel_initializer="he_normal"
    )

    outputs = last(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

LUT = {
    (0, 0, 0): 0, # background
    (128, 0, 0): 1, # aeroplane
    (0, 128, 0): 2, # bicycle
    (128, 128, 0): 3, # bird
    (0, 0, 128): 4, # boat
    (128, 0, 128): 5, # bottle
    (0, 128, 128): 6, # bus
    (128, 128, 128): 7, # car
    (64, 0, 0): 8, # cat
    (192, 0, 0): 9, # chair
    (64, 128, 0): 10, # cow
    (192, 128, 0): 11, # diningtable
    (64, 0, 128): 12, # dog
    (192, 0, 128): 13, # horse
    (64, 128, 128): 14, # motorbike
    (192, 128, 128): 15, # person
    (0, 64, 0): 16, # pottedplant
    (128, 64, 0): 17, # sheep
    (0, 192, 0): 18, # sofa
    (128, 192, 0): 19, # train
    (0, 64, 128): 20, # tvmonitor
    (255, 255, 255): 21, # undefined / don't care
}


class Voc2007Semantic(tfds.image.Voc2007): 
    """Pasval VOC 2007 - semantic segmentation.""" 
 
    VERSION = tfds.core.Version("0.1.0")
    def _info(self):
        parent_info = tfds.image.Voc2007().info
        return tfds.core.DatasetInfo(
            builder=self,
            description=parent_info.description,
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(None, None, 3)),
                    "image/filename": tfds.features.Text(),
                    "label": tfds.features.Image(shape=(None, None, 1)),
                }
            ),
            urls=parent_info.urls,
            citation=parent_info.citation,
        )

      
    def _generate_examples(self, data_path, set_name):
        set_filepath = os.path.join(
            data_path,
            "VOCdevkit/VOC2007/ImageSets/Segmentation/{}.txt".format(set_name),
        )
        with tf.io.gfile.GFile(set_filepath, "r") as f:
            for line in f:
                image_id = line.strip()

                image_filepath = os.path.join(
                    data_path, "VOCdevkit", "VOC2007", "JPEGImages", f"{image_id}.jpg"
                )
                label_filepath = os.path.join(
                    data_path,
                    "VOCdevkit",
                    "VOC2007",
                    "SegmentationClass",
                    f"{image_id}.png",
                )

                if not tf.io.gfile.exists(label_filepath):
                    continue

                label_rgb = tf.image.decode_image(
                    tf.io.read_file(label_filepath), channels=3
                )

                label = tf.Variable(
                    tf.expand_dims(
                        tf.zeros(shape=tf.shape(label_rgb)[:-1], dtype=tf.uint8), -1
                    )
                )

                for color, label_id in LUT.items():
                    match = tf.reduce_all(tf.equal(label_rgb, color), axis=[2])
                    labeled = tf.expand_dims(tf.cast(match, tf.uint8), axis=-1)
                    label.assign_add(labeled * label_id)

                colored = tf.not_equal(tf.reduce_sum(label), tf.constant(0, tf.uint8))
                # Certain labels have wrong RGB values
                if not colored.numpy():
                    tf.print("error parsing: ", label_filepath)
                    continue
                
                yield {
                    # Declaring in _info "image" as a tfds.feature.Image
                    # we can use both an image or a string. If a string is detected
                    # it is supposed to be the image path and tfds take care of the
                    # reading process.
                    "image": image_filepath,
                    "image/filename": f"{image_id}.jpg",
                    "label": label.numpy(),
                }

print(tfds.list_builders())
dataset, info = tfds.load("voc2007_semantic", with_info=True)

train_set = dataset["train"]

def resize_and_scale(row):
    # Resize and convert to float, [0,1] range
    row["image"] = tf.image.convert_image_dtype(
        tf.image.resize(
            row["image"],
            (256,256),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
        tf.float32)
    # Resize, cast to int64 since it is a supported label type
    row["label"] = tf.cast(
        tf.image.resize(
            row["label"],
            (256,256),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
        tf.int64)
    return row
  
def to_pair(row):
    return row["image"], row["label"]
batch_size= 32

train_set = train_set.map(resize_and_scale).map(to_pair)
train_set = train_set.batch(batch_size).prefetch(1)

validation_set = dataset["validation"].map(resize_and_scale)
validation_set = validation_set.map(to_pair).batch(batch_size)

model = get_unet()

optimizer = tf.optimizers.Adam()

checkpoint_path = "ckpt/pb.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
tensorboard = tf.keras.callbacks.TensorBoard(write_images=True)
model.compile(optimizer=optimizer,
              #loss=lambda y_true, y_pred: tf.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred) + tf.losses.MeanAbsoluteError()(y_true, y_pred),
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])#, tf.metrics.MeanIoU(num_classes=21)])


num_epochs = 50
model.fit(train_set, validation_data=validation_set, epochs=num_epochs,
          callbacks=[cp_callback, tensorboard])


sample = tf.image.decode_jpeg(tf.io.read_file("me.jpg"))
sample = tf.expand_dims(tf.image.convert_image_dtype(sample, tf.float32), axis=[0])
sample = tf.image.resize(sample, (512,512))
pred_image = tf.squeeze(tf.argmax(model(sample), axis=-1), axis=[0])

REV_LUT = {value: key for key, value in LUT.items()}

color_image = tf.Variable(tf.zeros((512,512,3), dtype=tf.uint8))
pixels_per_label = []
for label, color in REV_LUT.items():
    match = tf.equal(pred_image, label)
    labeled = tf.expand_dims(tf.cast(match, tf.uint8), axis=-1)
    pixels_per_label.append((label, tf.math.count_nonzero(labeled)))
    labeled = tf.tile(labeled, [1,1,3])
    color_image.assign_add(labeled * color)


for label, count in pixels_per_label:
    print(label, ": ", count.numpy())

tf.io.write_file("seg.jpg", tf.io.encode_jpeg(color_image))

```

# ch9 gan

CGAN
```
import tensorflow as tf
import tensorflow_datasets as tfds

dataset, info = tfds.load("fashion_mnist", split="train", with_info=True)


def convert(row):
    image = tf.image.convert_image_dtype(row["image"], tf.float32)
    label = tf.expand_dims(tf.cast(row["label"], tf.float32), axis=-1)
    return image, label


batch_size = 32
dataset = dataset.map(convert).batch(batch_size).prefetch(1)


def get_generator(latent_dimension):

    # Condition subnetwork: encode the condition in a hidden representation
    condition = tf.keras.layers.Input((1,))
    net = tf.keras.layers.Dense(32, activation=tf.nn.elu)(condition)
    net = tf.keras.layers.Dense(64, activation=tf.nn.elu)(net)

    # Concatenate the hidden condition representation to noise and upsample
    noise = tf.keras.layers.Input(latent_dimension)
    inputs = tf.keras.layers.Concatenate()([noise, net])

    # Convert inputs from (batch_size, latent_dimension + 1)
    # To a 4-D tensor, that can be used with convolutions
    inputs = tf.keras.layers.Reshape((1, 1, inputs.shape[-1]))(inputs)

    depth = 128
    kernel_size = 5
    net = tf.keras.layers.Conv2DTranspose(
        depth, kernel_size, padding="valid", strides=1, activation=tf.nn.relu
    )(
        inputs
    )  # 5x5
    net = tf.keras.layers.Conv2DTranspose(
        depth // 2, kernel_size, padding="valid", strides=2, activation=tf.nn.relu
    )(
        net
    )  # 13x13
    net = tf.keras.layers.Conv2DTranspose(
        depth // 4,
        kernel_size,
        padding="valid",
        strides=2,
        activation=tf.nn.relu,
        use_bias=False,
    )(
        net
    )  # 29x29
    # Standard convolution with a 2x2 kernel to obtain a 28x28x1 out
    # The output is a sigmoid, since the images are in the [0,1] range
    net = tf.keras.layers.Conv2D(
        1, 2, padding="valid", strides=1, activation=tf.nn.sigmoid, use_bias=False
    )(net)
    model = tf.keras.Model(inputs=[noise, condition], outputs=net)
    return model


latent_dimension = 100
G = get_generator(latent_dimension)


def get_discriminator():
    # Encoder subnetwork: feature extactor to get a feature vector
    image = tf.keras.layers.Input((28, 28, 1))
    depth = 32
    kernel_size = 3
    net = tf.keras.layers.Conv2D(
        depth, kernel_size, padding="same", strides=2, activation=tf.nn.relu
    )(
        image
    )  # 14x14x32
    net = tf.keras.layers.Conv2D(
        depth * 2, kernel_size, padding="same", strides=2, activation=tf.nn.relu
    )(
        net
    )  # 7x7x64

    net = tf.keras.layers.Conv2D(
        depth * 3, kernel_size, padding="same", strides=2, activation=tf.nn.relu
    )(
        net
    )  # 4x4x96

    feature_vector = tf.keras.layers.Flatten()(net)  # 4*4*96

    # Create a hidden representation of the condition
    condition = tf.keras.layers.Input((1,))
    hidden = tf.keras.layers.Dense(32, activation=tf.nn.elu)(condition)
    hidden = tf.keras.layers.Dense(64, activation=tf.nn.elu)(hidden)

    # Concatenate the feature vector and the hidden label representatio
    out = tf.keras.layers.Concatenate()([feature_vector, hidden])

    # Add the final classification layers with a single linear neuron
    out = tf.keras.layers.Dense(128, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dense(1)(out)

    model = tf.keras.Model(inputs=[image, condition], outputs=out)
    return model


D = get_discriminator()

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def d_loss(d_real, d_fake):
    """The disciminator loss function."""
    return bce(tf.ones_like(d_real), d_real) + bce(tf.zeros_like(d_fake), d_fake)


def g_loss(generated_output):
    """The Generator loss function."""
    return bce(tf.ones_like(generated_output), generated_output)


# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt

# %matplotlib inline


def train():
    # Define the optimizers and the train operations
    optimizer = tf.keras.optimizers.Adam(1e-5)

    @tf.function
    def train_step(image, label):
        with tf.GradientTape(persistent=True) as tape:
            noise_vector = tf.random.normal(
                mean=0, stddev=1, shape=(image.shape[0], latent_dimension)
            )
            # Sample from the Generator
            fake_data = G([noise_vector, label])
            # Compute the D loss
            d_fake_data = D([fake_data, label])
            d_real_data = D([image, label])

            d_loss_value = d_loss(d_real_data, d_fake_data)
            # Compute the G loss
            g_loss_value = g_loss(d_fake_data)
        # Now that we comptuted the losses we can compute the gradient (using the tape)
        # and optimize the networks
        d_gradients = tape.gradient(d_loss_value, D.trainable_variables)
        g_gradients = tape.gradient(g_loss_value, G.trainable_variables)
        # Deletng the tape, since we defined it as persistent (because we used it twice)
        del tape

        optimizer.apply_gradients(zip(d_gradients, D.trainable_variables))
        optimizer.apply_gradients(zip(g_gradients, G.trainable_variables))
        return g_loss_value, d_loss_value, fake_data[0], label[0]

    epochs = 50
    for epoch in range(epochs):
        for image, label in dataset:
            g_loss_value, d_loss_value, generated, condition = train_step(image, label)

        print("epoch ", epoch, "complete")
        print("loss:", g_loss_value, "d_loss: ", d_loss_value)
        print(
            "condition ",
            info.features["label"].int2str(
                tf.squeeze(tf.cast(condition, tf.int32)).numpy()
            ),
        )
        plt.imshow(tf.squeeze(generated).numpy(), cmap="gray")
        plt.show()


train()

```
gan
```
import tensorflow as tf
import matplotlib.pyplot as plt


def sample_dataset():
    dataset_shape = (2000, 1)
    return tf.random.normal(mean=10., shape=dataset_shape, stddev=0.1, dtype=tf.float32)


counts, bin, ignored = plt.hist(sample_dataset().numpy(), 100)
axes = plt.gca()
axes.set_xlim([-1,11])
axes.set_ylim([0, 60])
plt.show()

def generator(input_shape):
    """Defines the generator keras.Model.
    Args:
        input_shape: the desired input shape (e.g.: (latent_space_size))
    Returns:
        G: The generator model
    """
    inputs = tf.keras.layers.Input(input_shape)
    net = tf.keras.layers.Dense(units=64, activation=tf.nn.elu, name="fc1")(inputs)
    net = tf.keras.layers.Dense(units=64, activation=tf.nn.elu, name="fc2")(net)
    net = tf.keras.layers.Dense(units=1, name="G")(net)
    G = tf.keras.Model(inputs=inputs, outputs=net)
    return G

def disciminator(input_shape):
    """Defines the Discriminator keras.Model.
    Args:
        input_shape: the desired input shape (e.g.: (the generator output shape))
    Returns:
        D: the Discriminator model
    """
    inputs = tf.keras.layers.Input(input_shape)
    net = tf.keras.layers.Dense(units=32, activation=tf.nn.elu, name="fc1")(inputs)
    net = tf.keras.layers.Dense(units=1, name="D")(net)
    D = tf.keras.Model(inputs=inputs, outputs=net)
    return D

# Define the real input shape
input_shape = (1,)

# Define the Discriminator model
D = disciminator(input_shape)

# Arbitrary set the shape of the noise prior
latent_space_shape = (100,)
# Define the input noise shape and define the generator
G = generator(latent_space_shape)

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def d_loss(d_real, d_fake):
    """The disciminator loss function."""
    return bce(tf.ones_like(d_real), d_real) + bce(tf.zeros_like(d_fake), d_fake)

def train():
    # Define the optimizers and the train operations
    optimizer = tf.keras.optimizers.Adam(1e-5)
    
    @tf.function
    def train_step():
        with tf.GradientTape(persistent=True) as tape:
            real_data = sample_dataset()
            noise_vector = tf.random.normal(
                mean=0, stddev=1,
                shape=(real_data.shape[0], latent_space_shape[0]))
            # Sample from the Generator
            fake_data = G(noise_vector)
            # Compute the D loss
            d_fake_data = D(fake_data)
            d_real_data = D(real_data)
            d_loss_value = d_loss(d_real_data, d_fake_data)
            # Compute the G loss
            g_loss_value = g_loss(d_fake_data)
        # Now that we comptuted the losses we can compute the gradient
        # and optimize the networks
        d_gradients = tape.gradient(d_loss_value, D.trainable_variables)
        g_gradients = tape.gradient(g_loss_value, G.trainable_variables)
        # Deletng the tape, since we defined it as persistent
        # (because we used it twice)
        del tape
        
        optimizer.apply_gradients(zip(d_gradients, D.trainable_variables))
        optimizer.apply_gradients(zip(g_gradients, G.trainable_variables))
        return real_data, fake_data, g_loss_value, d_loss_value
    

    fig, ax = plt.subplots()
    for step in range(40000):
        real_data, fake_data,g_loss_value, d_loss_value = train_step()
        if step % 200 == 0:
            print("G loss: ", g_loss_value.numpy(), " D loss: ", d_loss_value.numpy(), " step: ", step)

            # Sample 5000 values from the Generator and draw the histogram
            ax.hist(fake_data.numpy(), 100)
            ax.hist(real_data.numpy(), 100)
            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            # place a text box in upper left in axes coords
            textstr = f"step={step}"
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)

            axes = plt.gca()
            axes.set_xlim([-1,11])
            axes.set_ylim([0, 60])
            display.display(pl.gcf())
            display.clear_output(wait=True)
            plt.gca().clear()
            
train()            



```
