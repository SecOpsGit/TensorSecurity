#
```

```
### IMDb-Movie-Review
```

```
### Sentiment Analysis on IMDb
```
https://paperswithcode.com/sota/sentiment-analysis-on-imdb
```

###
```
%tensorflow_version 2.x 
```

```
import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 10000
# truncate and pad input sequences
max_review_length = 80
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=top_words)
# X_train = tf.convert_to_tensor(X_train)
# y_train = tf.one_hot(y_train, depth=2)
print('Pad sequences (samples x time)')

x_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_review_length)
x_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_review_length)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


class RNN(keras.Model):

    def __init__(self, units, num_classes, num_layers):
        super(RNN, self).__init__()

        # self.cells = [keras.layers.LSTMCell(units) for _ in range(num_layers)]
        #
        # self.rnn = keras.layers.RNN(self.cells, unroll=True)
        self.rnn = keras.layers.LSTM(units, return_sequences=True)
        self.rnn2 = keras.layers.LSTM(units)

        # self.cells = (keras.layers.LSTMCell(units) for _ in range(num_layers))
        # #
        # self.rnn = keras.layers.RNN(self.cells, return_sequences=True, return_state=True)
        # self.rnn = keras.layers.LSTM(units, unroll=True)
        # self.rnn = keras.layers.StackedRNNCells(self.cells)

        # have 1000 words totally, every word will be embedding into 100 length vector
        # the max sentence lenght is 80 words
        self.embedding = keras.layers.Embedding(top_words, 100, input_length=max_review_length)
        self.fc = keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):

        # print('x', inputs.shape)
        # [b, sentence len] => [b, sentence len, word embedding]
        x = self.embedding(inputs)
        # print('embedding', x.shape)
        x = self.rnn(x) 
        x = self.rnn2(x) 
        # print('rnn', x.shape)

        x = self.fc(x)
        print(x.shape)

        return x


def main():

    units = 64
    num_classes = 2
    batch_size = 32
    epochs = 20

    model = RNN(units, num_classes, num_layers=2)

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # train
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test), verbose=1)

    # evaluate on test set
    scores = model.evaluate(x_test, y_test, batch_size, verbose=1)
    print("Final test loss and accuracy :", scores)


if __name__ == '__main__':
    main()
```
```
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
17465344/17464789 [==============================] - 0s 0us/step
Pad sequences (samples x time)
x_train shape: (25000, 80)
x_test shape: (25000, 80)
(None, 1)
Train on 25000 samples, validate on 25000 samples
Epoch 1/20
(None, 1)
(None, 1)
24992/25000 [============================>.] - ETA: 0s - loss: 0.4208 - accuracy: 0.7933(None, 1)
25000/25000 [==============================] - 121s 5ms/sample - loss: 0.4209 - accuracy: 0.7932 - val_loss: 0.3735 - val_accuracy: 0.8202
Epoch 2/20
25000/25000 [==============================] - 120s 5ms/sample - loss: 0.2774 - accuracy: 0.8816 - val_loss: 0.3649 - val_accuracy: 0.8379
Epoch 3/20
25000/25000 [==============================] - 120s 5ms/sample - loss: 0.1949 - accuracy: 0.9213 - val_loss: 0.4539 - val_accuracy: 0.8292
Epoch 4/20
25000/25000 [==============================] - 120s 5ms/sample - loss: 0.1317 - accuracy: 0.9477 - val_loss: 0.4900 - val_accuracy: 0.8291
Epoch 5/20


```
