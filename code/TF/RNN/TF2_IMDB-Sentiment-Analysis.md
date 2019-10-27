# 情緒分析sentiment analysis
```
Sentiment analysis (opinion mining or emotion AI) 
https://en.wikipedia.org/wiki/Sentiment_analysis


很多線上社群網站會蒐集使用者的資料，並且分析使用者行為，像是知名的Facebook在前幾年開始做「情緒分析(sentiment analysis)」，
它是以文字分析、自然語言處理NLP的方法，找出使用者的評價、情緒，進而預測出使用者行為來進行商業決策，
像這樣一連串利用情緒分析帶來的商業價值是相當可觀的。
```

```
陳宜欣/大數據下的情緒分析
https://www.slideshare.net/tw_dsconf/ss-64076883
```
```
基於文件為基礎的情緒分類(Document-based sentiment classification)
以主觀的概念做情緒分析(Subjectivity and sentiment classification)
以外觀(屬性)為基礎的情緒分析(Aspect‐based sentiment analysis)
建立情緒字彙的情緒分析(Lexicon‐based sentiment analysis)
```
```
wiki:https://zh.wikipedia.org/wiki/%E6%96%87%E6%9C%AC%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90

2008 Survey Article - Opinion mining and sentiment analysis (Pang & Lee)
     http://www.cs.cornell.edu/home/llee/omsa/omsa.pdf
2011 Survey Article - Comprehensive Review Of Opinion Summarization (Kim et al)
2013 Survey Article - New Avenues in Opinion Mining and Sentiment Analysis (Cambria et al)
https://web.archive.org/web/20141213014514/http://kavita-ganesan.com/sites/default/files/survey_opinionSummarization.pdf

```
```
[2015]網路美食評論情緒分析之研究.  Online Gastronomy Review Based on Sentiment Analysis
https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/login?o=dnclcdr&s=id=%22103NKHC0255011%22.&searchmode=basic

[2014]使用情緒分析於圖書館使用者滿意度評估之研究
A Study on Library Users’ Satisfaction Evaluation Using Sentimental Analysis
http://lac3.glis.ntnu.edu.tw/vj-attachment/2014/03/attach146.pdf

[繁體中文/NLP] 從word2vec到 情感分析
https://studentcodebank.wordpress.com/2019/02/22/%E7%B9%81%E9%AB%94%E4%B8%AD%E6%96%87-nlp-%E5%BE%9Eword2vec%E5%88%B0-%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90/
https://chunshan-theta.github.io/NLPLab/
```
### IMDb-Movie-Review
```
IMDb網路電影影評資料集
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
```

```
