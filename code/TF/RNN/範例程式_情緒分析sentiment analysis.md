# 情緒分析sentiment analysis
```
Sentiment analysis (opinion mining or emotion AI) 
https://en.wikipedia.org/wiki/Sentiment_analysis


很多線上社群網站會蒐集使用者的資料，並且分析使用者行為，
像是知名的Facebook在前幾年開始做「情緒分析(sentiment analysis)」，
它是以文字分析、自然語言處理NLP的方法，找出使用者的評價、情緒，
進而預測出使用者行為來進行商業決策，
像這樣一連串利用情緒分析帶來的商業價值是相當可觀的。
```
### 許多應用

```
陳宜欣/大數據下的情緒分析
https://www.slideshare.net/tw_dsconf/ss-64076883
```

```
[2015]網路美食評論情緒分析之研究.  
Online Gastronomy Review Based on Sentiment Analysis
https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/login?o=dnclcdr&s=id=%22103NKHC0255011%22.&searchmode=basic

[2014]使用情緒分析於圖書館使用者滿意度評估之研究
A Study on Library Users’ Satisfaction Evaluation Using Sentimental Analysis
http://lac3.glis.ntnu.edu.tw/vj-attachment/2014/03/attach146.pdf

[繁體中文/NLP] 從word2vec到 情感分析
https://studentcodebank.wordpress.com/2019/02/22/%E7%B9%81%E9%AB%94%E4%B8%AD%E6%96%87-nlp-%E5%BE%9Eword2vec%E5%88%B0-%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90/

https://chunshan-theta.github.io/NLPLab/
```
### 參考資料
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

# IMDb-Movie-Review
```
IMDb網路電影影評資料集
```
### Sentiment Analysis on IMDb
```
https://paperswithcode.com/sota/sentiment-analysis-on-imdb
```

### RNN
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

(X_train, y_train), (X_test, y_test) =
keras.datasets.imdb.load_data(num_words=top_words)

# X_train = tf.convert_to_tensor(X_train)
# y_train = tf.one_hot(y_train, depth=2)

print('Pad sequences (samples x time)')

x_train = keras.preprocessing.sequence.pad_sequences(X_train,
maxlen=max_review_length)

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

# 使用XLNet(2019)
```
XLNet: Generalized Autoregressive Pretraining for Language Understanding
Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le
(Submitted on 19 Jun 2019)
https://arxiv.org/abs/1906.08237

https://github.com/zihangdai/xlnet
```

```
XLNet is a new unsupervised language representation learning method 
based on a novel generalized permutation language modeling objective. 

Additionally, XLNet employs Transformer-XL as the backbone model, exhibiting excellent performance for language tasks involving long context. 

Overall, XLNet achieves state-of-the-art (SOTA) results on various downstream language tasks including question answering, natural language inference, sentiment analysis, and document ranking.
```
```
What is XLNet and why it outperforms BERT
https://towardsdatascience.com/what-is-xlnet-and-why-it-outperforms-bert-8d8fce710335

XLNet:运行机制及和Bert的异同比较
https://zhuanlan.zhihu.com/p/70257427
```
```
2019-NLP最強模型: XLNet

2019年6月中旬Google提出一個NLP模型XLNet，
在眾多NLP任務包括RACE, GLUE Benchmark以及許多Text-classification上輾壓眾生，
尤其是在號稱最困難的大型閱讀理解QA任務RACE足足超越BERT 6~9個百分點，
其中XLNet模型改善了ELMo, GPT, BERT的缺點，
有ELMo, GPT的AR性質，又有跟BERT一樣，使用AE性質能夠捕捉bidirectional context的訊息，
最後再把Transformer-XL能夠訓練大型文本的架構拿來用


目前NLP的發展趨勢越來越靠近Pre-train model+Downstream(transfer learning)，
即先訓練一個夠generalize的模型，接著依照下游任務的需求去更改結構並Finetune模型，

真正重要的其實是Pre-train model token之間依賴關係，即word embedding，
有良好的word embedding基本上效果也會不錯，

目前在Pre-train word embedding的方式都是預測Sequence本身的單詞，
即依賴上下文來預測單詞，著名的模型如ELMo, GPT, BERT, ERNIE。
```
## 範例程式
```
資料來源
https://colab.research.google.com/github/zihangdai/xlnet/blob/master/notebooks/colab_imdb_gpu.ipynb
```
```
! pip install sentencepiece
```
```
"""Download the pretrained XLNet model and unzip"""

# only needs to be done once
! wget https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip
! unzip cased_L-24_H-1024_A-16.zip

"""Download extract the imdb dataset - surpessing output"""

! wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
! tar zxf aclImdb_v1.tar.gz

"""Git clone XLNet repo for access to run_classifier and the rest of the xlnet module"""

! git clone https://github.com/zihangdai/xlnet.git
```
```
"""## Define Variables
Define all the dirs: data, xlnet scripts & pretrained model. 
If you would like to save models then you can authenticate a GCP account and use that for the OUTPUT_DIR & CHECKPOINT_DIR - you will need a large amount storage to fix these models. 

Alternatively it is easy to integrate a google drive account, checkout this guide for [I/O in colab](https://colab.research.google.com/notebooks/io.ipynb) but rememeber these will take up a large amount of storage.
"""

SCRIPTS_DIR = 'xlnet' #@param {type:"string"}
DATA_DIR = 'aclImdb' #@param {type:"string"}
OUTPUT_DIR = 'proc_data/imdb' #@param {type:"string"}
PRETRAINED_MODEL_DIR = 'xlnet_cased_L-24_H-1024_A-16' #@param {type:"string"}
CHECKPOINT_DIR = 'exp/imdb' #@param {type:"string"}
```
```
"""## Run Model
This will set off the fine tuning of XLNet. 

There are a few things to note here:
1.   This script will train and evaluate the model
2.   This will store the results locally on colab and will be lost when you are disconnected from the runtime
3.   This uses the large version of the model (base not released presently)
4.   We are using a max seq length of 128 with a batch size of 8 please refer to the [README](https://github.com/zihangdai/xlnet#memory-issue-during-finetuning) for why this is.
5. This will take approx 4hrs to run on GPU.
"""

train_command = "python xlnet/run_classifier.py \
  --do_train=True \
  --do_eval=True \
  --eval_all_ckpt=True \
  --task_name=imdb \
  --data_dir="+DATA_DIR+" \
  --output_dir="+OUTPUT_DIR+" \
  --model_dir="+CHECKPOINT_DIR+" \
  --uncased=False \
  --spiece_model_file="+PRETRAINED_MODEL_DIR+"/spiece.model \
  --model_config_path="+PRETRAINED_MODEL_DIR+"/xlnet_config.json \
  --init_checkpoint="+PRETRAINED_MODEL_DIR+"/xlnet_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=8 \
  --eval_batch_size=8 \
  --num_hosts=1 \
  --num_core_per_host=1 \
  --learning_rate=2e-5 \
  --train_steps=4000 \
  --warmup_steps=500 \
  --save_steps=500 \
  --iterations=500"

! {train_command}
```
```
"""## Running & Results
These are the results that I got from running this experiment
### Params
*    --max_seq_length=128 \
*    --train_batch_size= 8 

### Times
*   Training: 1hr 11mins
*   Evaluation: 2.5hr

### Results
*  Most accurate model on final step
*  Accuracy: 0.92416, eval_loss: 0.31708

### Model

*   The trained model checkpoints can be found in 'exp/imdb'
"""
```
```
.................
INFO:tensorflow:Finished evaluation at 2019-11-09-11:32:43
I1109 11:32:43.085186 140605255649152 evaluation.py:275] Finished evaluation at 2019-11-09-11:32:43
INFO:tensorflow:Saving dict for global step 4000: eval_accuracy = 0.92328, eval_loss = 0.32805777, global_step = 4000, loss = 0.32805777
I1109 11:32:43.085558 140605255649152 estimator.py:2049] Saving dict for global step 4000: eval_accuracy = 0.92328, eval_loss = 0.32805777, global_step = 4000, loss = 0.32805777
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 4000: exp/imdb/model.ckpt-4000
I1109 11:32:43.087548 140605255649152 estimator.py:2109] Saving 'checkpoint_path' summary for global step 4000: exp/imdb/model.ckpt-4000
INFO:tensorflow:================================================================================
I1109 11:32:43.088168 140605255649152 run_classifier.py:786] ================================================================================
INFO:tensorflow:Eval result | eval_accuracy 0.9232800006866455 | eval_loss 0.32805776596069336 | global_step 4000 | loss 0.32805776596069336 | path exp/imdb/model.ckpt-4000 | step 4000 | 
I1109 11:32:43.088358 140605255649152 run_classifier.py:790] Eval result | eval_accuracy 0.9232800006866455 | eval_loss 0.32805776596069336 | global_step 4000 | loss 0.32805776596069336 | path exp/imdb/model.ckpt-4000 | step 4000 | 
INFO:tensorflow:================================================================================
I1109 11:32:43.088484 140605255649152 run_classifier.py:795] ================================================================================
INFO:tensorflow:Best result | eval_accuracy 0.9232800006866455 | eval_loss 0.32805776596069336 | global_step 4000 | loss 0.32805776596069336 | path exp/imdb/model.ckpt-4000 | step 4000 | 
I1109 11:32:43.088574 140605255649152 run_classifier.py:799] Best result | eval_accuracy 0.9232800006866455 | eval_loss 0.32805776596069336 | global_step 4000 | loss 0.32805776596069336 | path exp/imdb/model.ckpt-4000 | step 4000 | 

```
## 範例程式關鍵解說
```

```
