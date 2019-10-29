#
```
詞向量(word vector，也被稱為 word embedding 或 representation)

```

# pretrain language representation
###  pretrain
```
神經網路在進行訓練的時候基本都是基於後向傳播（BP）演算法，通過對網路模型參數進行隨機初始化，
然後通過 BP 演算法利用例如 SGD 這樣的優化演算法去優化模型參數。


預訓練==>該模型的參數不再是隨機初始化，而是先有一個任務進行訓練得到一套模型參數，
        然後用這套參數對模型進行初始化，再進行訓練。

早期的使用自編碼器棧式搭建深度神經網路就是這個思想。
還有詞向量也可以看成是第一層 word embedding 進行了預訓練
基於神經網路的遷移學習中也大量用到了這個思想。

```
```
自然语言处理中的语言模型预训练方法
https://www.cnblogs.com/robert-dlut/p/9824346.html
https://www.jiqizhixin.com/articles/2018-10-22-3

ELMo 
OpenAI GPT | GPT2 
BERT
```
### ELMo
```
ELMo最好用词向量Deep Contextualized Word Representations

Deep Contextualized Word Representations

Semi-supervised sequence tagging with bidirectional language models
```
```
一個預訓練的詞表示應該能夠包含豐富的句法和語義資訊，並且能夠對多義詞進行建模。

傳統的詞向量（例如 word2vec）是上下文無關的。

例如下面"apple"的例子，這兩個"apple"根據上下文意思是不同的，但是在 word2vec 中，只有 apple 一個詞向量，無法對一詞多義進行建模。

他們利用語言模型來獲得一個上下文相關的預訓練表示，稱為 ELMo，並在 6 個 NLP 任務上獲得了提升。

使用的是一個雙向的 LSTM 語言模型，由一個前向和一個後向語言模型構成，
目標函數就是取這兩個方向語言模型的最大似然。
```
### OpenAI GPT | GPT2 
```
Improving Language Understanding by Generative Pre-Training 

這是 OpenAI 團隊前一段時間放出來的預印版論文。他們的目標是學習一個通用的表示，能夠在大量任務上進行應用。


利用Transformer網路代替了LSTM作為語言模型來更好的捕獲長距離語言結構。
然後在進行具體任務有監督微調時使用了語言模型作為附屬任務訓練目標。
在 12 個 NLP 任務上進行了實驗，9 個任務獲得了 SOTA。

```
### BERT
```

```

### Language Models are Unsupervised Multitask Learners
```
Language Models are Unsupervised Multitask Learners
https://zhpmatrix.github.io/2019/02/16/transformer-multi-task/
```
```

```
