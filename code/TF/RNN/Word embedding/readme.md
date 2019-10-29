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
