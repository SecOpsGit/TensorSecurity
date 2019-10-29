#
```

https://radimrehurek.com/gensim/auto_examples/


Gensim是一款開源的協力廠商Python工具包，
用於從原始的非結構化的文本中，無監督地學習到文本隱層的主題向量表達。
它支持包括TF-IDF，LSA，LDA，和word2vec在內的多種主題模型演算法，
支援流式訓練，並提供了諸如相似度計算，資訊檢索等一些常用任務的API

```

```
https://radimrehurek.com/gensim/corpora/wikicorpus.html
```

```
gensim-doc2vec实战
http://www.shuang0420.com/2016/06/01/gensim-doc2vec%E5%AE%9E%E6%88%98/

使用 gensim 中的 word2vec 訓練中文詞向量
https://github.com/ohya1004/gensim-word2vec


CKIP Lab 中文詞知識庫小組
https://ckip.iis.sinica.edu.tw
```


### 基本概念
```
•	語料（Corpus）：一組原始文本的集合，用於無監督地訓練文本主題的隱層結構。
                 語料中不需要人工標注的附加資訊。
                 在Gensim中，Corpus通常是一個可反覆運算的物件（比如清單）。
                 每一次反覆運算返回一個可用于表達文本物件的稀疏向量。
•	向量（Vector）：由一組文本特徵構成的清單。
                 是一段文本在Gensim中的內部表達。
•	稀疏向量（SparseVector）：通常，我們可以略去向量中多餘的0元素。
                           此時，向量中的每一個元素是一個(key, value)的元組
•	模型（Model）：是一個抽象的術語。
                 定義了兩個向量空間的變換（即從文本的一種向量表達變換為另一種向量表達）

https://zhuanlan.zhihu.com/p/37175253
```

## WordVec的初步使用
```
WordVec简介
https://blog.csdn.net/SumResort_LChaowei/article/details/80136497
```
## GloVe
```
https://zhuanlan.zhihu.com/p/35653749
```
