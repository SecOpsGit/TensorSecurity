# 三大主題
```
RNN Model
Word Embedding
NLP since 2018
```
### 實戰主題
```
[1]IMDb-Movie-Review IMDb網路電影影評資料集 Sentiment Analysis on IMDb
[2]TextGeneration文本生成===作詞機器人

[3]TimeSeriesPrediction時間序列預測
使用 LSTM RNN預測Bitcoins價格
```
##### 以後
```
聊天機器人
```

# RNN Model
```
RNN
LSTM
GRU
BiRNN
```

```
http://colah.github.io/posts/2015-08-Understanding-LSTMs/
http://karpathy.github.io/2015/05/21/rnn-effectiveness/
http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/
```

# Word Embedding
```
NNLM神經網路語言模型(2003)

Word2Vec(2013)
Glove(2013)

ELMO(2017) ELMO詞向量模型



```

```
Lecture 2 | Word Vector Representations: word2vec
https://www.youtube.com/watch?v=ERibwqs9p38

ELMO, BERT, GPT
https://www.youtube.com/watch?v=UYPa347-DdE

Language Model Overview: From word2vec to BERT
https://www.youtube.com/watch?v=ycXWAtm22-w
```
```
http://www.6aiq.com/article/1560265487336?p=1&m=0
```
### ELMO(2017) ELMO詞向量模型
```

```
```
pip install allennlp 
```

# NLP since 2018
```

Seq2Seq(2014)
Attention Model注意力機制(2014)
Google The transformer(2017) Self Attention 變形金剛

BERT(2018)
GPT-2
XLNET(2019)


```
```
自然語言處理中的語言模型預訓練方法
https://www.jiqizhixin.com/articles/2018-10-22-3

Seq2seq pay Attention to Self Attention
https://medium.com/@bgg/seq2seq-pay-attention-to-self-attention-part-1-d332e85e9aad
https://medium.com/%40bgg/seq2seq-pay-attention-to-self-attention-part-2-%E4%B8%AD%E6%96%87%E7%89%88-ef2ddf8597a4

https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
https://jalammar.github.io/illustrated-transformer/
https://jalammar.github.io/illustrated-bert/
https://blog.csdn.net/qq_41664845/article/details/84787969

```
### Seq2Seq(2014)
```
Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio
(Submitted on 3 Jun 2014 (v1), last revised 3 Sep 2014 (this version, v3))
https://arxiv.org/abs/1406.1078

In this paper, we propose a novel neural network model called RNN Encoder-Decoder 
that consists of two recurrent neural networks (RNN). 

One RNN encodes a sequence of symbols into a fixed-length vector representation, 
and the other decodes the representation into another sequence of symbols. 

The encoder and decoder of the proposed model are jointly trained 
to maximize the conditional probability of a target sequence given a source sequence. 

The performance of a statistical machine translation system is empirically found 
to improve by using the conditional probabilities of phrase pairs computed by the RNN Encoder-Decoder 
as an additional feature in the existing log-linear model. 

Qualitatively, we show that the proposed model learns 
a semantically and syntactically meaningful representation of linguistic phrases.
```
```
Sequence to Sequence Learning with Neural Networks
Ilya Sutskever, Oriol Vinyals, Quoc V. Le
(Submitted on 10 Sep 2014 (v1), last revised 14 Dec 2014 (this version, v3))
https://arxiv.org/abs/1409.3215

Deep Neural Networks (DNNs) are powerful models that have achieved excellent performance on difficult learning tasks. 

Although DNNs work well whenever large labeled training sets are available, 
they cannot be used to map sequences to sequences. 

In this paper, we present a general end-to-end approach to sequence learning 
that makes minimal assumptions on the sequence structure. 

Our method uses a multilayered Long Short-Term Memory (LSTM) 
to map the input sequence to a vector of a fixed dimensionality, 
and then another deep LSTM to decode the target sequence from the vector. 

Our main result is that on an English to French translation task from the WMT'14 dataset, 
the translations produced by the LSTM achieve a BLEU score of 34.8 on the entire test set, 
where the LSTM's BLEU score was penalized on out-of-vocabulary words. 

Additionally, the LSTM did not have difficulty on long sentences. 

For comparison, a phrase-based SMT system achieves a BLEU score of 33.3 on the same dataset. 
When we used the LSTM to rerank the 1000 hypotheses produced by the aforementioned SMT system, 
its BLEU score increases to 36.5, which is close to the previous best result on this task. 
The LSTM also learned sensible phrase and sentence representations 
that are sensitive to word order and are relatively invariant to the active and the passive voice. 

Finally, we found that reversing the order of the words in all source sentences (but not target sentences) 
improved the LSTM's performance markedly, 
because doing so introduced many short term dependencies between the source and the target sentence 
which made the optimization problem easier.

```

### Attention Model注意力機制(2014)
```
Neural Machine Translation by Jointly Learning to Align and Translate
Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
(Submitted on 1 Sep 2014 (v1), last revised 19 May 2016 (this version, v7))
https://arxiv.org/abs/1409.0473


```

```
https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247485853&idx=1&sn=7e1dae6b690d718d17bebd8c37d2689b&chksm=96e9d61da19e5f0bafe59a848feccd340ce4fe1ff98cc5dfe24ea08ade644e176b45b8da64a7&scene=21#wechat_redirect


```


### Google The transformer(2017) Self Attention 變形金剛
```





```

```


```

### BERT(2018)
```


```

```


```


### GPT-2

```
Language Models are Unsupervised Multitask Learners
https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf

https://github.com/openai/gpt-2

https://paperswithcode.com/paper/language-models-are-unsupervised-multitask
```

```
https://colab.research.google.com/drive/1VLG8e7YSEwypxU-noRNhsv5dW4NfTGce#scrollTo=H7LoMj4GA4n_&forceEdit=true&sandboxMode=true

https://zhpmatrix.github.io/2019/02/16/transformer-multi-task/
https://blog.csdn.net/DarrenXf/article/details/88369809
```
```
直觀理解 GPT-2 語言模型並生成金庸武俠小說
https://leemeng.tw/gpt2-language-model-generate-chinese-jing-yong-novels.html

我用 OpenAI 文本生成器續寫了《復仇者聯盟》
2019/05/26 
https://www.inside.com.tw/article/16479-open-ai-edit-the-avengers
```
```
Visualizing Attention in Transformer-Based Language Representation Models
Jesse Vig
(Submitted on 4 Apr 2019 (v1), last revised 11 Apr 2019 (this version, v2))
https://arxiv.org/abs/1904.02679

We present an open-source tool for visualizing multi-head self-attention 
in Transformer-based language representation models. 

The tool extends earlier work by visualizing attention at three levels of granularity: 
the attention-head level, the model level, and the neuron level. 

We describe how each of these views can help to interpret the model, 
and we demonstrate the tool on the BERT model and the OpenAI GPT-2 model. 

We also present three use cases for analyzing GPT-2: detecting model bias, 
identifying recurring patterns, and linking neurons to model behavior.
```
### XLNET(2019)
```
XLNet: Generalized Autoregressive Pretraining for Language Understanding
Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le
(Submitted on 19 Jun 2019)
https://arxiv.org/abs/1906.08237

https://github.com/zihangdai/xlnet

```

```


```








