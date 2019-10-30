# RNN
```

```

```

```

# Word Embedding
```

ELMO(2017) ELMO詞向量模型



```

```
Lecture 2 | Word Vector Representations: word2vec
https://www.youtube.com/watch?v=ERibwqs9p38
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
Attention Model注意力機制
Google The transformer(2017) Self Attention 變形金剛



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

###
```


```

```


```


###
```


```

```


```

###
```


```

```


```

###
```


```

```


```

###
```


```

```


```

###
```


```

```


```








