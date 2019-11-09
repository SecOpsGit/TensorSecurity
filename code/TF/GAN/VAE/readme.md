# 開創性的原創論文
```
Auto-Encoding Variational Bayes
Diederik P Kingma, Max Welling
(Submitted on 20 Dec 2013 (v1), last revised 1 May 2014 (this version, v10))

https://arxiv.org/abs/1312.6114

How can we perform efficient inference and learning in directed probabilistic models, 
in the presence of continuous latent variables 
with intractable posterior distributions, and large datasets? 

We introduce a stochastic variational inference and learning algorithm 
that scales to large datasets and, 
under some mild differentiability conditions, even works in the intractable case. 

Our contributions is two-fold. 

First, we show that a reparameterization of the variational lower bound yields a lower bound estimator 
that can be straightforwardly optimized using standard stochastic gradient methods. 

Second, we show that for i.i.d. datasets with continuous latent variables per datapoint, 
posterior inference can be made especially efficient by fitting an approximate inference model 
(also called a recognition model) to the intractable posterior using the proposed lower bound estimator. 

Theoretical advantages are reflected in experimental results.
```
```
VAE(Variational Autoencoder)的原理
https://www.cnblogs.com/huangshiyu13/p/6209016.html

變分自編碼器（VAEs）
https://zhuanlan.zhihu.com/p/25401928
```
### Youtube影片
```
Variational Auto-encoders (VAE)
https://www.youtube.com/watch?v=2m9E-aSXtl8

[Variational Autoencoder] Auto-Encoding Variational Bayes | AISC Foundational
https://www.youtube.com/watch?v=Tc-XfiDPLf4


Deep Learning 19: (1) Variational AutoEncoder : Introduction and Probability Refresher
https://www.youtube.com/watch?v=w8F7_rQZxXk

Deep Learning 20: (2) Variational AutoEncoder : Explaining KL (Kullback-Leibler) Divergence
https://www.youtube.com/watch?v=wdKYveLIxgU

```
### review

```
Variational Inference
David M. Blei
https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf
```
```
Tutorial on Variational Autoencoders
CARL DOERSCH
Carnegie Mellon / UC Berkeley
August 16, 2016

https://arxiv.org/pdf/1606.05908.pdf
```

```
An Introduction to Variational Autoencoders
Diederik P. Kingma, Max Welling
(Submitted on 6 Jun 2019 (v1), last revised 24 Jul 2019 (this version, v2))

Variational autoencoders provide a principled framework for learning deep latent-variable models 
and corresponding inference models. 

In this work, we provide an introduction to variational autoencoders and some important extensions.
```
# Tensorflow官方示範程式
```
Convolutional Variational Autoencoder
https://www.tensorflow.org/tutorials/generative/cvae
```
# 發展
```
A Hybrid Convolutional Variational Autoencoder for Text Generation
Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth
(Submitted on 8 Feb 2017)
https://arxiv.org/abs/1702.02390
```

```
Squeezed Convolutional Variational AutoEncoder for Unsupervised Anomaly Detection 
in Edge Device Industrial Internet of Things
Dohyung Kim, Hyochang Yang, Minki Chung, Sungzoon Cho
(Submitted on 18 Dec 2017)
https://arxiv.org/abs/1712.06343
```
