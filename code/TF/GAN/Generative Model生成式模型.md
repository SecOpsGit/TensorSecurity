
# Generative Model生成式模型  unsupervised Deep Learning
```
[A]Generative Adversarial Network (GAN) 

[B]Likelihood-based methods
  have three main categories: 
  [B1]Autoregressive models 
      Durk P Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling.
      Improved variational inference with inverse autoregressive flow. 
      In D. D. Lee, M. Sugiyama,U. V. Luxburg, I. Guyon, and R. Garnett, editors, 
      Advances in Neural Information Processing Systems 29, pages 4743–4751. Curran Associates, Inc., 2016.
      https://arxiv.org/abs/1606.04934
      
  [B2]Variational autoencoders (VAEs) Auto-Encoders 與Varitional Auto-Encoders
      Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. 
      In 2nd International Conference on Learning Representations, 
      ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings, 2014.
      https://arxiv.org/abs/1312.6114
      
      Tutorial on Variational Autoencoders  https://arxiv.org/abs/1606.05908
      近三年來，變分自編碼（VAE）作為一種無監督學習複雜分佈的方法受到人們關注，
      VAE因其基於標準函數近似（神經網路）而吸引人，並且可以通過隨機梯度下降進行訓練。
      VAE已經在許多生成複雜資料包括手寫字體[1,2]、人臉圖像[1,3,4]、住宅編碼[5,6]、CIFAR圖像[6]、物理模型場景[4]、分割[7]
      以及預測靜態圖像[8]上顯現優勢。本教程介紹VAE背後的靈感和數學解釋，以及一些實證。沒有變分貝葉斯方法假設的先驗知識。
      
      
      An Introduction to Variational Autoencoders
      Diederik P. Kingma, Max Welling
      (Submitted on 6 Jun 2019 (v1), last revised 24 Jul 2019 (this version, v2))
      https://arxiv.org/abs/1906.02691
      
      VAE(Variational Autoencoder)的原理
      https://www.cnblogs.com/huangshiyu13/p/6209016.html
      論文的理論推導見：https://zhuanlan.zhihu.com/p/25401928
      https://github.com/kvfrans/variational-autoencoder 
      和一个整理好的版本： https://jmetzen.github.io/2015-11-27/vae.html
      
  [B3]Flow-based models (2014/2018:GLOW)  
    The flow-based generative model is constructed using a sequence of invertible and tractable transformations, 
    the model explicitly learns the data distribution and therefore the loss function is simply a negative log-likelihood
    [延伸學習]
    Flow-based Generative Model: https://www.youtube.com/watch?v=uXY18nzdSsM
```
```
史丹佛大學著名課程 http://cs231n.stanford.edu/
Lecture 13 | Generative Models
https://www.youtube.com/watch?v=5WoItGTWV54&t=698s
http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf
```
# Generative Adversarial Network (GAN)發展史
```
Generative Adversarial Networks(2014)
Conditional GANs(2014)

DCGANs(2015)

Improved Techniques for Training GANs(2016)
Progressively Growing GANs(2017)

Pix2Pix(2016)
StackGAN(2016)
CycleGAN(2017)

BigGAN(2018)[史上最强 GAN 图像生成器]====BigGAN_Deep(2019)[更嚇人]
StyleGAN(2018) https://thispersondoesnotexist.com/
[最强非 GAN 生成器]Vector Quantized Variational AutoEncoder (VQ-VAE) models(2019)


https://towardsdatascience.com/must-read-papers-on-gans-b665bbae3317
```

### Awesome GAN
```
https://github.com/nightrome/really-awesome-gan
https://github.com/nashory/gans-awesome-applications
https://github.com/Faldict/awesome-GAN
https://github.com/dongb5/GAN-Timeline

https://github.com/hindupuravinash/the-gan-zoo
```
```
论文地址：https://arxiv.org/abs/1906.00446
更多样本地址：https://drive.google.com/file/d/1H2nr_Cu7OK18tRemsWn_6o5DGMNYentM/view



改进 GAN 训练的技术 —— Salimans et al. (2016)
这篇论文 (作者包括 Ian Goodfellow) 根据上述 DCGAN 论文中列出的架构指南，提供了一系列建议。
这篇论文将帮助你了解 GAN 不稳定性的最佳假设。
此外，本文还提供了许多用于稳定 DCGAN 训练的其他机器，包括特征匹配、 minibatch 识别、历史平均、单边标签平滑和虚拟批标准化。
使用这些技巧来构建一个简单的 DCGAN 实现是一个很好的练习，有助于更深入地了解 GAN。
Improved Techniques for Training GANs
Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen
https://arxiv.org/abs/1606.03498


Progressively Growing GANs— Karras et al. (2017)
Progressively Growing GAN (PG-GAN) 有着惊人的结果，以及对 GAN 问题的创造性方法，因此也是一篇必读论文。
这篇 GAN 论文来自 NVIDIA Research，提出以一种渐进增大（progressive growing）的方式训练 GAN，
通过使用逐渐增大的 GAN 网络（称为 PG-GAN）和精心处理的CelebA-HQ 数据集，
实现了效果令人惊叹的生成图像。
作者表示，这种方式不仅稳定了训练，GAN 生成的图像也是迄今为止质量最好的。

它的关键想法是渐进地增大生成器和鉴别器：
从低分辨率开始，随着训练的进展，添加新的层对越来越精细的细节进行建模。
“Progressive Growing” 指的是先训练 4x4 的网络，然后训练 8x8，不断增大，最终达到 1024x1024。
这既加快了训练速度，又大大稳定了训练速度，并且生成的图像质量非常高。
Progressively Growing GAN 的多尺度架构，模型从 4×4 逐步增大到 1024×1024

论文：
Progressive Growing of GANs for Improved Quality, Stability, and Variation
Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen
https://arxiv.org/abs/1710.10196
相关阅读：
迄今最真实的 GAN：英伟达渐进增大方式训练 GAN，生成前所未有高清图像


CycleGAN — Zhu et al. (2017)

CycleGAN 的论文不同于前面列举的 6 篇论文，因为它讨论的是 image-to-image 的转换问题，而不是随机向量的图像合成问题。
CycleGAN 更具体地处理了没有成对训练样本的 image-to-image 转换的情况。
然而，由于 Cycle-Consistency loss 公式的优雅性，以及如何稳定 GAN 训练的启发性，这是一篇很好的论文。
CycleGAN 有很多很酷的应用，比如超分辨率，风格转换，例如将马的图像变成斑马。

Cycle Consistency Loss 背后的主要想法，一个句子从法语翻译成英语，再翻译回法语，应该跟原来的是同一个句子

论文：
Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros
https://arxiv.org/abs/1703.10593

```
# HOT
### BigGAN(2018)====BigGAN_Deep(2019)
```
BigGAN(2018)[史上最強 GAN 圖像生成器]====BigGAN_Deep(2019)[更嚇人]

BigGAN 模型是基於 ImageNet 生成圖像品質最高的模型之一。
該模型很難在本地機器上實現，而且 BigGAN 有許多元件，如 Self-Attention、 Spectral Normalization 
和帶有投影鑒別器的 cGAN，這些組件在各自的論文中都有更好的解釋。
不過，這篇論文對構成當前最先進技術水準的基礎論文的思想提供了很好的概述，因此非常值得閱讀。
```
```
Large Scale GAN Training for High Fidelity Natural Image Synthesis
Andrew Brock, Jeff Donahue, Karen Simonyan
https://arxiv.org/abs/1809.11096

論文連結：https://openreview.net/pdf?id=B1xsqj09Fm
BigGAN TF Hub demo 地址：https://tfhub.dev/s?q=biggan

https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/biggan_generation_with_tf_hub.ipynb#scrollTo=Cd1dhL4Ykbm7
```
```
[BigGAN-PyTorch]https://github.com/ajbrock/BigGAN-PyTorch
```

### StyleGAN(2018)
```
https://github.com/paubric/thisrepositorydoesnotexist

http://botpoet.com/vote/o-little-root-dream/

https://thispersondoesnotexist.com/
https://thiscatdoesnotexist.com/
thesecatsdonotexist.com
https://www.thiswaifudoesnotexist.net/
https://www.obormot.net/demos/these-waifus-do-not-exist
https://thisairbnbdoesnotexist.com/
```
```
A Style-Based Generator Architecture for Generative Adversarial Networks
Tero Karras, Samuli Laine, Timo Aila

https://arxiv.org/abs/1812.04948
https://github.com/NVlabs/stylegan
FFHQ 資料集：https://github.com/NVlabs/ffhq-dataset

We propose an alternative generator architecture for generative adversarial networks, 
borrowing from style transfer literature. 

The new architecture leads to an automatically learned, unsupervised separation of high-level attributes 
(e.g., pose and identity when trained on human faces) 
and stochastic variation in the generated images (e.g., freckles, hair), 
and it enables intuitive, scale-specific control of the synthesis. 

The new generator improves the state-of-the-art in terms of traditional distribution quality metrics, 
leads to demonstrably better interpolation properties, 
and also better disentangles the latent factors of variation. 

To quantify interpolation quality and disentanglement, 
we propose two new, automated methods that are applicable to any generator architecture. 
Finally, we introduce a new, highly varied and high-quality dataset of human faces.
```

```
https://www.ifanr.com/1173717

StyleGAN 模型可以說是最先進的，特別是利用了潛在空間控制。
該模型借鑒了神經風格遷移中一種稱為自我調整實例標準化 (AdaIN) 的機制來控制潛在空間向量 z。
映射網路和 AdaIN 條件在整個生成器模型中的分佈的結合使得很難自己實現一個 StyleGAN，
```

### VQ-VAE-2(2019)
```
[最强非 GAN 生成器]Vector Quantized Variational AutoEncoder (VQ-VAE) models(2019)
更多样本地址：https://drive.google.com/file/d/1H2nr_Cu7OK18tRemsWn_6o5DGMNYentM/view
```
```
Generating Diverse High-Fidelity Images with VQ-VAE-2
Ali Razavi, Aaron van den Oord, Oriol Vinyals
(Submitted on 2 Jun 2019)
https://arxiv.org/abs/1906.00446
```
### openai/GLOW(2018)
```
https://blog.openai.com/glow/


Glow: Generative Flow with Invertible 1x1 Convolutions
Diederik P. Kingma, Prafulla Dhariwal
(Submitted on 9 Jul 2018 (v1), last revised 10 Jul 2018 (this version, v2))
In Advances in Neural Information Processing Systems (pp. 10215-10224).
https://arxiv.org/abs/1807.03039

[Tensorflow 1.8]https://github.com/openai/glow
https://kknews.cc/tech/j4ra2ey.html

Flow-based generative models (Dinh et al., 2014) are conceptually attractive 
due to tractability of the exact log-likelihood, tractability of exact latent-variable inference, 
and parallelizability of both training and synthesis. 

In this paper we propose Glow, a simple type of generative flow using an invertible 1x1 convolution. 

Using our method we demonstrate a significant improvement in log-likelihood on standard benchmarks.
Perhaps most strikingly, we demonstrate that a generative model optimized 
towards the plain log-likelihood objective 
is capable of efficient realistic-looking synthesis and manipulation of large images. 
```
#### 後續發展
```
WaveGlow: A Flow-based Generative Network for Speech Synthesis
Ryan Prenger, Rafael Valle, Bryan Catanzaro
(Submitted on 31 Oct 2018)

https://github.com/NVIDIA/waveglow


In this paper we propose WaveGlow: a flow-based network capable of 
generating high quality speech from mel-spectrograms[https://zh.wikipedia.org/wiki/梅爾倒頻譜]. 

# mel-frequency cepstrum (MFC) 
# https://github.com/astorfi/speechpy
# SpeechPy - A Library for Speech Processing and Recognition: http://speechpy.readthedocs.io/en/latest/

WaveGlow combines insights from Glow and WaveNet 
in order to provide fast, efficient and high-quality audio synthesis, 
without the need for auto-regression. 

WaveGlow is implemented using only a single network, trained using only a single cost function: 
maximizing the likelihood of the training data, 
which makes the training procedure simple and stable. 

Our PyTorch implementation produces audio samples at a rate of more than 500 kHz on an NVIDIA V100 GPU. 
Mean Opinion Scores show that it delivers audio quality as good as the best publicly available WaveNet implementation. 
All code will be made publicly available online.
```
```
Generative Model with Dynamic Linear Flow
Huadong Liao, Jiawei He, Kunxian Shu
(Submitted on 8 May 2019)
https://arxiv.org/abs/1905.03239
[Tensorflow 1.12]https://github.com/naturomics/DLF


結合flow-based methods 及autoregressive methods的優點
In this paper, we propose Dynamic Linear Flow (DLF), 
a new family of invertible transformations with partially autoregressive structure. 

Our method benefits from the efficient computation of flow-based methods 
and high density estimation performance of autoregressive methods. 

We demonstrate that the proposed DLF yields state-of-theart performance 
on ImageNet 32x32 and 64x64 out of all flow-based methods, 
and is competitive with the best autoregressive model. 

Additionally, our model converges 10 times faster than Glow
```
```
Generative Flow via Invertible nxn Convolution
Thanh-Dat Truong, Khoa Luu, Chi Nhan Duong, Ngan Le, Minh-Triet Tran
(Submitted on 24 May 2019)
https://arxiv.org/abs/1905.10170

In this paper, we propose a novel invertible nxn convolution approach 
that overcomes the limitations of the invertible 1x1 convolution. 

In addition, our proposed network is not only tractable and invertible 
but also uses fewer parameters than standard convolutions. 

The experiments on CIFAR-10, ImageNet, and Celeb-HQ datasets, 
have showed that our invertible nxn convolution helps to improve the performance of generative models significantly.
```
```
Label-Conditioned Next-Frame Video Generation with Neural Flows
David Donahue
(Submitted on 16 Oct 2019)
https://arxiv.org/abs/1910.11106

影像產生
```
```
高清變臉更快更真！比GAN更厲害的可逆生成模型來了｜論文+代碼
https://kknews.cc/tech/j4ra2ey.html
```

#### Flow-based generative models[since 2014]
```
NICE: Non-linear Independent Components Estimation
Laurent Dinh, David Krueger, Yoshua Bengio
(Submitted on 30 Oct 2014 (v1), last revised 10 Apr 2015 (this version, v6))
https://arxiv.org/abs/1410.8516

Yoshua Bengio and Yann LeCun, editors. 
3rd International Conference on Learning Representations, 
ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Workshop Track Proceedings,
2015.

Dinh, L., Sohl-Dickstein, J., and Bengio, S. (2016). 
Density estimation using Real NVP. 
https://arxiv.org/abs/1605.08803

Grover, A., Dhar, M., and Ermon, S. (2018). 
Flow-gan: Combining maximum likelihood and adversarial learning in generative models. 
In AAAI Conference on Artificial Intelligence.
https://arxiv.org/abs/1705.08868
```


```
Li, Y., Gan, Z., Shen, Y., Liu, J., Cheng, Y., Wu, Y., ... & Gao, J. (2018).
StoryGAN: A Sequential Conditional GAN for Story Visualization.
arXiv preprint arXiv:1812.02784.
```
# Classic GAN
### 2014年 Generative Adversarial Networks
```
Generative Adversarial Networks — Goodfellow et al. (2014)
Ian Goodfellow 的原始 GAN 論文對任何研究 GAN 的人來說都是必讀的。
這篇論文定義了 GAN 框架，並討論了 “非飽和” 損失函數。
論文還給出了最優判別器的推導，這是近年來 GAN 論文中經常出現的一個證明。
論文還在 MNIST、TFD 和 CIFAR-10 圖像資料集上對 GAN 的有效性進行了實驗驗證。

```
```
Generative Adversarial Networks
Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, 
Sherjil Ozair, Aaron Courville, Yoshua Bengio
(Submitted on 10 Jun 2014)

https://arxiv.org/abs/1406.2661
https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf
https://github.com/goodfeli/adversarial
https://www.youtube.com/watch?v=HN9NRhm9waY

We propose a new framework for estimating generative models via an adversarial process, 
in which we simultaneously train two models: 
a generative model G that captures the data distribution, 
and a discriminative model D that estimates the probability that a sample came from the training data rather than G. 

The training procedure for G is to maximize the probability of D making a mistake. 
This framework corresponds to a minimax two-player game. 
In the space of arbitrary functions G and D, a unique solution exists, 
with G recovering the training data distribution and D equal to 1/2 everywhere. 

In the case where G and D are defined by multilayer perceptrons, 
the entire system can be trained with backpropagation. 

There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples.
```
```
https://blog.csdn.net/solomon1558/article/details/52549409

GAN啟發自博弈論中的二人零和博弈（two-player game），由[Goodfellow et al, NIPS 2014]開創性地提出。

在二人零和博弈中，兩位博弈方的利益之和為零或一個常數，即一方有所得，另一方必有所失。
GAN模型中的兩位元博弈方分別由生成式模型（generative model）和判別式模型（discriminative model）充當。

生成模型G捕捉樣本資料的分佈，判別模型是一個二分類器，估計一個樣本來自於訓練資料（而非生成資料）的概率。
G和D一般都是非線性映射函數，例如多層感知機、卷積神經網路等。
對於D來說期望輸出低概率（判斷為生成樣本），
對於生成模型G來說要儘量欺騙D，使判別模型輸出高概率（誤判為真實樣本），從而形成競爭與對抗。
```
```
GAN的基本原理其實非常簡單，這裡以生成圖片為例進行說明。

假設我們有兩個網路，G（Generator）和D（Discriminator）。
正如它的名字所暗示的那樣，它們的功能分別是：
G是一個生成圖片的網路，它接收一個隨機的噪聲z，通過這個噪聲生成圖片，記做G(z)。
D是一個判別網路，判別一張圖片是不是“真實的”。
   它的輸入引數是x，x代表一張圖片，
   輸出D（x）代表x為真實圖片的概率，如果為1，就代表100%是真實的圖片，而輸出為0，就代表不可能是真實的圖片。

在訓練過程中，生成網路G的目標就是儘量生成真實的圖片去欺騙判別網路D。
而D的目標就是儘量把G生成的圖片和真實的圖片分別開來。
這樣，G和D構成了一個動態的“博弈過程”。

最後博弈的結果是什麼？
在最理想的狀態下，G可以生成足以“以假亂真”的圖片G(z)。
對於D來說，它難以判定G生成的圖片究竟是不是真實的，因此D(G(z)) = 0.5。

這樣我們的目的就達成了：

       我們得到了一個生成式的模型G，它可以用來生成圖片。

```

```
https://skymind.ai/wiki/generative-adversarial-network-gan

生成式对抗网络GAN研究进展（二）——原始GAN
https://blog.csdn.net/solomon1558/article/details/52549409
```
### GAN生成式對抗網路的四個優勢
```
與其他生成式模型相比較，生成式對抗網路有以下四個優勢【OpenAI Ian Goodfellow的Quora問答】：

[1]根據實際的結果，它們看上去可以比其它模型產生了更好的樣本（圖像更銳利、清晰）。
[2]生成對抗式網路框架能訓練任何一種生成器網路（理論上-實踐中，用 REINFORCE 來訓練帶有離散輸出的生成網路非常困難）。
  大部分其他的框架需要該生成器網路有一些特定的函數形式，比如輸出層是高斯的。
  重要的是所有其他的框架需要生成器網路遍佈非零品質（non-zero mass）。
  生成對抗式網路能學習可以僅在與資料接近的細流形（thin manifold）上生成點。
[3]不需要設計遵循任何種類的因式分解的模型，任何生成器網路和任何鑒別器都會有用。
[4]無需利用瑪律科夫鏈反復採樣，無需在學習過程中進行推斷（Inference），回避了近似計算棘手的概率的難題。

與PixelRNN相比，生成一個樣本的執行時間更小。
    GAN 每次能產生一個樣本，而 PixelRNN 需要一次產生一個圖元來生成樣本。

與VAE 相比，它沒有變化的下限。如果鑒別器網路能完美適合，那麼這個生成器網路會完美地恢復訓練分佈。
    換句話說，各種對抗式生成網路會漸進一致（asymptotically consistent），而 VAE 有一定偏置。

與深度玻爾茲曼機相比，既沒有一個變化的下限，也沒有棘手的分區函數。
    它的樣本可以一次性生成，而不是通過反復應用瑪律可夫鏈運算器（Markov chain operator）。

與 GSN 相比，它的樣本可以一次生成，而不是通過反復應用瑪律可夫鏈運算器。

與NICE 和 Real NVE 相比，在 latent code 的大小上沒有限制。
```
```
Ian Goodfellow NIPS 2016的演講
https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/
Neural-Information-Processing-Systems-Conference-NIPS-2016/Generative-Adversarial-Networks

[簡報]http://www.iangoodfellow.com/slides/2016-12-04-NIPS.pdf

[論文]https://arxiv.org/abs/1701.00160
NIPS 2016 Tutorial: Generative Adversarial Networks
Ian Goodfellow
(Submitted on 31 Dec 2016 (v1), last revised 3 Apr 2017 (this version, v4))
This report summarizes the tutorial presented by the author at NIPS 2016 on generative adversarial networks (GANs). 

The tutorial describes: 
(1) Why generative modeling is a topic worth studying, 
(2) how generative models work, and how GANs compare to other generative models, 
(3) the details of how GANs work, 
(4) research frontiers in GANs, and 
(5) state-of-the-art image models that combine GANs with other methods. 

Finally, the tutorial contains three exercises for readers to complete, and the solutions to these exercises.
```

### GAN常見的主要問題：
```
[1]不收斂（non-convergence）的問題。
    目前面臨的基本問題是：所有的理論都認為 GAN 應該在納什均衡（Nash equilibrium）上有卓越的表現，
    但梯度下降只有在凸函數的情況下才能保證實現納什均衡。
    當博弈雙方都由神經網路表示時，在沒有實際達到均衡的情況下，讓它們永遠保持對自己策略的調整是可能的【OpenAI Ian Goodfellow的Quora】。

[2]難以訓練：崩潰問題（collapse problem）
    GAN模型被定義為極小極大問題，沒有損失函數，在訓練過程中很難區分是否正在取得進展。
    GAN的學習過程可能發生崩潰問題（collapse problem），生成器開始退化，總是生成同樣的樣本點，無法繼續學習。
    當生成模型崩潰時，判別模型也會對相似的樣本點指向相似的方向，訓練無法繼續。【Improved Techniques for Training GANs】

[3]無需預先建模，模型過於自由不可控。
    與其他生成式模型相比，GAN這種競爭的方式不再要求一個假設的資料分佈，即不需要formulate p(x)，
    而是使用一種分佈直接進行採樣sampling，從而真正達到理論上可以完全逼近真實資料，這也是GAN最大的優勢。
    然而，這種不需要預先建模的方法缺點是太過自由了，對於較大的圖片，較多的 pixel的情形，基於簡單 GAN 的方式就不太可控了。
    在GAN[Goodfellow Ian, Pouget-Abadie J] 中，每次學習參數的更新過程，被設為D更新k回，G才更新1回，也是出於類似的考慮。
```
### 2014年 cGAN
```
為了解決GAN太過自由這個問題，一個很自然的想法是給GAN加一些約束，於是便有了Conditional Generative Adversarial Nets（CGAN）

這項工作提出了一種帶條件約束的GAN，在生成模型（D）和判別模型（G）的建模中均引入條件變數y（conditional variable y），
使用額外資訊y對模型增加條件，可以指導資料生成過程。
這些條件變數y可以基於多種資訊，
例如類別標籤，用於圖像修復的部分資料[2]，來自不同模態（modality）的資料。
如果條件變數y是類別標籤，可以看做CGAN 是把純無監督的 GAN 變成有監督的模型的一種改進。

這個簡單直接的改進被證明非常有效,並廣泛用於後續的相關工作中
```

```
Conditional GANs — Mirza and Osindero (2014)
這是一篇很好的論文，讀起來很順暢。
條件 GAN(Conditional GAN) 是最先進的 GAN之一。
論文展示了如何整合資料的類標籤，從而使 GAN 訓練更加穩定。
利用先驗資訊對 GAN 進行調節這樣的概念，
在此後的 GAN 研究中是一個反復出現的主題，對於側重於 image-to-image 或 text-to-image 的論文尤其重要。

Conditional GAN 架構：除了隨機雜訊向量 z 之外，類標籤 y 被連接在一起作為網路的輸入
```
```
Conditional Generative Adversarial Nets
Mehdi Mirza, Simon Osindero
https://arxiv.org/abs/1411.1784
```

### DCGAN(2015)
```
深度學習中對影象處理應用最好的模型是CNN，如何把CNN與GAN結合？

DCGAN是這方面最好的嘗試之一
```
```
Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
Alec Radford, Luke Metz, Soumith Chintala
(Submitted on 19 Nov 2015 (v1), last revised 7 Jan 2016 (this version, v2))

In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. 

In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. 

We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), 
that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. 

Training on various image datasets, we show convincing evidence that 
our deep convolutional adversarial pair learns a hierarchy of representations 
from object parts to scenes in both the generator and discriminator. 

Additionally, we use the learned features for novel tasks - 
demonstrating their applicability as general image representations.
```

```
DCGAN的原理和GAN是一樣的。
它只是把上述的G和D換成了兩個卷積神經網路（CNN）。

但不是直接換就可以了，DCGAN對卷積神經網路的結構做了一些改變，以提高樣本的質量和收斂的速度，這些改變有：

[1]取消所有pooling層。
   G網路中使用轉置卷積（transposed convolutional layer）進行上取樣，D網路中用加入stride的卷積代替pooling。
[2]在D和G中均使用batch normalization
[3]去掉FC層，使網路變為全卷積網路
[4]G網路中使用ReLU作為啟用函式，最後一層使用tanh
[5]D網路中使用LeakyReLU作為啟用函式
```
```
從頭開始GAN【論文】(二) —— DCGAN
https://www.itread01.com/content/1547274425.html

深度卷积对抗生成网络(DCGAN)
https://blog.csdn.net/stdcoutzyx/article/details/53872121

Tensorflow Day29 DCGAN with MNIST[2017]
https://ithelp.ithome.com.tw/articles/10188990
```
#### 各種實作
```
https://paperswithcode.com/paper/unsupervised-representation-learning-with-1
深度卷積生成對抗模型（DCGAN） https://github.com/Newmu/dcgan_code

[TensorFlow實作0.12.1] https://github.com/carpedm20/DCGAN-tensorflow
[Keras實作]：https://github.com/jacobgil/keras-dcgan
[Torch實作(使用lua開發)]：https://github.com/soumith/dcgan.torch
[PyTorch實作]https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
```
#### TFGAN
```
https://github.com/tensorflow/models/tree/master/research/gan
```
#### PYTORCH HUB(2019)實作
```
[PYTORCH HUB(2019)]https://pytorch.org/hub
      [https://www.infoq.cn/article/xM62FoE3K-mOhHRyPO7c]
      [https://pytorch.org/blog/towards-reproducible-research-with-pytorch-hub/]
PyTorch Hub API 手冊連結： https://pytorch.org/docs/stable/hub.html
模型提交位址： https://github.com/pytorch/hub
流覽 PyTorch Hub 網頁以學習更多可用模型： https://pytorch.org/hub
在 Paper with Code 上流覽更多模型： https://paperswithcode.com/
```

### iGAN(2016)
```
"Generative Visual Manipulation on the Natural Image Manifold", 
Jun-Yan Zhu, Philipp Krähenbühl, Eli Shechtman and Alexei A. Efros.  
In European Conference on Computer Vision (ECCV). 2016.
(Submitted on 12 Sep 2016 (v1), last revised 16 Dec 2018 (this version, v3))

https://arxiv.org/abs/1609.03552
http://people.csail.mit.edu/junyanz/projects/gvm/
https://github.com/junyanz/iGAN

https://www.youtube.com/watch?v=9c4z6YsBGQ0
```
### Pix2Pix(2016)
```
Image-to-Image Translation with Conditional Adversarial Networks
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros
https://arxiv.org/abs/1611.07004

https://www.tensorflow.org/tutorials/generative/pix2pix
```
```
Pix2Pix 是另一種圖像到圖像轉換的 GAN 模型。
該框架使用成對的訓練樣本，並在GAN 模型中使用多種不同的配置。

讀這篇論文時，我覺得最有趣部分是關於 PatchGAN的討論。
PatchGAN 通過觀察圖像的 70×70 的區域來判斷它們是真的還是假的，而不是查看整個圖像。
該模型還展示了一個有趣的 U-Net 風格的生成器架構，以及在生成器模型中使用 ResNet 風格的 skip connections。
Pix2Pix 有很多很酷的應用，比如將草圖轉換成逼真的照片。
```


## StackGAN(2016)==StackGAN++(2017)
```
StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks
Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, Dimitris Metaxas
https://arxiv.org/abs/1612.03242

StackGAN 的論文與本列表中的前幾篇論文相比非常不同。
它與 Conditional GAN 和Progressively Growing GANs 最為相似。
StackGAN 模型的工作原理與 Progressively Growing GANs 相似，因為它可以在多個尺度上工作。
StackGAN 首先輸出解析度為64×64 的圖像，然後將其作為先驗資訊生成一個 256×256 解析度的圖像。

StackGAN是從自然語言文本生成圖像。
這是通過改變文本嵌入來實現的，以便捕獲視覺特徵。
這是一篇非常有趣的文章，如果 StyleGAN 中顯示的潛在空間控制
與 StackGAN 中定義的自然語言介面相結合，想必會非常令人驚訝。
```
```
StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks
Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, Dimitris Metaxas
(Submitted on 19 Oct 2017 (v1), last revised 28 Jun 2018 (this version, v3))
https://arxiv.org/abs/1710.10916
```
```

https://kknews.cc/zh-tw/tech/ngeqmq5.html
```
### 

```
[pix2pix]: Torch implementation for learning a mapping from input images to output images.
[CycleGAN]: Torch implementation for learning an image-to-image translation (i.e., pix2pix) without input-output pairs.
[pytorch-CycleGAN-and-pix2pix]: PyTorch implementation for both unpaired and paired image-to-image translation.
```

```
InfoGAN:基於資訊最大化GANs的可解釋表達學習（InfoGAN:Interpretable Representation Learning by Information Maximizing Generative     Adversarial Nets）2016

原文連結：https://arxiv.org/pdf/1606.03657v1.pdf
```
