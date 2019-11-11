# Domain generation algorithms (DGAs) 
```
Domain generation algorithms (DGAs) are commonly used by botnets to generate domain names through 
which bots can establish a resilient communication channel with their command and control servers.

Domain generation algorithms are used to generate a large number of pseudo-random domain names, 
which are usually based on the date and a secret input (seed). 

A bot and its command and control server that wish to communicate will both execute the DGA
with a shared seed in order to generate a sequence of domain names
and identify the one through which their communication can take place. 

DGAs can be used to generate thousands of domain names per day which must be identified and analyzed in order to shutdown
of the botnet.

```

# Domain generation algorithms (DGAs) 分類器
```
deep learning, character-level classifiers that are able to detect algorithmically generated domain (AGD) names with high accuracy, 
and correspondingly, significantly reduce the effectiveness of DGAs for botnet communication. 

The detection of algorithmically generated domain (AGD) names initially focused on capturing binary samples of bots, extracting
their algorithms and seeds, and generating the domain names in advance for mitigation [35, 37, 43]. 

However, by using new input seeds, this approach can be easily evaded by botnets. 
In fact, between the years of 2017 and 2018 at least 150 new seeds were introduced by botnets,1
a figure which is more than two times the number of documented DGAs. 

Using machine learning in order to inspect the lexicographic patterns of domain names for detecting AGD names
and classify their generating algorithms [46, 52] is an alternative and more generalized approach.
```
### Machine learning-based DGA detection techniques 
```
have become extremely successful, with state of the art algorithms achieving high
detection rates on multiple datasets [42] with inline latency [23].

These techniques reduce the effectiveness of DGAs for maintaining resilient and stealthy botnet communication. 

```
### Adversarial machine learning
```
From the botnet operator’s perspective, a solution that can evade these state of the art detection mechanisms 
by using adversarial machine learning
to produce AGD names that are less likely to be detected can be beneficial

Adversarial machine learning is a technique in the field of machine learning which attempts to “fool” models (during either the
training or execution phases) through adversarial input [25], also referred to as adversarial samples.
```
#### DeepDGA(2016)
```
Hyrum S. Anderson, Jonathan Woodbridge, and Bobby Filar. 2016. 
DeepDGA: Adversarially-Tuned Domain Generation and Detection. 13–21. 
https://doi.org/10.1145/2996758.2996767
(Submitted on 6 Oct 2016)
https://arxiv.org/abs/1610.01969
```
#### MaskDGA(2019)
```
https://arxiv.org/abs/1902.08909
MaskDGA: A Black-box Evasion Technique Against DGA Classifiers and Adversarial Defenses
Lior Sidi, Asaf Nadler, Asaf Shabtai
(Submitted on 24 Feb 2019)

MaskDGA, a black-box adversarial learning technique 
 that adds perturbations to a character level representation of an AGD and transforms it to a new domain name that is falsely
classified as benign by DGA classifiers, without prior knowledge 1Based on the change-log of https://data.netlab.360.com/dga/
arXiv:1902.08909v1 [cs.CR] 24 Feb 2019
```
#### HotFlip(2017)
```
HotFlip: White-Box Adversarial Examples for Text Classification
Javid Ebrahimi, Anyi Rao, Daniel Lowd, Dejing Dou
(Submitted on 19 Dec 2017 (v1), last revised 24 May 2018 (this version, v2))
https://arxiv.org/abs/1712.06751
```
### DataSet
```
DMD-2018 dataset [27,47–49] 
which contains 100,000 benign domain names and 297,777 algorithmically generated domains that were evenly produced by
twenty different DGA families (see Table 1).

```

### 四個DGA分類器

```
Endgame [3] which is based on a single LSTM layer (denoted as LSTM[Endgame])
CMU [15] which is based on a forward LSTM layer and a backward LSTM layer (denoted as biLSTM[CMU])
Invincea [40] which based on parallel CNN layers (denoted as CNN[Invincea])
MIT [50] which based on stacked CNN layers and a single LSTM layer (denoted as CNN + LSTM[MIT])

```
```
Encrypted and Covert DNS Queries for Botnets: Challenges and Countermeasures
Constantinos Patsakis, Fran Casino, Vasilios Katos
(Submitted on 16 Sep 2019)
https://arxiv.org/abs/1909.07099
```

# 攻擊DGAs分類器

### MaskDGA(2019)
```
https://arxiv.org/abs/1902.08909
MaskDGA, a practical adversarial learning technique that adds perturbation to the 
character-level representation of algorithmically generated domain names in order to evade DGA classifiers,
without the attacker having any knowledge about the DGA classifier's architecture and parameters.

MaskDGA was evaluated using the DMD-2018 dataset of AGD names and four recently published DGA classifiers, 
in which the average F1-score of the classifiers degrades from 0.977 to 0.495 when applying the evasion technique. 

An additional evaluation was conducted using the same classifiers but with adversarial defenses implemented: 
adversarial re-training and distillation. 

The results of this evaluation show that MaskDGA can be used for improving the robustness of the character-level DGA classifiers 
against adversarial attacks, but that ideally DGA classifiers should incorporate additional features 
alongside character-level features that are demonstrated in this study to be vulnerable to adversarial attacks.

```

```
[4] Hyrum S Anderson, Jonathan Woodbridge, and Bobby Filar. 2016. DeepDGA:
Adversarially-tuned domain generation and detection. In Proceedings of the 2016
ACM Workshop on Artificial Intelligence and Security. ACM, 13–21.
[5] Manos Antonakakis, Roberto Perdisci, Yacin Nadji, Nikolaos Vasiloglou, Saeed
Abu-Nimeh, Wenke Lee, and David Dagon. 2012. From Throw-Away Traffic to
Bots: Detecting the Rise of DGA-Based Malware.. In USENIX security symposium,
Vol. 12.
[6] Pavol Bielik, Veselin Raychev, and Martin Vechev. 2017. Character Level Based
Detection of Dga Domain Names. 1–17.

[7] Battista Biggio, Igino Corona, Giorgio Fumera, Giorgio Giacinto, and Fabio Roli.
2011. Bagging classifiers for fighting poisoning attacks in adversarial classification
tasks. In International workshop on multiple classifier systems. Springer, 350–359.

[8] Battista Biggio, Igino Corona, Davide Maiorca, Blaine Nelson, Nedim Šrndić,
Pavel Laskov, Giorgio Giacinto, and Fabio Roli. 2013. 
Evasion attacks against
machine learning at test time. In Joint European conference on machine learning
and knowledge discovery in databases. Springer, 387–402.

[9] Battista Biggio, Blaine Nelson, and Pavel Laskov. 2012. Poisoning attacks against
support vector machines. arXiv preprint arXiv:1206.6389 (2012).

[10] Nicholas Carlini and David Wagner. 2017. 
Towards evaluating the robustness of neural networks. 
In 2017 IEEE Symposium on Security and Privacy (SP). IEEE,
39–57.



[12] Chhaya Choudhary, Raaghavi Sivaguru, Mayana Pereira, Anderson C Nascimento, and Martine De Cock. [n. d.]. 
Algorithmically Generated Domain Detection and Malware Family Classification. 1–15. 
https://umbrella.cisco.com/blog/
2016/12/14/cisco-umbrella-1-million/,

[13] Ryan R Curtin, Andrew B Gardner, Slawomir Grzonkowski, Alexey Kleymenov,
and Alejandro Mosquera. 2018. 
Detecting DGA domains with recurrent neural networks and side information. arXiv preprint arXiv:1810.02023 (2018).

[14] Prithviraj Dasgupta, Joseph Collins, and Anna Buhman. 2018. 
Gray-box Techniques for Adversarial Text Generation. 18–19.

[15] Bhuwan Dhingra, Zhong Zhou, Dylan Fitzpatrick, Michael Muehl, and William W
Cohen. 2016. Tweet2vec: Character-based distributed representations for social
media. arXiv preprint arXiv:1605.03481 (2016).

[16] Javid Ebrahimi, Anyi Rao, Daniel Lowd, and Dejing Dou. 2014. 
HotFlip: WhiteBox Adversarial Examples for Text Classification. (2014).

[17] Ji Gao, Jack Lanchantin, Mary Lou Soffa, and Yanjun Qi. 2018. 
Black-box generation of adversarial text sequences to evade deep learning classifiers. 
Proceedings -2018 IEEE Symposium on Security and Privacy Workshops, SPW 2018 (2018), 50–56.
https://doi.org/10.1109/SPW.2018.00016



[19] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. [n. d.]. Explaining
and harnessing adversarial examples. CoRR (2015). ([n. d.]).

[20] Weiwei Hu and Ying Tan. 2017. 
Generating Adversarial Malware Examples forBlack-Box Attacks Based on GAN. (2017). 
http://arxiv.org/abs/1702.05983

[21] Jonathan Hui. 2018. GAN - Why it is so hard to train Generative Adversarial Networks! (June 2018). 
https://medium.com/@jonathan_hui/
gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b



[23] Joewie J Koh and Barton Rhodes. 2018. 
Inline Detection of Domain Generation Algorithms with Context-Sensitive Word Embeddings. 
arXiv preprint arXiv:1811.08705 (2018).

[25] Alexey Kurakin, Ian J. Goodfellow, and Samy Bengio. 2017. 
Adversarial Machine Learning at Scale. https://arxiv.org/abs/1611.01236



[27] Vysakh S Mohan, R Vinayakumar, KP Soman, and Prabaharan Poornachandran.
2018. Spoof net: Syntactic patterns for identification of ominous online factors.
In 2018 IEEE Security and Privacy Workshops (SPW). IEEE, 258–263.

[28] Asaf Nadler, Avi Aminov, and Asaf Shabtai. 2019. Detection of malicious and
low throughput data exfiltration over the DNS protocol. 
Computers & Security 80 (2019), 36–53.

[29] Nicolas Papernot, Fartash Faghri, Nicholas Carlini, Ian Goodfellow, Reuben Feinman, Alexey Kurakin, 
Cihang Xie, Yash Sharma, Tom Brown, Aurko Roy, Alexander Matyasko, Vahid Behzadan, Karen Hambardzumyan, Zhishuai Zhang, Yi-Lin
Juang, Zhi Li, Ryan Sheatsley, Abhibhav Garg, Jonathan Uesato, Willi Gierke,
Yinpeng Dong, David Berthelot, Paul Hendricks, Jonas Rauber, and Rujun Long. 
Technical Report on the CleverHans v2.1.0 Adversarial Examples Library.
arXiv preprint arXiv:1610.00768 (2018).

[30] Nicolas Papernot, Patrick McDaniel, and Ian Goodfellow. 2016. 
Transferability in Machine Learning: from Phenomena to Black-Box Attacks using Adversarial
```
