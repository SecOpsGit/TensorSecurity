
# AI_based Attacks: powered by GAN
```
```
### Bot-Evading-Machine-Learning-Malware-Detection
```
https://www.blackhat.com/docs/us-17/thursday/
us-17-Anderson-Bot-Vs-Bot-Evading-Machine-Learning-Malware-Detection-wp.pdf
```
### PassGAN(2017)
```
PassGAN: A Deep Learning Approach for Password Guessing
Briland Hitaj, Paolo Gasti, Giuseppe Ateniese, Fernando Perez-Cruz
(Submitted on 1 Sep 2017 (v1), last revised 14 Feb 2019 (this version, v3))
https://arxiv.org/abs/1709.00440
https://github.com/brannondorsey/PassGAN

```
### MalGAN(2017)
```
Generating Adversarial Malware Examples for Black-Box Attacks Based on GAN
Weiwei Hu, Ying Tan
(Submitted on 20 Feb 2017)

https://arxiv.org/abs/1702.05983
https://github.com/yanminglai/Malware-GAN

https://medium.com/@falconives/day-67-malgan-b178bb5f975e
```
###

```
Improved MalGAN: Avoiding Malware Detector by Leaning Cleanware Features
https://ieeexplore.ieee.org/document/8669079

```
###  AI-Based隱寫術
```
Recent Advances of Image Steganography with Generative Adversarial Networks
Jia Liu, Yan Ke, Yu Lei, Zhuo Zhang, Jun Li, Peng Luo, Minqing Zhang, Xiaoyuan Yang
(Submitted on 18 Jun 2019)

https://arxiv.org/abs/1907.01886


Image steganography is dedicated to hiding secret messages in digital images, 
and has achieved the purpose of covert communication. 

Recently, research on image steganography has demonstrated great potential for using GAN and neural networks. 

In this paper we review different strategies for steganography such as cover modification, 
cover selection and cover synthesis by GANs, 
and discuss the characteristics of these methods as well as evaluation metrics 
and provide some possible future research directions in image steganography.
```
```
a stego-security classification is proposed based on the four levels of steganalysis attacks:
a) Stego-Cover Only Attack (SCOA): the steganalysis attacker can only access a set of stego-covers.
b) Known Cover Attack (KCA): being able to perform SCOA, the attacker can also obtain
some original cover carriers and their corresponding stego carriers. Within polynomial complexity, the
number of pairs is limited.
c) Chosen Cover Attack (CCA): an attacker can use the steganographic algorithm to
perform multiple message embedding and extraction operations with a priori knowledge under KCA.
Within polynomial complexity, the number of invocation operations is limited.

d) Adaptive Chosen Cover Attack (ACCA): The ACCA mode means that when the CCA mode
challenge fails, another CCA attacks can be performed until the attack is successful.
```
```
EncryptGAN: Image Steganography with Domain Transform
Ziqiang Zheng, Hongzhi Liu, Zhibin Yu, Haiyong Zheng, Yang Wu, Yang Yang, Jianbo Shi
(Submitted on 28 May 2019 (v1), last revised 29 May 2019 (this version, v2))
https://arxiv.org/abs/1905.11582

```
```
https://arxiv.org/abs/1804.07939
Spatial Image Steganography Based on Generative Adversarial Network
Jianhua Yang, Kai Liu, Xiangui Kang, Edward K.Wong, Yun-Qing Shi
(Submitted on 21 Apr 2018)
```
### SynGAN(2019)
```
SynGAN: Towards Generating Synthetic Network Attacks using GANs
Jeremy Charlier, Aman Singh, Gaston Ormazabal, Radu State, Henning Schulzrinne
(Submitted on 26 Aug 2019)
https://arxiv.org/abs/1908.09899

SynGAN, a framework that generates adversarial network attacks using the Generative Adversial Networks (GAN). 
SynGAN generates malicious packet flow mutations using real attack traffic, which can improve NIDS attack detection rates. 
As a first step, we compare two public datasets, NSL-KDD and CICIDS2017, 
for generating synthetic Distributed Denial of Service (DDoS) network attacks. 

We evaluate the attack quality (real vs. synthetic) using a gradient boosting classifier.
```
# 
```
Generative Adversarial Networks for Distributed Intrusion Detection in the Internet of Things
Aidin Ferdowsi, Walid Saad
(Submitted on 3 Jun 2019)
https://arxiv.org/abs/1906.00567
```

### GIDS (GAN based Intrusion Detection System)
```
GIDS: GAN based Intrusion Detection System for In-Vehicle Network
Eunbi Seo, Hyun Min Song, Huy Kang Kim
(Submitted on 17 Jul 2019)
https://arxiv.org/abs/1907.07377

Car-Hacking Dataset for the intrusion detection
http://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset


A Controller Area Network (CAN) bus in the vehicles is an efficient standard bus 
enabling communication between all Electronic Control Units (ECU). 

However, CAN bus is not enough to protect itself because of lack of security features. 
To detect suspicious network connections effectively, the intrusion detection system (IDS) is strongly required. 

Unlike the traditional IDS for Internet, there are small number of known attack signatures for vehicle networks. 
Also, IDS for vehicle requires high accuracy because any false-positive error can seriously affect the safety of the driver. 

To solve this problem, we propose a novel IDS model for in-vehicle networks, GIDS (GAN based Intrusion Detection System) 
using deep-learning model, Generative Adversarial Nets. 

GIDS can learn to detect unknown attacks using only normal data. 
As experiment result, GIDS shows high detection accuracy for four unknown attacks.
```
### https://arxiv.org/abs/1904.02426
```
Efficient GAN-based method for cyber-intrusion detection
Hongyu Chen, Li Jiang
(Submitted on 4 Apr 2019 (v1), last revised 24 Jul 2019 (this version, v2))
https://arxiv.org/abs/1904.02426

Ubiquitous anomalies endanger the security of our system constantly. 

They may bring irreversible damages to the system and cause leakage of privacy. 

Thus, it is of vital importance to promptly detect these anomalies. 

Traditional supervised methods such as Decision Trees and Support Vector Machine (SVM) are used to classify normality and abnormality. 

However, in some case the abnormal status are largely rarer than normal status, 
which leads to decision bias of these methods. 

Generative adversarial network (GAN) has been proposed to handle the case. With its strong generative ability, 
it only needs to learn the distribution of normal status, and identify the abnormal status through the gap between it and the learned distribution. 

Nevertheless, existing GAN-based models are not suitable to process data with discrete values, leading to immense degradation of detection performance. 

To cope with the discrete features, in this paper, we propose an efficient GAN-based model with specifically-designed loss function. Experiment results show that our model outperforms state-of-the-art models on discrete dataset and remarkably reduce the overhead.

```
