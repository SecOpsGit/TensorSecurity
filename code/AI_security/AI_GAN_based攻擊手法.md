
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
```
dagrate/gan_network
https://github.com/dagrate/gan_network
```
# 分散式IDS
```
AI強化的偵測技術 for IOT
Generative Adversarial Networks for Distributed Intrusion Detection in the Internet of Things
Aidin Ferdowsi, Walid Saad
(Submitted on 3 Jun 2019)
https://arxiv.org/abs/1906.00567

................
Moreover, in many scenarios such as health and financial applications, the datasets are private 
and IoTDs may not intend to share such data. 

To this end, in this paper, a distributed generative adversarial network (GAN) is proposed 
to provide a fully distributed IDS for the IoT so as to detect anomalous behavior without reliance on any centralized controller. 

In this architecture, every IoTD can monitor its own data as well as neighbor IoTDs to detect internal and external attacks. 

In addition, the proposed distributed IDS does not require sharing the datasets between the IoTDs, 
thus, it can be implemented in IoTs that preserve the privacy of user data such as health monitoring systems or financial applications. 

It is shown analytically that the proposed distributed GAN has higher accuracy of detecting intrusion compared to a standalone IDS that has access to only a single IoTD dataset. 

Simulation results show that, the proposed distributed GAN-based IDS has up to 20% higher accuracy, 
25% higher precision, and 60% lower false positive rate compared to a standalone GAN-based IDS.
```

### GIDS (GAN based Intrusion Detection System)[]
```
AI強化的偵測技術 FOR 車聯網

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
偵測到4個未知攻擊
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

Generative adversarial network (GAN) has been proposed to handle the case. 

With its strong generative ability,  it only needs to learn the distribution of normal status, 
and identify the abnormal status through the gap between it and the learned distribution. 

Nevertheless, existing GAN-based models are not suitable to process data with discrete values, leading to immense degradation of detection performance. 

To cope with the discrete features, in this paper, we propose an efficient GAN-based model with specifically-designed loss function. Experiment results show that our model outperforms state-of-the-art models on discrete dataset and remarkably reduce the overhead.
```

### IDSGAN:產生無法被IDS偵測到的攻擊
```
https://arxiv.org/abs/1809.02077

IDSGAN: Generative Adversarial Networks for Attack Generation against Intrusion Detection
Zilong Lin, Yong Shi, Zhi Xue
(Submitted on 6 Sep 2018 (v1), last revised 16 Jun 2019 (this version, v3))

As an important tool in security, the intrusion detection system bears the responsibility of the defense to network attacks 
performed by malicious traffic. 
Nowadays, with the help of machine learning algorithms, the intrusion detection system develops rapidly. 
However, the robustness of this system is questionable when it faces the adversarial attacks. 

To improve the detection system, more potential attack approaches should be researched. 

In this paper, a framework of the generative adversarial networks, IDSGAN, is proposed to generate the adversarial attacks, 
which can deceive and evade the intrusion detection system. 

Considering that the internal structure of the detection system is unknown to attackers, 
adversarial attack examples perform the black-box attacks against the detection system. 

IDSGAN leverages a generator to transform original malicious traffic into adversarial malicious traffic. 
A discriminator classifies traffic examples and simulates the black-box detection system. 

More significantly, we only modify part of the attacks' nonfunctional features to guarantee the validity of the intrusion. 
Based on the dataset NSL-KDD, the feasibility of the model is demonstrated to attack many detection systems with different attacks and the excellent results are achieved. 

Moreover, the robustness of IDSGAN is verified by changing the amount of the unmodified features.
```
