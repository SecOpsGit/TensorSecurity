
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

### IDSGAN:產生無法被IDS偵測到的攻擊[2018]
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
