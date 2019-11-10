# AI_GAN_強化的偵測技術

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
