#


# dataset 
```
ISCX Botnet Dataset 
http://www.iscx.ca/publications/
Information Centre of Excellence for Tech Innovation

Canadian Institute for Cybersecurity datasets 
https://www.unb.ca/cic/datasets/index.html


https://www.unb.ca/cic/datasets/botnet.html

Distribution of botnet types in the training dataset

Botnet name | Type | Portion of flows in dataset
Neris | IRC | 21159 (12%)
Rbot | IRC | 39316 (22%)
Virut | HTTP | 1638 (0.94 %)
NSIS | P2P | 4336 (2.48%)
SMTP Spam | P2P | 11296 (6.48%)
Zeus | P2P | 31 (0.01%)
Zeus control (C & C) | P2P | 20 (0.01%)

Distribution of botnet types in the test dataset

Botnet name | Type | Portion of flows in dataset
Neris | IRC | 25967 (5.67%)
Rbot | IRC | 83 (0.018%)
Menti | IRC | 2878(0.62%)
Sogou | HTTP | 89 (0.019%)
Murlo | IRC | 4881 (1.06%)
Virut | HTTP | 58576 (12.80%)
NSIS | P2P | 757 (0.165%)
Zeus | P2P | 502 (0.109%)
SMTP Spam | P2P | 21633 (4.72%)
UDP Storm | P2P | 44062 (9.63%)
Tbot | IRC | 1296 (0.283%)
Zero Access | P2P | 1011 (0.221%)
Weasel | P2P | 42313 (9.25%)
Smoke Bot | P2P | 78 (0.017%)
Zeus Control (C&C) | P2P | 31 (0.006%)
ISCX IRC bot | P2P | 1816 (0.387%)
```
```
CTU University Dataset.
```
```
殭屍網絡及DDoS數據集
https://www.twblogs.net/a/5caf8318bd9eee48d7883991
```
#
```
https://arxiv.org/abs/1907.08276
An AI-based, Multi-stage detection system of banking botnets
Li Ling, Zhiqiang Gao, Michael A Silas, Ian Lee, Erwan A Le Doeuff
(Submitted on 18 Jul 2019 (v1), last revised 25 Jul 2019 (this version, v3))

Banking Trojans, botnets are primary drivers of financially-motivated cybercrime. 

In this paper, we first analyzed how an APT-based banking botnet works step by step through the whole lifecycle. 
Specifically, we present a multi-stage system that detects malicious banking botnet activities 
which potentially target the organizations. 
The system leverages Cyber Data Lake as well as multiple artificial intelligence techniques at different stages. 
The evaluation results using public datasets showed that Deep Learning based detections were highly successful 
compared with baseline models.

```
# p2p botnet detection
```
J. Zhang, R. Perdisci, W. Lee, X. Luo, and U. Sarfraz, 
“Building a scalable system for stealthy p2p-botnet detection,” 
Information Forensics and Security, IEEE Transactions on, vol. 9, no. 1, pp. 27–38, 2014.

```
# ref
```
1. Mohammad Aluthaman, Nauman Aslam, Li Zhang and Rafe Aslem, 
"A P2P Botnet Detection Scheme based on Decision Tree and Adaptive Multi-layer Neural Networks”. 
Journal of Neural Computing and Applications, 2016


6. Stephen Doswell, Nauman Aslam, David Kendall and Graham Sexton, "Please slow down! The impact on Tor performance from mobility", 3rd Annual ACM CCS Workshop on Security and Privacy in Smartphones and Mobile Devices (SPSM), in conjunction with the 20th ACM Conference on Computer and Communications Security (CCS), November 4-8, 2013, Berlin, Germany.
7. Abidalrahman Mohammad, Nauman Aslam, William Phillips and William Robertson, “C-Sec: Energy Efficient Link Layer Encryption Protocol for Wireless Sensor Networks", 9th IEEE Consumer Communication and Networking Conference, Las Vegas, USA, Jan 14 - 17, 2012
```
```
RTbust: Exploiting Temporal Patterns for Botnet Detection on Twitter
Michele Mazza, Stefano Cresci, Marco Avvenuti, Walter Quattrociocchi, Maurizio Tesconi
(Submitted on 12 Feb 2019)

Within OSNs, many of our supposedly online friends may instead be fake accounts called social bots, part of large groups that purposely re-share targeted content. Here, we study retweeting behaviors on Twitter, with the ultimate goal of detecting retweeting social bots. We collect a dataset of 10M retweets. We design a novel visualization that we leverage to highlight benign and malicious patterns of retweeting activity. In this way, we uncover a 'normal' retweeting pattern that is peculiar of human-operated accounts, and 3 suspicious patterns related to bot activities. Then, we propose a bot detection technique that stems from the previous exploration of retweeting behaviors. Our technique, called Retweet-Buster (RTbust), leverages unsupervised feature extraction and clustering. An LSTM autoencoder converts the retweet time series into compact and informative latent feature vectors, which are then clustered with a hierarchical density-based algorithm. Accounts belonging to large clusters characterized by malicious retweeting patterns are labeled as bots. RTbust obtains excellent detection results, with F1 = 0.87, whereas competitors achieve F1 < 0.76. Finally, we apply RTbust to a large dataset of retweets, uncovering 2 previously unknown active botnets with hundreds of accounts.
```
