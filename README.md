# TensorSecurity

```
https://scikit-learn.org/stable/modules/classes.html
TensorFlow Machine Learning Cookbook, Nick McClure(2017)
https://github.com/nfmcclure/tensorflow_cookbook

```
```
http://speech.ee.ntu.edu.tw/~tlkagk/index.html
```
# Tensorflow 人工智慧與資訊安全應用實務課程

### 學員自我評量
```
完成多少作業?   完成多少研讀作業?    參加AI競賽!!!
```
# A.人工智慧及開發平台
### 人工智慧範疇
```
人工智慧    機器學習    深度學習與強化學習
```
## 機器學習類型:
```
1.監督學習Supervised learning   從有Label的資料學習
2.無監督學習 Unsupervised learning
3.半監督學習 semi-Supervised learning
4.增強學習Reinforcement learning
```
### 監督學習Supervised learning  
```

https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
1.11. Ensemble methods
1.17. Neural network models (supervised)  
   1.17.1. MLP==Multi-layer Perceptron
      1.17.2. Classification==> Class MLPClassifier類別
      1.17.3. Regression==> Class MLPRegressor類別
```
### 無監督學習 Unsupervised learning
```

```



### 人工智慧開發平台

```
Tensorflow(2015,Google)
PyTorch(2017, FaceBook)
Apache MXNet
   https://github.com/apache/incubator-mxnet
```
```
Comparison of deep-learning software
https://en.wikipedia.org/wiki/Comparison_of_deep-learning_software
```
## Tensorflow Ecosystem
```
Tensorflow Ecosystem
Tensor及其運算
```

### 學員作業
```

```

### From Numpy to Pandas:A Quick Tour[請參閱上本教材]
```
Numpy 
Data Visualization資料視覺化:Matplotlib/seaborn/Plotit
Pandas@Colab CSV資料讀取技術
```
### 學員作業
```
Numpy_ex1
Matplotlib_ex1:完成五張Data Visualization資料視覺化作業
Pandas_ex1:Pandas基本運算
Pandas_ex2:Pandas讀取Mysql資料庫資料實作
```
### 學員研讀作業:Pandas
```
Python資料分析 第二版 Python for Data Analysis, 2nd Edition
作者：Wes McKinney  譯者： 張靜雯 歐萊禮出版社 出版日期：2018/10/03

Python 資料運算與分析實戰：一次搞懂 NumPy•SciPy•Matplotlib•pandas 最強套件
作者： 中久喜健司  譯者： 莊永裕  旗標出版社：  出版日期：2018/02/05

Pandas資料分析實戰：使用Python進行高效能資料處理及分析
Learning pandas - Second Edition: High-performance data manipulation and analysis in Python
作者： Michael Heydt  譯者： 陳建宏   博碩出版社：  出版日期：2019/08/22
```

```
NumPy 官網 http://www.numpy.org/
NumPy 原始程式碼：https://github.com/numpy/numpy
SciPy 官網：https://www.scipy.org/
SciPy 原始程式碼：https://github.com/scipy/scipy
Matplotlib 官網：https://matplotlib.org/
Matplotlib 原始程式碼：https://github.com/matplotlib/matplotlib
Pandas官網 https://pandas.pydata.org/
pandas原始程式碼：https://github.com/pandas-dev/pandas
```
# B. Machine Learning
```
Survey of Machine Learning Techniques for Malware Analysis
Daniele Ucci, Leonardo Aniello, Roberto Baldoni
(Submitted on 23 Oct 2017 (v1), last revised 26 Nov 2018 (this version, v3))
https://arxiv.org/abs/1710.08189
https://www.sciencedirect.com/science/article/pii/S0167404818303808


A state-of-the-art survey of malware detection approaches using data mining techniques
Alireza Souri & Rahil Hosseini 
Human-centric Computing and Information Sciencesvolume 8, Article number: 3 (2018) 

```
## Machine Learning:scikit-learn vs Tensorflow

scikit-learn

## B1.回歸與預測

### B1.1:回歸的類型

### B1.2 scikit-learn技術與實作

#### scikit-learn支援的Regression技術
```
https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
1.1. Generalized Linear Models
1.1.1. Ordinary Least Squares
1.1.2. Ridge Regression
1.1.3. Lasso
1.1.4. Multi-task Lasso
1.1.5. Elastic-Net
1.1.6. Multi-task Elastic-Net
1.1.7. Least Angle Regression
1.1.8. LARS Lasso
1.1.9. Orthogonal Matching Pursuit (OMP)
1.1.10. Bayesian Regression
1.1.11. Logistic regression
1.1.12. Stochastic Gradient Descent - SGD
1.1.13. Perceptron
1.1.14. Passive Aggressive Algorithms

1.1.15. Robustness regression: outliers and modeling errors
1.1.15.5. Notes

1.1.16. Polynomial regression: extending linear models with basis functions
```
#### scikit-learn的Regression技術應用
```
sklearn_1_
sklearn_2_
```
### B1.3 Tensorflow
```
TF_1
TF_2
```
### B1.4.[資安應用]金融詐欺偵測之實作
```
PwC大中華區首席資料科學家揭露金融業4大AI成熟應用
AI技術在金融業較成熟的4大應用是對抗詐欺風險、管控經營風險、消除風險和精準行銷。
https://www.ithome.com.tw/news/117607

中國一家排名前三大的保險公司，將AI技術應用在反詐欺上，根據過去20年數據，將每個核賠案件，依照特徵貼上標籤並分類，透過機器學習技術，
學習類別的特徵，自動比對分析後，就能快速辨識出可能詐欺的案件，
另外，還要加入圖像識別的技術，處理文件內容識別，和車禍圖像比對，來辨識圖片資訊，該家保險公司，
透過多項技術的輔助，每年可以識別出3,000萬件欺詐案，他表示，以前都只能用人工來處理，非常費時。

2019 金融業數位轉型趨勢：使用 ML 技術打擊金融詐騙
https://buzzorange.com/techorange/2018/11/27/sas-antifraud/

信用卡詐欺（credit card fraud）
https://report.nat.gov.tw/ReportFront/PageSystem/reportFileDownload/C10601536/001
```

## B2.分類

### B2.1.分類演算法:從Decision tree、 Random Forest到SVM
```
LogisticRegression (LR)
K-Nearest Neighbors (K-NN)
Support Vector Machine (SVM)
KernelSVM
NaiveBayes
DecisionTreeClassification
Random Forest Classification
```
### B2.2.分類演算法的實作
```
Decision tree實作
SVM實作
MLP Neural Network實作
```
### B2.3.[資安應用]惡意網址偵測之實作
```

```
#### 學員作業
```

```
#### 學員研讀作業:
```
spam emial detction
APT detection
Fake news detection
```
### B3.聚類與無監督學習

#### B3.1.無監督學習
```
Clustering 
dimension Reduction: PCA
Deep Learning的 GAN
```
#### scikit-learn支援的 Clustering
```
https://scikit-learn.org/stable/modules/clustering.html#clustering
2.3. Clustering
2.3.1. Overview of clustering methods
2.3.2. K-means
2.3.2.1. Mini Batch K-Means
2.3.3. Affinity Propagation
2.3.4. Mean Shift
2.3.5. Spectral clustering
2.3.5.1. Different label assignment strategies
2.3.5.2. Spectral Clustering Graphs
2.3.6. Hierarchical clustering
2.3.6.1. Different linkage type: Ward, complete, average, and single linkage
2.3.6.2. Adding connectivity constraints
2.3.6.3. Varying the metric
2.3.7. DBSCAN
2.3.8. OPTICS
2.3.9. Birch
```
#### scikit-learn支援的 Dimensionality reduction
```
https://scikit-learn.org/stable/modules/decomposition.html#decompositions
2.5. Decomposing signals in components (matrix factorization problems)
2.5.1. Principal component analysis (PCA)
   2.5.1.1. Exact PCA and probabilistic interpretation
   2.5.1.2. Incremental PCA
   2.5.1.3. PCA using randomized SVD
   2.5.1.4. Kernel PCA
   2.5.1.5. Sparse principal components analysis (SparsePCA and MiniBatchSparsePCA)
2.5.2. Truncated singular value decomposition and latent semantic analysis
2.5.3. Dictionary Learning
   2.5.3.1. Sparse coding with a precomputed dictionary
   2.5.3.2. Generic dictionary learning
   2.5.3.3. Mini-batch dictionary learning
2.5.4. Factor Analysis
2.5.5. Independent component analysis (ICA)
2.5.6. Non-negative matrix factorization (NMF or NNMF)
   2.5.6.1. NMF with the Frobenius norm
   2.5.6.2. NMF with a beta-divergence
2.5.7. Latent Dirichlet Allocation (LDA)
```
#### B3.2.K-Means實戰
```
10 Interesting Use Cases for the K-Means Algorithm
https://dzone.com/articles/10-interesting-use-cases-for-the-k-means-algorithm


https://perso.telecom-paristech.fr/qleroy/aml/lab5.html
```
```
A Cybersecurity Analysis of the National Vulnerability Database Using k-means Clustering
https://www.kaggle.com/walterhaydock/cybersecurity-vulnerability-analysis-with-k-means
```
####  Fraud detection using K-Means clustering

```
https://www.kaggle.com/isaikumar/credit-card-fraud-detection-using-k-means-and-knn

Credit Card Fraud Detection Dataset
https://www.kaggle.com/isaikumar/creditcardfraud
```
# C.Deep Leaning
```
Application of deep learning to cybersecurity: A survey
Samaneh Mahdavifar Ali A.Ghorbani
https://www.sciencedirect.com/science/article/pii/S0925231219302954


A short review on Applications of Deep learning for Cyber security
Mohammed Harun Babu R, Vinayakumar R, Soman KP
(Submitted on 15 Dec 2018 (v1), last revised 30 Jan 2019 (this version, v2))
https://arxiv.org/abs/1812.06292

Distributed attack detection scheme using deep learning approach for Internet of Things
Abebe AbeshuDiro  Naveen Chilamkurti
https://www.sciencedirect.com/science/article/pii/S0167739X17308488

https://ccdcoe.org/uploads/2018/10/Art-19-On-the-Effectiveness-of-Machine-and-Deep-Learning-for-Cyber-Security.pdf
```
### Deep Learning and Ransomware Detection
```
Deep Learning for Ransomware Detection
```

# C1.CNN 卷積神經網絡  Convolutional Neural Network
```
CNN Variants
https://www.youtube.com/watch?v=0g2DGh2JQEY


```
## C1.1.Computer Vision機器視覺與CNN

## C1.2_圖像分類與CNN:從ImageNet與ILSVRC (2010-2017)看CNN進展
```
CNN Case Study
https://www.youtube.com/watch?v=h9Gt67Gntvc
```
## C1.3_CNN 實作CNN_lab
```
CNN_lab1:Image Claiification實作
CNN_lab2:Resnet實作
CNN_lab3:Trnasfer learning實作
```
## C1.4_CNN與資安應用
```
圖形驗證碼破解
```

#### 學員研讀作業:
```
Malware Classification惡意程式分類
Microsoft Malware Classification Challenge (BIG 2015)
Classify malware into families based on file content and characteristics
http://arxiv.org/abs/1802.10135
https://www.kaggle.com/c/malware-classification


Applying Supervised Learning on Malware Authorship Attribution
Author: Coen BOOT

```

# C2.RNN遞歸神經網絡Recurrent Neural Network
```
千呼萬喚十多年！中研院終於開源釋出國產自動化中文斷詞工具
中研院近日正式開源釋出了自行研發多年的中文斷詞程式，提供給在臺灣從事
中文自然語言領域研究的學術圈或開發者來使用，目前已放上GitHub平臺。
https://www.ithome.com.tw/news/132838

根據該工具的文件說明，這套CkipTagger中文處理工具，不只提供繁體中文斷詞的功能，
也加入詞性標注和18類專有名詞的實體辨識（Named entity recognition）等功能，
甚至當以多達5萬句的ASBC 4.0漢語語料庫測試集，來進行中文斷詞測試時，
CkipTagger表現遠高於中國的結巴，中研院在中文斷詞準確度可達到97.49%，相較之下，中國的結巴只有90.51%。

新版本也提供幾大特色，包括了加強斷詞表現、可以不自動刪／改字，並且能夠支援不限長度的句子，
另外，新版也加入使用者自訂功能，提供參考／強制 詞典的自訂功能。

不只開源中文斷詞程式，我們今年初採訪該詞庫小組計畫負責人馬偉雲時，他也提到，接下來，
將用4年的時間，將中文維基百科上的百萬詞彙，自動建立其知識圖譜，來打造成為一個具有百萬中文詞庫的知識庫。
他更計畫以現有9萬詞彙所建立的知識圖譜架構，發展相關的自然語言處理工具，可以自動建立其知識圖譜。

Remedying BiLSTM-CNN Deficiency in Modeling Cross-Context for NER
Peng-Hsuan Li, Tsu-Jui Fu, Wei-Yun Ma
(Submitted on 29 Aug 2019)
https://arxiv.org/abs/1908.11046
ckiplab/ckiptagger
https://github.com/ckiplab/ckiptagger
```
```
全臺第一個金融聊天機器人！玉山銀搶先用Chatbot提供3大金融業務諮詢(2017)
申辦貸款、信用卡或是外匯服務，未來將變得超簡單，透過聊天機器人就可以完成。
玉山銀行搶先在金融業推出Chatbot，預計今年第二季正式上線
https://www.ithome.com.tw/news/112450
```

```
Early Stage Malware Prediction Using Recurrent Neural Networks
Matilda Rhode, Pete Burnap, Kevin Jones
(Submitted on 11 Aug 2017 (v1), last revised 18 Jun 2018 (this version, v3))
https://arxiv.org/abs/1708.03513
```

## C2.1_Sequence Data的分析
```
NLP自然語言處理(Natural Language Processing)與RNN
時間序列(Time series)分析與RNN
```
## C2.2_RNN基礎架構與Tensorflowg技術

## C2.3_RNN 實作RNN_lab
```
RNN_lab1:時間序列(Time serires)預測
RNN_lab2:NLP自然語言處理實測
```
## C2.4_RNN與資安應用
```
[資安應用]殭屍網路偵測

Tran D., Mac H., Tong V., Tran H.A. and Nguyen L.G., 
"A LSTM based framework for handling multiclass imbalance in DGA botnet detection." 
Neurocomputing, vol. 275, pp. 2401-2413, 2018
https://www.sciencedirect.com/science/article/pii/S0925231217317320
https://github.com/BKCS-HUST/LSTM-MI

Deep Neural Networks for Bot Detection
Sneha Kudugunta, Emilio Ferrara
(Submitted on 12 Feb 2018 (v1), last revised 18 Feb 2018 (this version, v2))
https://arxiv.org/abs/1802.04289


An Analysis of Recurrent Neural Networks for Botnet Detection Behavior
http://users.wpi.edu/~kmus/ECE579M_files/ReadingMaterials/RNN_Botnet_Detection[2093].pdf


Tracking Temporal Evolution of Network Activity for Botnet Detection
Kapil Sinha, Arun Viswanathan, Julian Bunn
(Submitted on 9 Aug 2019)
https://arxiv.org/abs/1908.03443
```

```
Most sophisticated bots use Domain Generation Algorithms (DGA) to pseudo-randomly 
generate a large number of domains 
and select a subset in order to communicate with Command and Control (C&C) server.

Predicting Domain Generation Algorithms with Long Short-Term Memory Networks
http://users.wpi.edu/~kmus/ECE579M_files/ReadingMaterials/LSTM_Domain_Prediction[2092].pdf
```
## 學員報告:進階實測
```

```
#### 學員研讀作業:
```
An AI-based, Multi-stage detection system of banking botnets
Li Ling, Zhiqiang Gao, Michael A Silas, Ian Lee, Erwan A Le Doeuff
(Submitted on 18 Jul 2019 (v1), last revised 25 Jul 2019 (this version, v3))
https://arxiv.org/abs/1907.08276
```
```
Your Smart Home Can't Keep a Secret: Towards Automated Fingerprinting of IoT Traffic with Neural Networks
Shuaike Dong, Zhou Li, Di Tang, Jiongyi Chen, Menghan Sun, Kehuan Zhang
(Submitted on 31 Aug 2019)
https://arxiv.org/abs/1909.00104
```

```
Omni SCADA Intrusion Detection Using Deep Learning Algorithms
Jun Gao, Luyun Gan, Fabiola Buschendorf, Liao Zhang, Hua Liu, Peixue Li, Xiaodai Dong, Tao Lu
(Submitted on 6 Aug 2019)
https://arxiv.org/abs/1908.01974
```

```
AppMine: Behavioral Analytics for Web Application Vulnerability Detection
Indranil Jana, Alina Oprea
(Submitted on 6 Aug 2019)
https://arxiv.org/abs/1908.01928
```

```
Dynamic Malware Analysis with Feature Engineering and Feature Learning
Zhaoqi Zhang, Panpan Qi, Wei Wang
(Submitted on 17 Jul 2019 (v1), last revised 21 Jul 2019 (this version, v2))
https://arxiv.org/abs/1907.07352
```
# C3.GAN 生成對抗網路 Generative Adversarial Network

### C3.1GAN基本觀念與應用
```
GAN
image-to-image translation與GAN
text-to-image與GAN
```
```
運動影象的去模糊化DeblurGAN(2017)
DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks
Orest Kupyn, Volodymyr Budzan, Mykola Mykhailych, Dmytro Mishkin, Jiri Matas
(Submitted on 19 Nov 2017 (v1), last revised 3 Apr 2018 (this version, v4))

https://zhuanlan.zhihu.com/p/32260634
https://www.itread01.com/content/1544765222.html
```
### C3.2_Tensorflow的GAN
```

TFGAN(2018.11)
Keras-GAN
   https://github.com/eriklindernoren/Keras-GAN
```

## C3.3_GAN 實作
```
GAN_lab1:DCGAN實作
GAN_lab2:TFGan實測
GAN_lab3:Keras-Gan實測
```

### C3.4_GAN與資安應用

```
PassGAN實作

PassGAN: A Deep Learning Approach for Password Guessing
Briland Hitaj, Paolo Gasti, Giuseppe Ateniese, Fernando Perez-Cruz
(Submitted on 1 Sep 2017 (v1), last revised 14 Feb 2019 (this version, v3))
https://arxiv.org/abs/1709.00440

https://github.com/brannondorsey/PassGAN
```
#### 學員報告:進階實測
```
CipherGAN(2018)
https://arxiv.org/abs/1801.04883
Unsupervised Cipher Cracking Using Discrete GANs
Aidan N. Gomez, Sicong Huang, Ivan Zhang, Bryan M. Li, Muhammad Osama, Lukasz Kaiser
(Submitted on 15 Jan 2018)
https://github.com/for-ai/CipherGAN

專業評論: https://openreview.net/forum?id=BkeqO7x0-


報導
https://zhuanlan.zhihu.com/p/33672256
https://www.pixpo.net/others/0IJw4qqN.html
```
```
Decoding Anagrammed Texts Written in an Unknown Language and Script
Bradley Hauer, Grzegorz Kondrak
https://transacl.org/ojs/index.php/tacl/article/view/821

報導
https://zhuanlan.zhihu.com/p/33672256
https://zhuanlan.zhihu.com/p/34063499
https://www.sciencealert.com/ai-may-have-finally-decoded-the-bizarre-mysterious-voynich-manuscript
https://gizmodo.com/artificial-intelligence-may-have-cracked-freaky-600-yea-1822519232?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=The%20Wild%20Week%20in%20AI
```
```
Zero-day malware detection using transferred generative adversarial networks based on deep autoencoders
Author links open overlay panelJin-YoungKimSeok-JunBuSung-BaeCho
https://www.sciencedirect.com/science/article/pii/S0020025518303475
```
# D.最新發展:AI2018-2019, AI_Sec2018-2019

## D1.人工智慧的最新發展AI2018-2019

## D2.人工智慧在資安領域的最新發展AI_Sec2018-2019
```
1.AI強化的攻擊技術
2.AI強化的防禦技術
3.攻擊人工智慧系統
```
# 學員成果報告

### AI for Industrial Internet of Things (IIoT) 

```
A Pvalue-guided Anomaly Detection Approach Combining Multiple Heterogeneous Log Parser Algorithms on IIoT Systems
Xueshuo Xie, Zhi Wang, Xuhang Xiao, Lei Yang, Shenwei Huang, Tao Li
(Submitted on 5 Jul 2019)
https://arxiv.org/abs/1907.02765
```

### blockchain security
```
SmartEmbed(2019)
https://arxiv.org/abs/1908.08615
SmartEmbed: A Tool for Clone and Bug Detection in Smart Contracts through Structural Code Embedding
Zhipeng Gao, Vinoj Jayasundara, Lingxiao Jiang, Xin Xia, David Lo, John Grundy
(Submitted on 22 Aug 2019)

Blockchain based access control systems: State of the art and challenges
Sara Rouhani, Ralph Deters
(Submitted on 22 Aug 2019)
https://arxiv.org/abs/1908.08503
```


### Phishing Web Page Detection 
```
HTMLPhish: Enabling Accurate Phishing Web Page Detection by Applying Deep Learning Techniques on HTML Analysis
Chidimma Opara, Bo Wei, Yingke Chen
(Submitted on 28 Aug 2019)
https://arxiv.org/abs/1909.01135
```
### APT偵測與預防
```
Kidemonas: The Silent Guardian
Rudra Prasad Baksi, Shambhu J. Upadhyaya
(Submitted on 3 Dec 2017)
https://arxiv.org/abs/1712.00841

Bidirectional RNN-based Few-shot Training for Detecting Multi-stage Attack
Di Zhao, Jiqiang Liu, Jialin Wang, Wenjia Niu, Endong Tong, Tong Chen, Gang Li
(Submitted on 9 May 2019)
https://arxiv.org/abs/1905.03454

"Feint Attack", as a new type of APT attack, has become the focus of attention. 
It adopts a multi-stage attacks mode which can be concluded as a combination of virtual attacks and real attacks. 


[碩士論文]Detection and Prevention of Advanced Persistent Threats
http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/7057/pdf/imm7057.pdf

[碩士論文]Detection of APT Malware through External and Internal Network Traffic Correlation
Author: Terence Slot(2015)
https://pdfs.semanticscholar.org/0590/c5e44476418dd138fbd495a1cd933f9877c4.pdf

[博士論文]NETWORK-BASED TARGETED ATTACKS DETECTION(2019)IBRAHIM GHAFIR
https://is.muni.cz/th/iq0qi/Plny_text_prace.pdf

[博士論文]Real-time detection of Advanced Persistent Threats using Information Flow Tracking and Hidden Markov Models
Guillaume Brogi(2018)
https://tel.archives-ouvertes.fr/tel-01793709/document
```
```
[碩士論文]Towards a roadmap for development of intelligent data analysis based cyber attack detection systems
https://pdfs.semanticscholar.org/ba82/76ed46a9a4e44582a1e96cc6bba9bd51b842.pdf
```

APT attack software Poison Ivy


###
```
Report on the First Knowledge Graph Reasoning Challenge 2018 -- Toward the eXplainable AI System
Takahiro Kawamura, Shusaku Egami, Koutarou Tamura, Yasunori Hokazono, Takanori Ugai, Yusuke Koyanagi, Fumihito Nishino, Seiji Okajima, Katsuhiko Murakami, Kunihiko Takamatsu, Aoi Sugiura, Shun Shiramatsu, Shawn Zhang, Kouji Kozaki
(Submitted on 22 Aug 2019)
https://arxiv.org/abs/1908.08184
```
