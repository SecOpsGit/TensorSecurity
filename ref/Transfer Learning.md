# 遷移學習 Transfer Learning  

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE) [![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)


Everything about Transfer Learning (Probably the **most complete** repository?). *Your contribution is highly valued!* If you find this repo helpful, please cite it as follows:

關於遷移學習的所有資料，包括：介紹、綜述文章、最新文章、代表工作及其代碼、常用資料集、碩博士論文、比賽等等。(可能是**目前最全**的遷移學習資料庫？) *歡迎一起貢獻！* 如果認為本倉庫有用，請在你的論文和其他出版物中進行引用！ 

```
@Misc{transferlearning.xyz,
howpublished = {\url{http://transferlearning.xyz}},   
title = {Everything about Transfer Learning and Domain Adapation},  
author = {Wang, Jindong and others}  
}  
```

- [遷移學習 Transfer Learning](#遷移學習-transfer-learning)
	- [0.Latest Publications (最新論文)](#0latest-publications-最新論文)
	- [1.Introduction and Tutorials (簡介與教程)](#1introduction-and-tutorials-簡介與教程)
	- [2.Transfer Learning Areas and Papers (研究領域與相關論文)](#2transfer-learning-areas-and-papers-研究領域與相關論文)
	- [3.Theory and Survey (理論與綜述)](#3theory-and-survey-理論與綜述)
	- [4.Code (代碼)](#4code-代碼)
	- [5.Transfer Learning Scholars (著名學者)](#5transfer-learning-scholars-著名學者)
	- [6.Transfer Learning Thesis (碩博士論文)](#6transfer-learning-thesis-碩博士論文)
	- [7.Datasets and Benchmarks (資料集與評測結果)](#7datasets-and-benchmarks-資料集與評測結果)
	- [8.Transfer Learning Challenges (遷移學習比賽)](#8transfer-learning-challenges-遷移學習比賽)
	- [Applications (遷移學習應用)](#applications-遷移學習應用)
	- [Other Resources (其他資源)](#other-resources-其他資源)
	- [Contributing (歡迎參與貢獻)](#contributing-歡迎參與貢獻)


> 關於機器學習和行為識別的資料，請參考：[行為識別](https://github.com/jindongwang/activityrecognition)｜[機器學習](https://github.com/jindongwang/MachineLearning)

- - -

## 0.Latest Publications (最新論文)

**A good website to see the latest arXiv preprints by search: [Transfer learning](http://arxitics.com/search?q=transfer%20learning&sort=updated#1904.01376/abstract), [Domain adaptation](http://arxitics.com/search?q=domain%20adaptation&sort=updated)**

**一個很好的網站，可以直接看到最新的arXiv文章: [Transfer learning](http://arxitics.com/search?q=transfer%20learning&sort=updated#1904.01376/abstract), [Domain adaptation](http://arxitics.com/search?q=domain%20adaptation&sort=updated)**

[遷移學習文章匯總 Awesome transfer learning papers](https://github.com/jindongwang/transferlearning/tree/master/doc/awesome_paper.md)

- **Latest publications**

	- 20190910 BMVC-19 [Curriculum based Dropout Discriminator for Domain Adaptation](https://arxiv.org/abs/1907.10628)
    	- Curriculum dropout for domain adaptation
    	- 基於課程學習的dropout用於DA

	- 20190909 IJCAI-FML-19 [FedHealth: A Federated Transfer Learning Framework for Wearable Healthcare](http://jd92.wang/assets/files/a15_ijcai19.pdf)
    	- The first work on federated transfer learning for wearable healthcare
    	- 第一個將聯邦遷移學習用於可穿戴健康監護的工作

	- 20190909 PAMI [Inferring Latent Domains for Unsupervised Deep Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/8792192)
    	- Inferring latent domains for unsupervised deep domain
    	- 在深度遷移學習中推斷隱含領域

	- 20190902 AAAI-19 [Aligning Domain-Specific Distribution and Classifier for Cross-Domain Classification from Multiple Sources](https://www.aaai.org/ojs/index.php/AAAI/article/download/4551/4429)
    	- Multi-source domain adaptation using both features and classifier adaptation
    	- 利用特徵和分類器同時適配進行多源遷移

	- 20190829 EMNLP-19 [Investigating Meta-Learning Algorithms for Low-Resource Natural Language Understanding Tasks](https://arxiv.org/abs/1908.10423)
    	- Investigating MAML for low-resource NMT
    	- 調查了MAML方法用於低資源的NMT問題的表現


- **Preprints on arXiv** (Not peer-reviewed)

	- 20190829 arXiv [A survey of cross-lingual features for zero-shot cross-lingual semantic parsing](https://arxiv.org/abs/1908.10461)
    	- 一個跨語言學習綜述

	- 20190828 arXiv [VAE-based Domain Adaptation for Speaker Verification](https://arxiv.org/abs/1908.10092)
    	- Speaker verification using VAE domain adaptation
    	- 基於VAE的speaker verification

	- 20190821 arXiv [Transfer Learning-Based Label Proportions Method with Data of Uncertainty](https://arxiv.org/abs/1908.06603)
    	- Transfer learning with source and target having uncertainty
    	- 當source和target都有不確定label時進行遷移

	- 20190821 arXiv [Transfer in Deep Reinforcement Learning using Knowledge Graphs](https://arxiv.org/abs/1908.06556)
    	- Use knowledge graph to transfer in reinforcement learning
    	- 用知識圖譜進行強化遷移

	- 20190821 arXiv [Shallow Domain Adaptive Embeddings for Sentiment Analysis](https://arxiv.org/abs/1908.06082)
    	- Domain adaptative embedding for sentiment analysis
    	- 遷移學習用於情感分類


[**更多 More...**](https://github.com/jindongwang/transferlearning/tree/master/doc/awesome_paper.md)

- - -

## 1.Introduction and Tutorials (簡介與教程)

- 簡介文字資料
	- [簡單的中文簡介 Chinese introduction](https://github.com/jindongwang/transferlearning/blob/master/doc/%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E7%AE%80%E4%BB%8B.md)
	- [PPT(English)](http://jd92.wang/assets/files/l03_transferlearning.pdf)
	- [PPT(中文)](http://jd92.wang/assets/files/l08_tl_zh.pdf)
	- 遷移學習中的領域自我調整方法 Domain adaptation: [PDF](http://jd92.wang/assets/files/l12_da.pdf) ｜ [Video](http://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247484940&idx=2&sn=35e64e07fde9a96afbb65dbf40a945eb&chksm=ec1febf5db6862e38d5e02ff3278c61b376932a46c5628c7d9cb1769c572bfd31819c13dd468&mpshare=1&scene=1&srcid=1219JpTNZFiNDCHsTUrUxwqy#rd)
	- 清華大學龍明盛老師的深度遷移學習報告 Transfer learning report by Mingsheng Long @ THU：[PPT(Samsung)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/transfer-learning-talk.pdf)、[PPT(Google China)](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-transfer-learning-talk.pdf)

- 入門教程
	- [**《遷移學習簡明手冊》Transfer Learning Tutorial**](https://zhuanlan.zhihu.com/p/35352154) [開發維護地址](https://github.com/jindongwang/transferlearning-tutorial)

- 視頻教程
	- [臺灣大學李巨集毅的視頻講解(中文視頻)](https://www.youtube.com/watch?v=qD6iD4TFsdQ)
	- [遷移學習中的領域自我調整方法(中文)](http://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247484940&idx=2&sn=35e64e07fde9a96afbb65dbf40a945eb&chksm=ec1febf5db6862e38d5e02ff3278c61b376932a46c5628c7d9cb1769c572bfd31819c13dd468&mpshare=1&scene=1&srcid=1219JpTNZFiNDCHsTUrUxwqy#rd)

- [遷移學習領域的著名學者、代表工作及實驗室介紹 Transfer Learning Scholars and Labs](https://github.com/jindongwang/transferlearning/blob/master/doc/scholar_TL.md)

- 什麼是[負遷移(negative transfer)](https://www.zhihu.com/question/66492194/answer/242870418)？

- 動手教程、代碼、資料 Hands-on Codes
	- [基於深度學習和遷移學習的識花實踐 Using Transfer Learning for Flower Recognition](https://cosx.org/2017/10/transfer-learning/)
	- [基於Pytorch的圖像分類 Using Transfer Learning for Image Classification](https://github.com/miguelgfierro/sciblog_support/blob/master/A_Gentle_Introduction_to_Transfer_Learning/Intro_Transfer_Learning.ipynb)
	- [使用Pytorch進行finetune Using Pytorch for Fine-tune](https://github.com/Spandan-Madan/Pytorch_fine_tuning_Tutorial)
	- [基於AlexNet和ResNet的finetune Fine-tune based on Alexnet and Resnet](https://github.com/jindongwang/transferlearning/tree/master/code/AlexNet_ResNet)
	- [更多 More...](https://github.com/jindongwang/transferlearning/tree/master/code)

- - -

## 2.Transfer Learning Areas and Papers (研究領域與相關論文)

Related articles by research areas:

- [General Transfer Learning (普通遷移學習)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#general-transfer-learning-%E6%99%AE%E9%80%9A%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
  - [Theory (理論)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#theory-%E7%90%86%E8%AE%BA)
  - [Others (其他)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#others-%E5%85%B6%E4%BB%96)
- [Domain Adaptation (領域自我調整)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#domain-adaptation-%E9%A2%86%E5%9F%9F%E8%87%AA%E9%80%82%E5%BA%94)
  - [Traditional Methods (傳統遷移方法)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#traditional-methods-%E4%BC%A0%E7%BB%9F%E8%BF%81%E7%A7%BB%E6%96%B9%E6%B3%95)
  - [Deep / Adversarial Methods (深度/對抗遷移方法)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#deep--adversarial-methods-%E6%B7%B1%E5%BA%A6%E5%AF%B9%E6%8A%97%E8%BF%81%E7%A7%BB%E6%96%B9%E6%B3%95)
- [Domain Generalization](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#domain-generalization)
- [Multi-source Transfer Learning (多源遷移學習)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#multi-source-transfer-learning-%E5%A4%9A%E6%BA%90%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
- [Heterogeneous Transfer Learning (異構遷移學習)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#heterogeneous-transfer-learning-%E5%BC%82%E6%9E%84%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
- [Online Transfer Learning (線上遷移學習)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#online-transfer-learning-%E5%9C%A8%E7%BA%BF%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
- [Zero-shot / Few-shot Learning](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#zero-shot--few-shot-learning)
- [Deep Transfer Learning (深度遷移學習)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#deep-transfer-learning-%E6%B7%B1%E5%BA%A6%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
  - [Non-Adversarial Transfer Learning (非對抗深度遷移)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#non-adversarial-transfer-learning-%E9%9D%9E%E5%AF%B9%E6%8A%97%E6%B7%B1%E5%BA%A6%E8%BF%81%E7%A7%BB)
  - [Deep Adversarial Transfer Learning (對抗遷移學習)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#deep-adversarial-transfer-learning-%E5%AF%B9%E6%8A%97%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
- [Multi-task Learning (多工學習)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#multi-task-learning-%E5%A4%9A%E4%BB%BB%E5%8A%A1%E5%AD%A6%E4%B9%A0)
- [Transfer Reinforcement Learning (強化遷移學習)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#transfer-reinforcement-learning-%E5%BC%BA%E5%8C%96%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
- [Transfer Metric Learning (遷移度量學習)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#transfer-metric-learning-%E8%BF%81%E7%A7%BB%E5%BA%A6%E9%87%8F%E5%AD%A6%E4%B9%A0)
- [Transitive Transfer Learning (傳遞遷移學習)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#transitive-transfer-learning-%E4%BC%A0%E9%80%92%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
- [Lifelong Learning (終身遷移學習)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#lifelong-learning-%E7%BB%88%E8%BA%AB%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0)
- [Negative Transfer (負遷移)](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#negative-transfer-%E8%B4%9F%E8%BF%81%E7%A7%BB)
- [Transfer Learning Applications (應用)](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md)

[Paperweekly](http://www.paperweekly.site/collections/231/papers): 一個推薦、分享論文的網站比較好，上面會持續整理相關的文章並分享閱讀筆記。

- - -

## 3.Theory and Survey (理論與綜述)

Here are some articles on transfer learning theory and survey.

- 遷移學習領域最具代表性的綜述是[A survey on transfer learning](http://ieeexplore.ieee.org/abstract/document/5288526/)，發表於2010年，對遷移學習進行了比較權威的定義。 -- The most influential survey on transfer learning.

- 遷移學習的**理論分析** Transfer Learning Theory：

	- 遷移學習方面一直以來都比較缺乏理論分析與證明的文章，以下三篇連貫式的理論文章成為了經典 Transfer learning theory：
		- NIPS-06 [Analysis of Representations for Domain Adaptation](https://dl.acm.org/citation.cfm?id=2976474)
		- ML-10 [A Theory of Learning from Different Domains](https://link.springer.com/article/10.1007/s10994-009-5152-4)
		- NIPS-08 [Learning Bounds for Domain Adaptation](http://papers.nips.cc/paper/3212-learning-bounds-for-domain-adaptation)

	- 許多研究者在遷移學習的研究中會應用MMD(Maximum Mean Discrepancy)這個最大均值差異來衡量不同domain之間的距離。MMD的理論文章是：
		- MMD的提出：[A Hilbert Space Embedding for Distributions](https://link.springer.com/chapter/10.1007/978-3-540-75225-7_5) 以及 [A Kernel Two-Sample Test](http://www.jmlr.org/papers/v13/gretton12a.html)
		- 多核MMD(MK-MMD)：[Optimal kernel choice for large-scale two-sample tests](http://papers.nips.cc/paper/4727-optimal-kernel-choice-for-large-scale-two-sample-tests)
		- MMD及多核MMD代碼：[Matlab](https://github.com/lopezpaz/classifier_tests/tree/master/code/unit_test_mmd) | [Python](https://github.com/jindongwang/transferlearning/tree/master/code/basic/mmd.py)
	- 理論研究方面，重點關注Alex Smola、Ben-David、Bernhard Schölkopf、Arthur Gretton等人的研究即可。

- 較新的綜述 Latest survey：

    - 用transfer learning進行sentiment classification的綜述：[A Survey of Sentiment Analysis Based on Transfer Learning](https://ieeexplore.ieee.org/abstract/document/8746210) 
	- 2019 一篇新survey：[Transfer Adaptation Learning: A Decade Survey](https://arxiv.org/abs/1903.04687)
	- 2018 一篇遷移度量學習的綜述: [Transfer Metric Learning: Algorithms, Applications and Outlooks](https://arxiv.org/abs/1810.03944)
	- 2018 一篇最近的非對稱情況下的異構遷移學習綜述：[Asymmetric Heterogeneous Transfer Learning: A Survey](https://arxiv.org/abs/1804.10834)
	- 2018 Neural style transfer的一個survey：[Neural Style Transfer: A Review](https://arxiv.org/abs/1705.04058)
	- 2018 深度domain adaptation的一個綜述：[Deep Visual Domain Adaptation: A Survey](https://www.sciencedirect.com/science/article/pii/S0925231218306684)
	- 2017 多工學習的綜述，來自香港科技大學楊強團隊：[A survey on multi-task learning](https://arxiv.org/abs/1707.08114)
	- 2017 異構遷移學習的綜述：[A survey on heterogeneous transfer learning](https://link.springer.com/article/10.1186/s40537-017-0089-0)
	- 2017 跨領域資料識別的綜述：[Cross-dataset recognition: a survey](https://arxiv.org/abs/1705.04396)
	- 2016 [A survey of transfer learning](https://pan.baidu.com/s/1gfgXLXT)。其中交代了一些比較經典的如同構、異構等學習方法代表性文章。
	- 2015 中文綜述：[遷移學習研究進展](https://pan.baidu.com/s/1bpautob)

- 遷移學習的應用
	- 視覺domain adaptation綜述：[Visual Domain Adaptation: A Survey of Recent Advances](https://pan.baidu.com/s/1o8BR7Vc)
	- 遷移學習應用于行為識別綜述：[Transfer Learning for Activity Recognition: A Survey](https://pan.baidu.com/s/1kVABOYr)
	- 遷移學習與增強學習：[Transfer Learning for Reinforcement Learning Domains: A Survey](https://pan.baidu.com/s/1slfr0w1)
	- 多個源域進行遷移的綜述：[A Survey of Multi-source Domain Adaptation](https://pan.baidu.com/s/1eSGREF4)。

_ _ _

## 4.Code (代碼)

請見[這裡](https://github.com/jindongwang/transferlearning/tree/master/code) | Please see [HERE](https://github.com/jindongwang/transferlearning/tree/master/code) for some popular transfer learning codes.

_ _ _

## 5.Transfer Learning Scholars (著名學者)

Here are some transfer learning scholars and labs.

**全部清單以及代表工作性見[這裡](https://github.com/jindongwang/transferlearning/blob/master/doc/scholar_TL.md)**

Please refer to [here](https://github.com/jindongwang/transferlearning/blob/master/doc/scholar_TL.md) to see a complete list.

- [Qiang Yang](http://www.cs.ust.hk/~qyang/)：中文名楊強。香港科技大學電腦系講座教授，遷移學習領域世界性專家。IEEE/ACM/AAAI/IAPR/AAAS fellow。[[Google scholar](https://scholar.google.com/citations?user=1LxWZLQAAAAJ&hl=zh-CN)]

- [Sinno Jialin Pan](http://www.ntu.edu.sg/home/sinnopan/)：楊強的學生，香港科技大學博士，現任新加坡南洋理工大學助理教授。遷移學習領域代表性綜述A survey on transfer learning的第一作者（Qiang Yang是二作）。[[Google scholar](https://scholar.google.com/citations?user=P6WcnfkAAAAJ&hl=zh-CN)]

- [Wenyuan Dai](https://scholar.google.com.sg/citations?user=AGR9pP0AAAAJ&hl=zh-CN)：中文名戴文淵，上海交通大學碩士，現任第四范式人工智慧創業公司CEO。遷移學習領域著名的牛人，在頂級會議上發表多篇高水準文章，每篇論文引用量巨大。[[Google scholar](https://scholar.google.com.hk/citations?hl=zh-CN&user=AGR9pP0AAAAJ)]

- [Lixin Duan](http://www.lxduan.info/)：中文名段立新，新加坡南洋理工大學博士，現就職于電子科技大學，教授。[[Google scholar](https://scholar.google.com.hk/citations?user=inRIcS0AAAAJ&hl=zh-CN&oi=ao)]

- [Boqing Gong](http://boqinggong.info/index.html)：南加州大學博士，現就職於騰訊AI Lab(西雅圖)。曾任中佛羅里達大學助理教授。[[Google scholar](https://scholar.google.com/citations?user=lv9ZeVUAAAAJ&hl=en)]

- [Fuzhen Zhuang](http://www.intsci.ac.cn/users/zhuangfuzhen/)：中文名莊福振，中科院計算所博士，現任中科院計算所副研究員。[[Google scholar](https://scholar.google.com/citations?user=klJBYrAAAAAJ&hl=zh-CN&oi=ao)]

- [Mingsheng Long](http://ise.thss.tsinghua.edu.cn/~mlong/)：中文名龍明盛，清華大學博士，現任清華大學助理教授、博士生導師。[[Google scholar](https://scholar.google.com/citations?view_op=search_authors&mauthors=mingsheng+long&hl=zh-CN&oi=ao)]

- [Qingyao Wu](https://sites.google.com/site/qysite/)：中文名吳慶耀，現任華南理工大學副教授。主要做線上遷移學習、異構遷移學習方面的研究。[[Google scholar](https://scholar.google.com.hk/citations?user=n6e_2IgAAAAJ&hl=zh-CN&oi=ao)]

- [Weike Pan](https://sites.google.com/site/weikep/)：中文名潘微科，楊強的學生，現任深圳大學副教授，香港科技大學博士畢業。主要做遷移學習在推薦系統方面的一些工作。 [[Google Scholar](https://scholar.google.com/citations?user=pC5Q26MAAAAJ&hl=en)]

- [Tongliang Liu](http://ieeexplore.ieee.org/abstract/document/8259375/)：中文名劉同亮，現任悉尼大學助理教授。主要做遷移學習的一些理論方面的工作。[[Google scholar](https://scholar.google.com.hk/citations?hl=zh-CN&user=EiLdZ_YAAAAJ)]

- [Tatiana Tommasi](http://tatianatommasi.wixsite.com/tatianatommasi/3)：Researcher at the Italian Institute of Technology.

- [Vinod K Kurmi](https://github.com/vinodkkurmi)[[home page](https://github.com/vinodkkurmi)]: Researcher at the Indian Institute of Technology Kanpur(India)
_ _ _

## 6.Transfer Learning Thesis (碩博士論文)

Here are some popular thesis on transfer learning.

碩博士論文可以讓我們很快地對遷移學習的相關領域做一些瞭解，同時，也能很快地瞭解概括相關研究者的工作。其中，比較有名的有

- 2016 Baochen Sun的[Correlation Alignment for Domain Adaptation](http://www.cs.uml.edu/~bsun/papers/baochen_phd_thesis.pdf)

- 2015 南加州大學的Boqing Gong的[Kernel Methods for Unsupervised Domain Adaptation](https://pan.baidu.com/s/1bpbawv9)

- 2014 清華大學龍明盛的[遷移學習問題與方法研究](http://ise.thss.tsinghua.edu.cn/~mlong/doc/phd-thesis-mingsheng-long.pdf)

- 2014 中科院計算所趙中堂的[自我調整行為識別中的遷移學習方法研究](https://pan.baidu.com/s/1kVqYXnh)

- 2012 楊強的學生Hao Hu的[Learning based Activity Recognition](https://pan.baidu.com/s/1bp2K9HX)

- 2012 楊強的學生Wencheng Zheng的[Learning with Limited Data in Sensor-based Human Behavior Prediction](https://pan.baidu.com/s/1o8MbbBk)

- 2010 楊強的學生Sinno Jialin Pan的[Feature-based Transfer Learning and Its Applications](https://pan.baidu.com/s/1bUqMfW)

- 2009 上海交通大學戴文淵的[基於實例和特徵的遷移學習演算法研究](https://pan.baidu.com/s/1i4Vyygd)

其他的文章，請見[完整版](https://pan.baidu.com/s/1bqXEASn)。

- - -

## 7.Datasets and Benchmarks (資料集與評測結果)

Please see [HERE](https://github.com/jindongwang/transferlearning/blob/master/data) for the popular transfer learning **datasets and certain benchmark** results.

[這裡](https://github.com/jindongwang/transferlearning/blob/master/data)整理了常用的公開資料集和一些已發表的文章在這些資料集上的實驗結果。

- - -

## 8.Transfer Learning Challenges (遷移學習比賽)

一些關於遷移學習的國際比賽。

- [Visual Domain Adaptation Challenge (VisDA)](http://ai.bu.edu/visda-2018/)

- - -

## Applications (遷移學習應用)

See [HERE](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md) for transfer learning applications.

遷移學習應用請見[這裡](https://github.com/jindongwang/transferlearning/blob/master/doc/transfer_learning_application.md)。

- - -
  
## Other Resources (其他資源)

Call for papers about transfer learning:

- [Transfer Learning for Multimedia Applications(A Special Issue on Multimedia Tools and Applications (MTAP))](https://lijin118.github.io/mtap/)

Related projects:

- Salad: [A semi-supervised domain adaptation library](https://domainadaptation.org)


- - -
