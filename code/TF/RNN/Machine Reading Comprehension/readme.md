# 機器閱讀理解
```
機器閱讀理解(Machine Reading Comprehension)是指讓機器閱讀文本，然後回答和閱讀內容相關的問題。
閱讀理解是自然語言處理和人工智慧領域的重要前沿課題，對於提升機器智慧水準、使機器具有持續知識獲取能力具有重要價值，
近年來受到學術界和工業界的廣泛關注。
```

## 清華 NLP 團隊推薦：必讀的77篇機器閱讀理解論文
```
2018-11-06 08:15:51

https://github.com/thunlp/RCPapers
```

### 【模型結構篇】
```
Memory networks. Jason Weston, Sumit Chopra, and Antoine Bordes. arXiv preprint arXiv:1410.3916 (2014).

Teaching Machines to Read and Comprehend. Hermann, Karl Moritz, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. NIPS 2015.

Text Understanding with the Attention Sum Reader Network. Rudolf Kadlec, Martin Schmid, Ondrej Bajgar, and Jan Kleindienst. ACL 2016.

A Thorough Examination of the Cnn/Daily Mail Reading Comprehension Task. Danqi Chen, Jason Bolton, and Christopher D. Manning. ACL 2016.

Long Short-Term Memory-Networks for Machine Reading. Jianpeng Cheng, Li Dong, and Mirella Lapata. EMNLP 2016.

Key-value Memory Networks for Directly Reading Documents. Alexander Miller, Adam Fisch, Jesse Dodge, Amir-Hossein Karimi, Antoine Bordes, and Jason Weston. EMNLP 2016.

Modeling Human Reading with Neural Attention. Michael Hahn and Frank Keller. EMNLP 2016.

Learning Recurrent Span Representations for Extractive Question Answering Kenton Lee, Shimi Salant, Tom Kwiatkowski, Ankur Parikh, Dipanjan Das, and Jonathan Berant. arXiv preprint arXiv:1611.01436 (2016).

Multi-Perspective Context Matching for Machine Comprehension. Zhiguo Wang, Haitao Mi, Wael Hamza, and Radu Florian. arXiv preprint arXiv:1612.04211.

Natural Language Comprehension with the Epireader. Adam Trischler, Zheng Ye, Xingdi Yuan, and Kaheer Suleman. EMNLP 2016.

Iterative Alternating Neural Attention for Machine Reading. Alessandro Sordoni, Philip Bachman, Adam Trischler, and Yoshua Bengio. arXiv preprint arXiv:1606.02245 (2016).

Bidirectional Attention Flow for Machine Comprehension. Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, and Hannaneh Hajishirzi. ICLR 2017.

Machine Comprehension Using Match-lstm and Answer Pointer. Shuohang Wang and Jing Jiang. arXiv preprint arXiv:1608.07905 (2016).

Gated Self-matching Networks for Reading Comprehension and Question Answering. Wenhui Wang, Nan Yang, Furu Wei, Baobao Chang, and Ming Zhou. ACL 2017.

Attention-over-attention Neural Networks for Reading Comprehension. Yiming Cui, Zhipeng Chen, Si Wei, Shijin Wang, Ting Liu, and Guoping Hu. ACL 2017.

Gated-attention Readers for Text Comprehension. Bhuwan Dhingra, Hanxiao Liu, Zhilin Yang, William W. Cohen, and Ruslan Salakhutdinov. ACL 2017.

A Constituent-Centric Neural Architecture for Reading Comprehension. Pengtao Xie and Eric Xing. ACL 2017.

Structural Embedding of Syntactic Trees for Machine Comprehension. Rui Liu, Junjie Hu, Wei Wei, Zi Yang, and Eric Nyberg. EMNLP 2017.

Accurate Supervised and Semi-Supervised Machine Reading for Long Documents. Izzeddin Gur, Daniel Hewlett, Alexandre Lacoste, and Llion Jones. EMNLP 2017.

MEMEN: Multi-layer Embedding with Memory Networks for Machine Comprehension. Boyuan Pan, Hao Li, Zhou Zhao, Bin Cao, Deng Cai, and Xiaofei He. arXiv preprint arXiv:1707.09098 (2017).

Dynamic Coattention Networks For Question Answering. Caiming Xiong, Victor Zhong, and Richard Socher. ICLR 2017

R-NET: Machine Reading Comprehension with Self-matching Networks. Natural Language Computing Group, Microsoft Research Asia.

Reasonet: Learning to Stop Reading in Machine Comprehension. Yelong Shen, Po-Sen Huang, Jianfeng Gao, and Weizhu Chen. KDD 2017.

FusionNet: Fusing via Fully-Aware Attention with Application to Machine Comprehension. Hsin-Yuan Huang, Chenguang Zhu, Yelong Shen, and Weizhu Chen. ICLR 2018.

Making Neural QA as Simple as Possible but not Simpler. Dirk Weissenborn, Georg Wiese, and Laura Seiffe. CoNLL 2017.

Efficient and Robust Question Answering from Minimal Context over Documents. Sewon Min, Victor Zhong, Richard Socher, and Caiming Xiong. ACL 2018.

Simple and Effective Multi-Paragraph Reading Comprehension. Christopher Clark and Matt Gardner. ACL 2018.

Neural Speed Reading via Skim-RNN. Minjoon Seo, Sewon Min, Ali Farhadi, and Hannaneh Hajishirzi. ICLR2018.

Hierarchical Attention Flow forMultiple-Choice Reading Comprehension. Haichao Zhu,� Furu Wei, Bing Qin, and Ting Liu. AAAI 2018.

Towards Reading Comprehension for Long Documents. 
Yuanxing Zhang, Yangbin Zhang, Kaigui Bian, and Xiaoming Li. IJCAI 2018.

Joint Training of Candidate Extraction and Answer Selection for Reading Comprehension. 
Zhen Wang, Jiachen Liu, Xinyan Xiao, Yajuan Lyu, and Tian Wu. ACL 2018.

Multi-Passage Machine Reading Comprehension with Cross-Passage Answer Verification. 
Yizhong Wang, Kai Liu, Jing Liu, Wei He, Yajuan Lyu, Hua Wu, Sujian Li, and Haifeng Wang. ACL 2018.

Reinforced Mnemonic Reader for Machine Reading Comprehension. 
Minghao Hu, Yuxing Peng, Zhen Huang, Xipeng Qiu, Furu Wei, and Ming Zhou. IJCAI 2018.

Stochastic Answer Networks for Machine Reading Comprehension. 
Xiaodong Liu, Yelong Shen, Kevin Duh, and Jianfeng Gao. ACL 2018.

Multi-Granularity Hierarchical Attention Fusion Networks for Reading Comprehension and Question Answering. 
Wei Wang, Ming Yan, and Chen Wu. ACL 2018.

A Multi-Stage Memory Augmented Neural Networkfor Machine Reading Comprehension. 
Seunghak Yu, Sathish Indurthi, Seohyun Back, and Haejun Lee. ACL 2018 workshop.

S-NET: From Answer Extraction to Answer Generation for Machine Reading Comprehension. 
Chuanqi Tan, Furu Wei, Nan Yang, Bowen Du, Weifeng Lv, and Ming Zhou. AAAI2018.

Ask the Right Questions: Active Question Reformulation with Reinforcement Learning. 
Christian Buck, Jannis Bulian, Massimiliano Ciaramita, Wojciech Gajewski, Andrea Gesmundo, Neil Houlsby, and Wei Wang. ICLR2018.

QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension. 
Adams Wei Yu, David Dohan, Minh-Thang Luong, Rui Zhao, Kai Chen, Mohammad Norouzi, and Quoc V. Le. ICLR2018.

Read + Verify: Machine Reading Comprehension with Unanswerable Questions. 
Minghao Hu, Furu Wei, Yuxing Peng, Zhen Huang, Nan Yang, and Ming Zhou. NAACL2018.
```
 
### 【利用額外知識篇】
```
Leveraging Knowledge Bases in LSTMs for Improving Machine Reading. 
Bishan Yang and Tom Mitchell. ACL 2017.

Learned in Translation: Contextualized Word Vectors. 
Bryan McCann, James Bradbury, Caiming Xiong, and Richard Socher. 
arXiv preprint arXiv:1708.00107 (2017).

Knowledgeable Reader: Enhancing Cloze-Style Reading Comprehension with External Commonsense Knowledge. 
Todor Mihaylov and Anette Frank. ACL 2018.

A Comparative Study of Word Embeddings for Reading Comprehension. 
Bhuwan Dhingra, Hanxiao Liu, Ruslan Salakhutdinov, and William W. Cohen. 
arXiv preprint arXiv:1703.00993 (2017).

Deep contextualized word representations. 
Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 
NAACL 2018.

Improving Language Understanding by Generative Pre-Training.
Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 
OpenAI.

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. 
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 
arXiv preprint arXiv:1810.04805 (2018).
```

###【探索篇】
```
Adversarial Examples for Evaluating Reading Comprehension Systems. 
Robin Jia, and Percy Liang. EMNLP 2017.

Did the Model Understand the Question? 
Pramod Kaushik Mudrakarta, Ankur Taly, Mukund Sundararajan, and Kedar Dhamdhere. ACL 2018.
```
 

###【開放域問答篇】
```
Reading Wikipedia to Answer Open-Domain Questions. Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. ACL 2017.

R^3: Reinforced Reader-Ranker for Open-Domain Question Answering. Shuohang Wang, Mo Yu, Xiaoxiao Guo, Zhiguo Wang, Tim Klinger, Wei Zhang, Shiyu Chang, Gerald Tesauro, Bowen Zhou, and Jing Jiang. AAAI 2018.

Evidence Aggregation for Answer Re-Ranking in Open-Domain Question Answering. Shuohang Wang, Mo Yu, Jing Jiang, Wei Zhang, Xiaoxiao Guo, Shiyu Chang, Zhiguo Wang, Tim Klinger, Gerald Tesauro, and Murray Campbell. ICLR 2018.

Denoising Distantly Supervised Open-Domain Question Answering. Yankai Lin, Haozhe Ji, Zhiyuan Liu, and Maosong Sun. ACL 2018.
```
### 【資料集篇】
```
(SQuAD 1.0) SQuAD: 100,000+ Questions for Machine Comprehension of Text. Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. EMNLP 2016.

(SQuAD 2.0) Know What You Don't Know: Unanswerable Questions for SQuAD. Pranav Rajpurkar, Robin Jia, and Percy Liang. ACL 2018.

(MS MARCO) MS MARCO: A Human Generated MAchine Reading COmprehension Dataset. Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. arXiv preprint arXiv:1611.09268 (2016).

(Quasar) Quasar: Datasets for Question Answering by Search and Reading. Bhuwan Dhingra, Kathryn Mazaitis, and William W. Cohen. arXiv preprint arXiv:1707.03904 (2017).

(TriviaQA) TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. Mandar Joshi, Eunsol Choi, Daniel S. Weld, Luke Zettlemoyer. arXiv preprint arXiv:1705.03551 (2017).

(SearchQA) SearchQA: A New Q&A Dataset Augmented with Context from a Search Engine. Matthew Dunn, Levent Sagun, Mike Higgins, V. Ugur Guney, Volkan Cirik, and Kyunghyun Cho. arXiv preprint arXiv:1704.05179 (2017).

(QuAC) QuAC : Question Answering in Context. Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wen-tau Yih, Yejin Choi, Percy Liang, and Luke Zettlemoyer. arXiv preprint arXiv:1808.07036 (2018).

(CoQA) CoQA: A Conversational Question Answering Challenge. Siva Reddy, Danqi Chen, and Christopher D. Manning. arXiv preprint arXiv:1808.07042 (2018).

(MCTest) MCTest: A Challenge Dataset for the Open-Domain Machine Comprehension of Text. Matthew Richardson, Christopher J.C. Burges, and Erin Renshaw. EMNLP 2013.

(CNN/Daily Mail) Teaching Machines to Read and Comprehend. Hermann, Karl Moritz, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. NIPS 2015.

(CBT) The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations. Felix Hill, Antoine Bordes, Sumit Chopra, and Jason Weston. arXiv preprint arXiv:1511.02301 (2015).

(bAbi) Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks. Jason Weston, Antoine Bordes, Sumit Chopra, Alexander M. Rush, Bart van Merriënboer, Armand Joulin, and Tomas Mikolov. arXiv preprint arXiv:1502.05698 (2015).

(LAMBADA) The LAMBADA Dataset:Word Prediction Requiring a Broad Discourse Context. Denis Paperno, Germ ́an Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fern ́andez. ACL 2016.

(SCT) LSDSem 2017 Shared Task: The Story Cloze Test. Nasrin Mostafazadeh, Michael Roth, Annie Louis,Nathanael Chambers, and James F. Allen. ACL 2017 workshop.

(Who did What) Who did What: A Large-Scale Person-Centered Cloze Dataset Takeshi Onishi, Hai Wang, Mohit Bansal, Kevin Gimpel, and David McAllester. EMNLP 2016.

(NewsQA) NewsQA: A Machine Comprehension Dataset. Adam Trischler, Tong Wang, Xingdi Yuan, Justin Harris, Alessandro Sordoni, Philip Bachman, and Kaheer Suleman. arXiv preprint arXiv:1611.09830 (2016).

(RACE) RACE: Large-scale ReAding Comprehension Dataset From Examinations. Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. EMNLP 2017.

(ARC) Think you have Solved Question Answering?Try ARC, the AI2 Reasoning Challenge. Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot,Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. arXiv preprint arXiv:1803.05457 (2018).

(MCScript) MCScript: A Novel Dataset for Assessing Machine Comprehension Using Script Knowledge. Simon Ostermann, Ashutosh Modi, Michael Roth, Stefan Thater, and Manfred Pinkal. arXiv preprint arXiv:1803.05223.

(NarrativeQA) The NarrativeQA Reading Comprehension Challenge. Tomáš Kočiský, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gábor Melis, and Edward Grefenstette. TACL 2018.

(DuoRC) DuoRC: Towards Complex Language Understanding with Paraphrased Reading Comprehension. Amrita Saha, Rahul Aralikatte, Mitesh M. Khapra, and Karthik Sankaranarayanan. ACL 2018.

(CLOTH) Large-scale Cloze Test Dataset Created by Teachers. Qizhe Xie, Guokun Lai, Zihang Dai, and Eduard Hovy. EMNLP 2018.

(DuReader) DuReader: a Chinese Machine Reading Comprehension Dataset from Real-world Applications. Wei He, Kai Liu, Yajuan Lyu, Shiqi Zhao, Xinyan Xiao, Yuan Liu, Yizhong Wang, Hua Wu, Qiaoqiao She, Xuan Liu, Tian Wu, and Haifeng Wang. ACL 2018 Workshop.

(CliCR) CliCR: a Dataset of Clinical Case Reports for Machine Reading Comprehension. Simon Suster and Walter Daelemans. NAACL 2018.

```
