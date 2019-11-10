# 遷移學習transfer learning
```
將一個場景中學到的知識遷移到另一個場景

將貓狗分類的學習模型遷移到其它相似的任務上面
用來分辨老鷹和布穀鳥（因為都是拍攝的真實圖片，所以屬於相同的域，抽取特徵的方法相同），
或者是分別卡通影象（卡通影象和真實圖片屬於不同的域，遷移時要消除域之間的差異）
```
```
一、什麼是遷移學習？
二、為什麼要遷移學習？
三、具體怎麼做？
3.1目標資料和原始資料都有標籤
    3.1.1模型Fine-tune
    3.1.2模型Multitask Learning
3.2原始資料有標籤，目標資料沒有標籤
    3.2.1域對抗 Domain-adversarial training
    3.2.2零樣本學習 Zero-shot Learning
```
```
臺灣大學李宏毅老師的機器學習課程
ML Lecture 19: Transfer Learning
https://www.youtube.com/watch?v=qD6iD4TFsdQ
https://blog.csdn.net/xzy_thu/article/details/71921263
```
### 為什麼要遷移學習？
```
使用深度學習技術解決問題的過程中，最常見的障礙在於：模型有大量的引數需要訓練，因此需要海量的訓練資料作支撐

在面對某一領域的具體問題[醫療圖片....]時，通常可能無法得到構建模型所需的規模的資料
藉助遷移學習，在一個模型訓練任務中針對某種型別資料獲得的關係也可以輕鬆地應用於同一領域的不同問題
```
### 各種不同的遷移學習作法與策略
```
A Survey on Transfer Learning
Sinno Jialin Pan and Qiang Yang    Fellow, IEEE
http://www.cse.ust.hk/faculty/qyang/Docs/2009/tkde_transfer_learning.pdf
```
目標資料和原始資料都有標籤:Fine-tune
```
任務現狀：
1. 原始資料和目標資料都有標籤     2. 目標資料很少

範例：監督學習|電商小品類商品識別
來源資料：Imagenet多種物體圖片    目標資料：電商小品類商品圖片

基本思想：
1.用原始Imagenet訓練一個效果比較好的模型
2.用訓練好的模型在自己的資料集上進行調優（重點：微微調整，防止過擬合）

模型Fine-tune實現方法：
方法1：保守訓練 Conservative Training
      先訓練好一個Model，得到引數和是某類圖片的概率向量，
      接著有兩種不同角度來訓練新的資料（注意：重點是微微調整，防止過擬合）：
      保證原模型引數變化不大
      讓輸出結果的概率向量很接近
方法2：層遷移 Layer Transfer
      規定某些層的引數固定（learning rate設定為0或者很小），不允許變動。
      防止無法無天的變動
      挑戰問題：應該拷貝哪些層的引數呢？
          語音識別問題一般是最後的一些層
          影象的通常是開始的一些層
```
### 域對抗 Domain-adversarial training[2015]
```
https://arxiv.org/abs/1505.07818
Domain-Adversarial Training of Neural Networks
Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, 
François Laviolette, Mario Marchand, Victor Lempitsky
(Submitted on 28 May 2015 (v1), last revised 26 May 2016 (this version, v4))

We introduce a new representation learning approach for domain adaptation, 
in which data at training and test time come from similar but different distributions. 
Our approach is directly inspired by the theory on domain adaptation suggesting that, 
for effective domain transfer to be achieved, 
predictions must be made based on features 
that cannot discriminate between the training (source) and test (target) domains. 

The approach implements this idea in the context of neural network architectures 
that are trained on labeled data from the source domain 
and unlabeled data from the target domain (no labeled target-domain data is necessary). 

As the training progresses, the approach promotes the emergence of features that are 
(i) discriminative for the main learning task on the source domain and 
(ii) indiscriminate with respect to the shift between the domains. 

We show that this adaptation behaviour can be achieved in almost any feed-forward model 
by augmenting it with few standard layers and a new gradient reversal layer. 

The resulting augmented architecture can be trained using standard backpropagation 
and stochastic gradient descent, and can thus be implemented with little effort 
using any of the deep learning packages. 

We demonstrate the success of our approach for two distinct classification problems 
(document sentiment analysis and image classification), 
where state-of-the-art domain adaptation performance on standard benchmarks is achieved. 
We also validate the approach for descriptor learning task in the context of 
person re-identification application.
```
零樣本學習 Zero-shot Learning
```
任務描述：
有了一個能區分貓和狗的模型，但此時來了一個之前從未見過的草泥馬，應該怎麼區分呢？

解決思路：
不識別動物本身，只識別動物的屬性，
將動物的屬性放在一個大的資料庫裡面，
有了新的樣本，直接對比資料庫中的屬性來區分是哪種動物

https://www.itread01.com/content/1541847496.html
```

```
http://cs231n.github.io/transfer-learning/[完整資料來源]

https://blog.csdn.net/tommorrow12/article/details/80318956[不完整]

遷移學習兩種類型：三種
[1]ConvNet as fixed feature extractor：
利用在大資料集(如ImageNet)上預訓練過的ConvNet(如AlexNet，VGGNet)，移除最後幾層（一般是最後分類器），
將剩下的ConvNet作為應用於新資料集的固定不變的特徵提取器，輸出特徵稱為CNN codes，
如果在預訓練網路上是經過ReLUd，那這些codes也要經過ReLUd（important for performance）；

提取出所有CNN codes之後，再基於新資料集訓練一個線性分類器（Linear SVM or Softmax classifier）；

[2]Fine-tuning the ConvNet：
第一步：在新資料集上，替換預訓練ConvNet頂層的分類器並retrain該分類器；
第二步：以較小的學習率繼續反向傳播來微調預訓練網路的權重，
兩種做法：微調ConvNet的所有層，或者保持some earlier layers fixed (due to overfitting concerns) ，
只微調some higher-level portion of the network；

原理：一般認為CNN中前端（靠近輸入圖片）的層提取的是紋理、色彩等基本特徵，
越靠近後端，提取的特徵越高級、抽象、面向具體任務。

所以更普遍的微調方法是：固定其他參數不變，替換預訓練網路最後幾層，
基於新資料集重新訓練最後幾層的參數（之前的層參數保持不變，作為特徵提取器），
之後再用較小的學習率將網路整體訓練。

一些開源的Pretrained models：Model Zoo

When and how to fine-tune?
四個主要場景：
新資料集小，且與原始資料集相似：要考慮小資料集過度擬合問題；利用CNN codes 訓練一個線性分類器
新資料集大，且與原始資料集相似：不用考慮過度擬合，可嘗試微調整個神經網路；
新資料集小，並與原始資料集差距大：訓練一個線性分類器，而新資料集與原始資料集差距大，
         work better to train the SVM classifier from activations somewhere earlier in the network
新資料集大，且與原始資料集差距大，使用預訓練模型參數，基於新資料集微調整個神經網路

Practical advice:a few additional things to keep in mind when performing Transfer Learning:
Constraints from pretrained models：使用預訓練網路，新資料集使用的架構將受限，
                      比如不能隨意take out Conv layers from the pretrained network；

Learning rates：微調 ConvNet權重（ConvNet weights are relatively good）時的學習率要比新的線性分類器
               （權重是隨機初始化的）的學習率要小；
```
### 範例學習
```
Transfer learning with a pretrained ConvNet
https://www.tensorflow.org/tutorials/images/transfer_learning

```
# 各類型的應用

###
```
Transfer Learning from Transformers to Fake News Challenge Stance Detection (FNC-1) Task
Valeriya Slovikovskaya
(Submitted on 31 Oct 2019)

https://arxiv.org/abs/1910.14353

In this paper, we report improved results of 
the Fake News Challenge Stage 1 (FNC-1) stance detection task. 

This gain in performance is due to the generalization power of large language models 
based on Transformer architecture, invented, trained and publicly released over the last two years. 

Specifically 
(1) we improved the FNC-1 best performing model adding BERT sentence embedding 
    of input sequences as a model feature, 
(2) we fine-tuned BERT, XLNet, and RoBERTa transformers on FNC-1 extended dataset 
    and obtained state-of-the-art results on FNC-1 task.
```
### 醫學圖片辨識
```
Predictive modeling of brain tumor: A Deep learning approach
Priyansh Saxena, Akshat Maheshwari, Saumil Maheshwari
(Submitted on 6 Nov 2019)
https://arxiv.org/abs/1911.02265

Image processing concepts can visualize the different anatomy structure of the human body. 

Recent advancements in the field of deep learning have made it possible to 
detect the growth of cancerous tissue just by a patient's brain Magnetic Resonance Imaging (MRI) scans. 

These methods require very high accuracy and meager false negative rates to be of any practical use. 

This paper presents a Convolutional Neural Network (CNN) based transfer learning approach 
to classify the brain MRI scans into two classes using three pre-trained models. 

The performances of these models are compared with each other. 

Experimental results show that 
the Resnet-50 model achieves the highest accuracy and least false negative rates as 95% 
and zero respectively. 

It is followed by VGG-16 and Inception-V3 model with an accuracy of 90% and 55% respectively.
```

### 超奇怪字母的手寫辨識 Devanagari alphabets
```
Transfer Learning using CNN for Handwritten Devanagari Character Recognition
Nagender Aneja, Sandhya Aneja
(Submitted on 19 Sep 2019)
https://arxiv.org/abs/1909.08774

This paper presents an analysis of pre-trained models to recognize handwritten Devanagari alphabets 
using transfer learning for Deep Convolution Neural Network (DCNN). 

This research implements AlexNet, DenseNet, Vgg, and Inception ConvNet as a fixed feature extractor. 

We implemented 15 epochs for each of AlexNet, DenseNet 121, DenseNet 201, Vgg 11, Vgg 16, Vgg 19, 
and Inception V3. 

Results show that Inception V3 performs better in terms of accuracy achieving 99% accuracy 
with average epoch time 16.3 minutes 
while AlexNet performs fastest with 2.2 minutes per epoch and achieving 98\% accuracy.
```
