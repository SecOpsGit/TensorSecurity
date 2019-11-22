#
```

```

### 論文
```
Foolbox: A Python toolbox to benchmark the robustness of machine learning models
Jonas Rauber, Wieland Brendel, Matthias Bethge
(Submitted on 13 Jul 2017 (v1), last revised 20 Mar 2018 (this version, v3))
https://arxiv.org/abs/1707.04131

Even todays most advanced machine learning models are easily fooled by almost imperceptible perturbations of their inputs. 
Foolbox is a new Python package to generate such adversarial perturbations 
and to quantify and compare the robustness of machine learning models. 
It is build around the idea that the most comparable robustness measure is 
the minimum perturbation needed to craft an adversarial example. 

To this end, Foolbox provides reference implementations of most published adversarial attack methods alongside some new ones, 
all of which perform internal hyperparameter tuning to find the minimum adversarial perturbation. 

Additionally, Foolbox interfaces with most popular deep learning frameworks 
such as PyTorch, Keras, TensorFlow, Theano and MXNet 
and allows different adversarial criteria such as targeted misclassification and top-k misclassification 
as well as different distance measures. 

The code is licensed under the MIT license and is openly available at this https URL . 
The most up-to-date documentation can be found at this http URL .
```
### User Guide
```
https://foolbox.readthedocs.io/en/latest/
```
### 
```
foolbox：一款神奇的Python工具箱
PYTHON教程 · 發表 2018-10-04

https://www.itread01.com/p/524611.html


原文:foolbox

作者:Jonas Rauber
&;&;Wieland Brendel


翻譯:Vincent


Foolbox簡介
Foolbox是一個Python工具箱,它可以建立一些對抗樣本從而來迷惑神經網路。 它需要Python,NumPy和SciPy。


安裝
pip install foolbox
我們測試的時候使用的是Python 2.7、3.5和3.6版本。當然Python其它版本也可以工作。但是我們建議使用Python 3版本!


文件

可在readthedocs上獲取文件:http://foolbox.readthedocs.io/


樣例
import foolbox
import keras
from keras.applications.resnet50 import ResNet50, preprocess_input
# instantiate model
keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='imagenet')
fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255),preprocess_fn=preprocess_input)
# get source image and label
image, label = foolbox.utils.imagenet_example()
# apply attack on source image
attack= foolbox.attacks.FGSM(fmodel)
adversarial = attack(image, label)
其它一系列深度學習的介面也是可以用的,例如TensorFlow,PyTorch,Theano,Lasagne和MXNet等。


model = foolbox.models.TensorFlowModel(images, logits, bounds=(0, 255))
model = foolbox.models.PyTorchModel(torchmodel, bounds=(0, 255), num_classes=1000)
# etc.
不同的對抗標準,例如Top-k,特定的Target Class或Original Class、Target Class的目標概率值可以傳遞給attack,例如:


criterion = foolbox.criteria.TargetClass(22)
attack= foolbox.attacks.FGSM(fmodel, criterion)
```
### https://medium.com/@falconives
```
Day 97 — Adversarial Example Attack against Keras ResNet50
Day 98 — Adversarial Example Attack against Keras InceptionV3
Day 99 — Foolbox LBFGS Attack against All Keras Applications


Day 96 — WGAN for MNIST
Day 95 — DC-GAN for MNIST Dataset
Day 94 — AC-GAN for MNIST Dataset
Day 93 — Adversarial Autoencoder (AAE) for MNIST Dataset
Day 92 — Simple GAN for MNIST
今日主題：使用原版生成對抗網路模擬MNIST圖像資料

Day 91 — Gensim for Text Summarization
Day 89 — FastText for IMDB Classification
Day 88 — Word2Vec Nietzsche’s Style Generator
今日主題：使用Word2Vec產生尼采風格的短文
Day 87 — Word Embedding for News Classification
今日主題：使用預訓練的Word Embedding做新聞文本分類

```
```
使用Foolbox LBFGS Attack攻擊Keras內建分類器
Falconives  Aug 28, 2018 

https://medium.com/@falconives/day-99-foolbox-lbfgs-attack-against-all-keras-applications-6e2ae6e29837
```
```
綜合評比一下各模型的抵抗力（數字越大抵抗力越強）：
InceptionResNet V2: 436.04573130607605
Inception V3: 160.60878705978494
Xception V1: 142.7184829711914
VGG-16: 80.00009250640869
VGG-19: 84.68800973892212
ResNet50: 109.09112095832825
NASNet: 424.07666277885437
```
