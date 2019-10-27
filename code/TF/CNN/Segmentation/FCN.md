#
```
Fully Convolutional Networks for Semantic Segmentation
Evan Shelhamer, Jonathan Long, Trevor Darrell
(Submitted on 20 May 2016)
https://arxiv.org/abs/1605.06211
論文地址：https://arxiv.org/pdf/1605.06211v1.pdf
論文視訊地址：http://techtalks.tv/talks/fully-convolutional-networks-for-semantic-segmentation/61606/
GitHub資源：https://github.com/shekkizh/FCN.tensorflow
            https://github.com/EternityZY/FCN-TensorFlow

https://www.itread01.com/content/1546918763.html
https://blog.csdn.net/qq_16761599/article/details/80069824
```
```
Convolutional networks are powerful visual models that yield hierarchies of features. 

We show that convolutional networks by themselves, trained end-to-end, pixels-to-pixels, 
improve on the previous best result in semantic segmentation. 

Our key insight is to build "fully convolutional" networks that 
take input of arbitrary size and produce correspondingly-sized output with efficient inference and learning. 

We define and detail the space of fully convolutional networks, 
explain their application to spatially dense prediction tasks, 
and draw connections to prior models. 

We adapt contemporary classification networks (AlexNet, the VGG net, and GoogLeNet) into fully convolutional networks 
and transfer their learned representations by fine-tuning to the segmentation task. 

We then define a skip architecture that combines semantic information from a deep, coarse layer 
with appearance information from a shallow, fine layer to produce accurate and detailed segmentations. 

Our fully convolutional network achieves improved segmentation of 
PASCAL VOC (30% relative improvement to 67.2% mean IU on 2012), NYUDv2, SIFT Flow, and PASCAL-Context, 
while inference takes one tenth of a second for a typical image.
```

```

```
