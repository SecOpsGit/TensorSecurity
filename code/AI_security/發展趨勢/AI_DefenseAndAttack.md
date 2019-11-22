# AI_DefenseAndAttack
```
https://openreview.net/pdf?id=BkJ3ibb0-
```
# BLACK-BOX ATTACK MODELS
```


```

# WHITE-BOX ATTACK MODELS
```
Fast Gradient Sign Method (FGSM) 
```

```
Randomized Fast Gradient Sign Method (RAND+FGSM)
```

```
The Carlini-Wagner (CW) attack The CW attack is an effective optimization-based attack model
(Carlini & Wagner, 2017).

```

```
the Fast Gradient Sign Method (FGSM) (Goodfellow et al., 2015), 
the Randomized Fast Gradient Sign Method(RAND+FGSM) (Tramer et al., 2017)
the Carlini-Wagner (CW) attack (Carlini & Wagner, `2017). 

Although other attack models exist, such as 
the Iterative FGSM (Kurakin et al., 2017), 
theJacobian-based Saliency Map Attack (JSMA) (Papernot et al., 2016b), and 
Deepfool (MoosaviDezfooli et al., 2016), 

we focus on these three models as they cover a good breadth of attack algorthims. 

FGSM is a very simple and fast attack algorithm which makes it extremely amenable to real-time attack deployment. 
On the other hand, RAND+FGSM, an equally simple attack, increases the power of FGSM for white-box attacks (Tramer et al., 2017), 
and finally, the CW attack is one of the most powerful white-box attacks to-date (Carlini & Wagner, 2017)
```


# DEFENSE MECHANISMS防禦機制
```
ADVERSARIAL TRAINING

gradient masking (Papernot et al., 2016c; 2017; Tramer et al., 2017). 
```

```
Defensive distillation (Papernot et al., 2016d)

while defensive distillation is effective against white-box attacks, it fails to adequately protect against black-box
attacks transferred from other networks (Carlini & Wagner, 2017)
```

```
MAGNET
Meng & Chen (2017)
```

### ADD-GAN(2018)
```
https://arxiv.org/abs/1809.04758
Anomaly Detection Discriminative GAN (ADD-GAN)
https://github.com/zblasingame/ADD-GAN

Anomaly Detection with Generative Adversarial Networks for Multivariate Time Series
Dan Li, Dacheng Chen, Jonathan Goh, See-kiong Ng
(Submitted on 13 Sep 2018 (v1), last revised 15 Jan 2019 (this version, v3))
```

### Defense-GAN

```
Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models (published in ICLR2018)

https://github.com/kabkabm/defensegan
https://openreview.net/pdf?id=BkJ3ibb0-
```

```


```


```

```



```


```
