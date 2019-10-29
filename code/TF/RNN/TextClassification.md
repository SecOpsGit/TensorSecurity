#
```

```

### DataSets
```
20NEWS


```
### 
```
Simplifying Graph Convolutional Networks

Graph Convolutional Networks for Text Classification


 GraphStar
Graph Star Net for Generalized Multi-Task Learning
```

### SGC(2019)
```
Simplifying Graph Convolutional Networks
Felix Wu, Tianyi Zhang, Amauri Holanda de Souza Jr., Christopher Fifty, Tao Yu, Kilian Q. Weinberger
(Submitted on 19 Feb 2019 (v1), last revised 20 Jun 2019 (this version, v2))
https://arxiv.org/abs/1902.07153
[PyTorch實作]https://github.com/Tiiiger/SGC
[PyTorch實作]https://github.com/reallygooday/60daysofudacity

Graph Convolutional Networks (GCNs) and their variants have experienced significant attention 
and have become the de facto methods for learning graph representations. 

GCNs derive inspiration primarily from recent deep learning approaches, 
and as a result, may inherit unnecessary complexity and redundant computation. 

In this paper, we reduce this excess complexity through successively removing nonlinearities 
and collapsing weight matrices between consecutive layers. 

We theoretically analyze the resulting linear model and show that it corresponds to a fixed low-pass filter 
followed by a linear classifier. 

Notably, our experimental evaluation demonstrates that these simplifications do not negatively impact accuracy 
in many downstream applications. Moreover, the resulting model scales to larger datasets, 
is naturally interpretable, and yields up to two orders of magnitude speedup over FastGCN.
```

