# CatBoost (2017) categorical boosting
```
官方地址：https://tech.yandex.com/CatBoost/
文檔地址：https://tech.yandex.com/catboost/doc/dg/concepts/about-docpage/
https://catboost.ai/

github位址：https://github.com/catboost/catboost
論文地址：https://arxiv.org/abs/1706.09516
```

```
https://blog.csdn.net/appleyuchi/article/details/85413352

https://www.biaodianfu.com/catboost.html
```

###
```
Gradient Boosting(梯度提升) + Categorical Features(類別型特徵)

許多利用GBDT技術的演算法（例如，XGBoost、LightGBM），構建一棵樹分為兩個階段：
[Step 1]選擇樹結構
[Step 2]在樹結構固定後計算葉子節點的值。

為了選擇最佳的樹結構，
演算法通過枚舉不同的分割，用這些分割構建樹，
對得到的葉子節點中計算值，
然後對得到的樹計算評分，
最後選擇最佳的分割。

兩個階段葉子節點的值都是被當做梯度或牛頓步長的近似值來計算。

CatBoost
第一階段採用梯度步長的無偏估計，
第二階段使用傳統的GBDT方案執行。
```
### catboost @ Google Colab
```
!pip install catboost

!pip list | grep catboost
```
### 範例程式
```
CatBoost tutorials
https://github.com/catboost/tutorials
```
