#
```
https://www.knowledgedict.com/tutorial/sklearn-dataset.html
```
# sklearn.datasets.load_boston
```
波士頓房價資料集
scikit-learn內建波士頓房價資料集
該資料集來源於1978年美國某經濟學雜誌上。
該資料集包含若干波士頓房屋的價格及其各項資料，

每個資料項目包含14個相關特徵資料，
分別是房屋均價及周邊犯罪率、是否在河邊、師生比等相關資訊，其中
最後一項資料是該區域房屋均價。

波士頓房價資料集是一個回歸問題，共有506個樣本，13個輸入變數和1個輸出變數。

資料集的創建者： Harrison, D. and Rubinfeld, D.L.
```
### 波士頓房價資料集的相關統計
```
資料集樣本實例數：506個。

特徵（屬性）個數：13個特徵屬性和1個目標數值。

特徵（屬性）資訊（按照順序）：

CRIM - 城鎮人均犯罪率，per capita crime rate by town
ZN - 住宅用地所占比例（每25000平方英尺），proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS - 城鎮非商業用地所占比例，proportion of non-retail business acres per town
CHAS - 查理斯河的指標虛擬化（區域在河附近用1表示，否則為0），Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
NOX - 一氧化氮濃度，nitric oxides concentration (parts per 10 million)
RM - 每棟住宅的房間數，average number of rooms per dwelling
AGE - 1940年之前建成的自用住宅的比例，proportion of owner-occupied units built prior to 1940
DIS - 距離5個波士頓就業中心的加權距離，weighted distances to five Boston employment centres
RAD - 距離高速公路的便利指數，index of accessibility to radial highways
TAX - 每10000美元的全值財產稅率，full-value property-tax rate per $10,000
PTRATIO - 城鎮師生比例，pupil-teacher ratio by town
B - 城鎮中黑人比例，1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT - 低收入人群的百分比，% lower status of the population
MEDV - 房屋房價的中位數（以千美元為單位），Median value of owner-occupied homes in $1000's
```
```
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.keys())
print(boston.data.shape)
print(boston.feature_names)
print(boston.DESCR)
```


# sklearn.datasets.load_diabetes()
```
原始碼分析
https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/datasets/base.py

sklearn.datasets.load_diabetes(return_X_y=False)
L569-L619
```
### 函數回傳
```

    return Bunch(data=data, target=target, DESCR=fdescr,
                 feature_names=['age', 'sex', 'bmi', 'bp',
                                's1', 's2', 's3', 's4', 's5', 's6'],
                 data_filename=data_filename,
                 target_filename=target_filename)
```


```
原始碼分析
https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/datasets/base.py

L327-L399

```
# sklearn.datasets.load_iris()
### 函數回傳
```
    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['sepal length (cm)', 'sepal width (cm)',
                                'petal length (cm)', 'petal width (cm)'],
                 filename=iris_csv_filename)
```
### 測試範例
```
from sklearn.datasets import load_iris
data = load_iris()
data.target[[10, 25, 50]]
list(data.target_names)

```

#
