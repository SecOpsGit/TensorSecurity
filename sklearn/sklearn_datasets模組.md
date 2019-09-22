#
```
https://www.knowledgedict.com/tutorial/sklearn-dataset.html
```
# sklearn.datasets.load_boston
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
