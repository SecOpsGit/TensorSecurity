#
```
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.keys())
print(boston.data.shape)
print(boston.feature_names)
print(boston.DESCR)
```
```
bos = pd.DataFrame(boston.data)
print(bos.head())
bos.columns = boston.feature_names
print(bos.head())
print(boston.target.shape)
bos['PRICE'] = boston.target
print(bos.head())
```

# Exploratory Data Analysis
```
print(bos.describe())

```
```
https://medium.com/@haydar_ai/learning-data-science-day-9-linear-regression-on-boston-housing-dataset-cd62a80775ef
https://towardsdatascience.com/machine-learning-project-predicting-boston-house-prices-with-regression-b4e47493633d
```
# Linear Regression
```
Simple Linear Regression
```
```
Multiple Linear Regression
```
# Tree-based Regressor
```
DecisionTreeRegressor

```

```
GradientBoostingRegressor

```
#
```
SVR

```
