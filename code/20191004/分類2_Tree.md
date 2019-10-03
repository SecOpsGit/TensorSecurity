

# DecisionTreeClassifier

```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# 導入數據
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filename, names=names)

# 將資料分為輸入資料和輸出結果
array = data.values

X = array[:, 0:8]
Y = array[:, 8]


num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)

model = DecisionTreeClassifier()

result = cross_val_score(model, X, Y, cv=kfold)

print(result.mean())

```

# Ensemble Learning
```
A Comprehensive Guide to Ensemble Learning (with Python codes)
https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/

https://wacamlds.podia.com/coding-first-project-with-diabetes-dataset-end-to-end-data-science-recipes-in-r-and-mysql
```
# VotingClassifier
```
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# 導入數據
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filename, names=names)


# 將資料分為輸入資料和輸出結果
array = data.values

X = array[:, 0:8]
Y = array[:, 8]


num_folds = 10
seed = 7

kfold = KFold(n_splits=num_folds, random_state=seed)


cart = DecisionTreeClassifier()

models = []

model_logistic = LogisticRegression()
models.append(('logistic', model_logistic))

model_cart = DecisionTreeClassifier()
models.append(('cart', model_cart))

model_svc = SVC()
models.append(('svm', model_svc))

ensemble_model = VotingClassifier(estimators=models)

result = cross_val_score(ensemble_model, X, Y, cv=kfold)

print(result.mean())
```

# Bagging

## BaggingClassifier
```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# 導入數據
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filename, names=names)

# 將資料分為輸入資料和輸出結果
array = data.values

X = array[:, 0:8]
Y = array[:, 8]

num_folds = 10
seed = 7

kfold = KFold(n_splits=num_folds, random_state=seed)


cart = DecisionTreeClassifier()

num_tree = 100

model = BaggingClassifier(base_estimator=cart, n_estimators=num_tree, random_state=seed)

result = cross_val_score(model, X, Y, cv=kfold)

print(result.mean())

```


## RandomForestClassifier
```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 導入數據
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filename, names=names)


# 將資料分為輸入資料和輸出結果
array = data.values

X = array[:, 0:8]
Y = array[:, 8]


num_folds = 10
seed = 7

kfold = KFold(n_splits=num_folds, random_state=seed)


num_tree = 100
max_features = 3

model = RandomForestClassifier(n_estimators=num_tree, random_state=seed, max_features=max_features)

result = cross_val_score(model, X, Y, cv=kfold)

print(result.mean())

```


# ExtraTreesClassifier
```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier

# 導入數據
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filename, names=names)

# 將資料分為輸入資料和輸出結果
array = data.values
X = array[:, 0:8]
Y = array[:, 8]


num_folds = 10
seed = 7

kfold = KFold(n_splits=num_folds, random_state=seed)


num_tree = 100
max_features = 7

model = ExtraTreesClassifier(n_estimators=num_tree, random_state=seed, max_features=max_features)

result = cross_val_score(model, X, Y, cv=kfold)

print(result.mean())

```


# boosting


## GradientBoostingClassifier
```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

# 導入數據
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filename, names=names)


# 將資料分為輸入資料和輸出結果
array = data.values

X = array[:, 0:8]
Y = array[:, 8]


num_folds = 10
seed = 7

kfold = KFold(n_splits=num_folds, random_state=seed)

num_tree = 100

model = GradientBoostingClassifier(n_estimators=num_tree, random_state=seed)

result = cross_val_score(model, X, Y, cv=kfold)

print(result.mean())

```
## AdaBoostClassifier
```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier


# 導入數據
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filename, names=names)


# 將資料分為輸入資料和輸出結果
array = data.values

X = array[:, 0:8]
Y = array[:, 8]


num_folds = 10
seed = 7

kfold = KFold(n_splits=num_folds, random_state=seed)


num_tree = 30

model = AdaBoostClassifier(n_estimators=num_tree, random_state=seed)

result = cross_val_score(model, X, Y, cv=kfold)

print(result.mean())
```




## xgboost
```
!pip install xgboost
```
```
改自https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/

作業:https://iter01.com/166318.html
```
```
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 導入數據
filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv(filename, names=names)

# 將資料分為輸入資料和輸出結果
array = data.values

X = array[:, 0:8]
Y = array[:, 8]

# split data into train and test sets
seed = 7
test_size = 0.33

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data

model = XGBClassifier()

model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

```
Accuracy: 77.95%
```
##  LightGBM
```

```
## catboost
```
https://catboost.ai/

pip install catboost
```
```
https://github.com/catboost/tutorials/blob/master/classification/classification_tutorial.ipynb
```
```

```

## h2o

```

```

```
https://inclass.kaggle.com/sudalairajkumar/getting-started-with-h2o
```


```

```
