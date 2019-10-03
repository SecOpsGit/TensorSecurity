# 測試資料集

```
!wget https://raw.githubusercontent.com/MyDearGreatTeacher/AI201909/master/data/pima_data.csv
```

# LogisticRegression()

```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
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

model = LogisticRegression()

result = cross_val_score(model, X, Y, cv=kfold)

print(result.mean())

```
# KNN
```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


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

model = KNeighborsClassifier()

result = cross_val_score(model, X, Y, cv=kfold)

print(result.mean())

```

# SVC

```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

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


model = SVC()

result = cross_val_score(model, X, Y, cv=kfold)

print(result.mean())

```

# naive_bayes GaussianNB

```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

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


model = GaussianNB()

result = cross_val_score(model, X, Y, cv=kfold)

print(result.mean())

```

# LDA==LinearDiscriminantAnalysis()

```
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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


model = LinearDiscriminantAnalysis()

result = cross_val_score(model, X, Y, cv=kfold)

print(result.mean())

```

#

```

```

#

```

```


