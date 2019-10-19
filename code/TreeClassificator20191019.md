
# xgboost
```
# 安裝xgboost
!pip install xgboost

# 下載資料集
!wget https://raw.githubusercontent.com/MyDearGreatTeacher/Data/master/iris.data
```
#### 範例程式
```
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
	iris_feature_E = "sepal lenght", "sepal width", "petal length", "petal width"
	iris_feature = "the length of sepal", "the width of sepal", "the length of petal", "the width of petal"
	iris_class = "Iris-setosa", "Iris-versicolor", "Iris-virginica"
	
	data = pd.read_csv("iris.data", header=None)
	iris_types = data[4].unique()
	for i, type in enumerate(iris_types):
		data.set_value(data[4] == type, 4, i)
	x, y = np.split(data.values, (4,), axis=1)
 
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=1)
 
	data_train = xgb.DMatrix(x_train, label=y_train)
	data_test = xgb.DMatrix(x_test, label=y_test)
	watchlist = [(data_test, 'eval'), (data_train, 'train')]
	param = {'max_depth':3, 'eta':1, 'silent':1, 'objective':'multi:softmax', 'num_class':3}
 
	bst = xgb.train(param, data_train, num_boost_round=10, evals=watchlist)
	y_hat = bst.predict(data_test)
	result = y_test.reshape(1, -1) == y_hat
	print('the accuracy:\t', float(np.sum(result)) / len(y_hat))
```

# LightGBM(2017)  Light Gradient Boosting Machine
```
https://github.com/Microsoft/LightGBM
https://lightgbm.readthedocs.io/en/latest/
https://lightgbm.apachecn.org/#/
```
```
Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree". Advances in Neural Information Processing Systems 30 (NIPS 2017), pp. 3149-3157.

Qi Meng, Guolin Ke, Taifeng Wang, Wei Chen, Qiwei Ye, Zhi-Ming Ma, Tie-Yan Liu. "A Communication-Efficient Parallel Algorithm for Decision Tree". Advances in Neural Information Processing Systems 29 (NIPS 2016), pp. 1279-1287.

Huan Zhang, Si Si and Cho-Jui Hsieh. "GPU Acceleration for Large-scale Tree Boosting". SysML Conference, 2018.

```
### LGBMRegressor()
```
https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor
```

### 範例程式
```
import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

iris = load_iris()
data=iris.data
target = iris.target

X_train,X_test,y_train,y_test =train_test_split(data,target,test_size=0.25)

gbm = lgb.LGBMRegressor(learning_rate=0.03,n_estimators=200，max_depth=8)

gbm.fit(X_train, y_train)

#預測結果

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

```

# catboost
```
catboost === Gradient Boosting(梯度提升) + Categorical Features(類別型特徵)

```
```
官方地址：https://catboost.ai/
         https://tech.yandex.com/CatBoost/
github位址：https://github.com/catboost/catboost
論文地址：https://arxiv.org/abs/1706.09516
文檔地址：https://tech.yandex.com/catboost/doc/dg/concepts/about-docpage/

```

```
!pip install catboost
```
### catboost範例程式
```
import numpy as np
import catboost as cb
 
train_data = np.random.randint(0, 100, size=(100, 10))
train_label = np.random.randint(0, 2, size=(100))
test_data = np.random.randint(0,100, size=(50,10))
 
model = cb.CatBoostClassifier(iterations=2, depth=2, learning_rate=0.5, loss_function='Logloss',
                              logging_level='Verbose')
model.fit(train_data, train_label, cat_features=[0,2,5])
preds_class = model.predict(test_data)
preds_probs = model.predict_proba(test_data)
print('class = ',preds_class)
print('proba = ',preds_probs)
```
