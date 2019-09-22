
# 安裝xgboost
!pip install xgboost

# 下載資料集
!wget https://raw.githubusercontent.com/MyDearGreatTeacher/Data/master/iris.data

# 主程式

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

