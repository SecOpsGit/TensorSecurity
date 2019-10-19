# 決策樹分類鳶尾花資料

```
# 下載資料集
!wget https://raw.githubusercontent.com/MyDearGreatTeacher/Data/master/iris.data

```
#### 決策樹分類範例:鳶尾花資料的分類
```
資料來源:
https://blog.csdn.net/OliverkingLi/article/details/80596229
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pydotplus
 
if __name__ == "__main__":
   
	iris_feature_E = "sepal lenght", "sepal width", "petal length", "petal width"
	iris_feature = "the length of sepal", "the width of sepal", "the length of petal", "the width of petal"
	iris_class = "Iris-setosa", "Iris-versicolor", "Iris-virginica"
	
	data = pd.read_csv("iris.data", header=None)
	iris_types = data[4].unique()
	for i, type in enumerate(iris_types):
		data.set_value(data[4] == type, 4, i)
	x, y = np.split(data.values, (4,), axis=1)
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
	print(y_test)
 
	model = DecisionTreeClassifier(criterion='entropy', max_depth=6)
	model = model.fit(x_train, y_train)
	y_test_hat = model.predict(x_test)
	with open('iris.dot', 'w') as f:
		tree.export_graphviz(model, out_file=f)
	dot_data = tree.export_graphviz(model, out_file=None, feature_names=iris_feature_E, class_names=iris_class,
		filled=True, rounded=True, special_characters=True)
	graph = pydotplus.graph_from_dot_data(dot_data)
	graph.write_pdf('iris.pdf')
	f = open('iris.png', 'wb')
	f.write(graph.create_png())
	f.close()
 
	# 畫圖
	# 橫縱各採樣多少個值
	N, M = 50, 50
	# 第0列的範圍
	x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
	# 第1列的範圍
	x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
	t1 = np.linspace(x1_min, x1_max, N)
	t2 = np.linspace(x2_min, x2_max, M)
	# 生成網格採樣點
	x1, x2 = np.meshgrid(t1, t2)
    # # 無意義，只是為了湊另外兩個維度
    # # 打開該注釋前，確保注釋掉x = x[:, :2]
	x3 = np.ones(x1.size) * np.average(x[:, 2])
	x4 = np.ones(x1.size) * np.average(x[:, 3])
	# 測試點
	x_show = np.stack((x1.flat, x2.flat, x3, x4), axis=1)
	print("x_show_shape:\n", x_show.shape)
 
	cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
	cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
	# 預測值
	y_show_hat = model.predict(x_show)
	print(y_show_hat.shape)
	print(y_show_hat)
	# 使之與輸入的形狀相同
	y_show_hat = y_show_hat.reshape(x1.shape)
	print(y_show_hat)
	plt.figure(figsize=(15, 15), facecolor='w')
	# 預測值的顯示
	plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)
	print(y_test)
	print(y_test.ravel())
	# 測試資料
	plt.scatter(x_test[:, 0], x_test[:, 1], c=np.squeeze(y_test), edgecolors='k', s=120, cmap=cm_dark, marker='*')
	# 全部資料
	plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), edgecolors='k', s=40, cmap=cm_dark)
	plt.xlabel(iris_feature[0], fontsize=15)
	plt.ylabel(iris_feature[1], fontsize=15)
	plt.xlim(x1_min, x1_max)
	plt.ylim(x2_min, x2_max)
	plt.grid(True)
	plt.title('yuanwei flowers regressiong with DecisionTree', fontsize=17)
	plt.show()
 
	# 訓練集上的預測結果
	y_test = y_test.reshape(-1)
	print(y_test_hat)
	print(y_test)
	# True則預測正確，False則預測錯誤
	result = (y_test_hat == y_test)
	acc = np.mean(result)
	print('accuracy: %.2f%%' % (100 * acc))
 
    # 過擬合：錯誤率
	depth = np.arange(1, 15)
	err_list = []
	for d in depth:
		clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
		clf = clf.fit(x_train, y_train)
		# 測試資料
		y_test_hat = clf.predict(x_test)
		# True則預測正確，False則預測錯誤
		result = (y_test_hat == y_test)
		err = 1 - np.mean(result)
		err_list.append(err)
		print(d, 'error ratio: %.2f%%' % (100 * err))
	plt.figure(figsize=(15, 15), facecolor='w')
	plt.plot(depth, err_list, 'ro-', lw=2)
	plt.xlabel('DecisionTree Depth', fontsize=15)
	plt.ylabel('error ratio', fontsize=15)
	plt.title('DecisionTree Depth and Overfit', fontsize=17)
	plt.grid(True)
	plt.show()

```
# GradientBoostingClassifier

```
import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report


titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
#特徵選取
X = titanic[['pclass','age','sex']]
y = titanic['survived']
#對空白的age列進行填充，因為中位數和平均數對模型的影響最小，所以使用平均數進行填充
X['age'].fillna(X['age'].mean(),inplace=True)
#進行訓練集和測試集的分割
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)


vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
print(vec.feature_names_)#經過特徵轉換以後，我們發現凡是類別形的特徵都單獨剝離出來，數值型的則保持不變
X_test = vec.transform(X_test.to_dict(orient='record'))#對測試資料進行特徵轉換


#使用單一決策樹模型訓練及分析資料
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_y_predict = dtc.predict(X_test)


#使用隨機森林分類器進行整合模型的訓練及預測分析
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_predict = rfc.predict(X_test)


#使用梯度提升決策樹整合模型的訓練及分析
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_predict = gbc.predict(X_test)


print('The accuracy of decision tree is: ',dtc.score(X_test,y_test))
print(classification_report(dtc_y_predict,y_test))
print('\n'*2)
print('The accuracy of random forest classifier:',rfc.score(X_test,y_test))
print(classification_report(rfc_y_predict,y_test))
print('\n'*2)
print('The accuracy of gradient tree boosting',gbc.score(X_test,y_test))
print(classification_report(gbc_y_predict,y_test))

```


# xgboost
```
# 安裝xgboost
!pip install xgboost

# 下載資料集
!wget https://raw.githubusercontent.com/MyDearGreatTeacher/Data/master/iris.data
```

### 推薦文章
```
史上最詳細的XGBoost實戰
https://zhuanlan.zhihu.com/p/31182879
```
#### xgboost範例程式
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
#### xgboost範例程式
```
# Ieva Zarina, 2016, licensed under the Apache 2.0 licnese
# 資料來源 https://gist.github.com/IevaZarina/ef63197e089169a9ea9f3109058a9679

import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.datasets import dump_svmlight_file
from sklearn.externals import joblib
from sklearn.metrics import precision_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# use DMatrix for xgbosot
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# use svmlight file for xgboost
dump_svmlight_file(X_train, y_train, 'dtrain.svm', zero_based=True)
dump_svmlight_file(X_test, y_test, 'dtest.svm', zero_based=True)
dtrain_svm = xgb.DMatrix('dtrain.svm')
dtest_svm = xgb.DMatrix('dtest.svm')

# set xgboost params  使用key-values pair
param = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 3}  # the number of classes that exist in this datset

num_round = 20  # the number of training iterations

#------------- numpy array ------------------
# training and testing - numpy matrices
bst = xgb.train(param, dtrain, num_round)
preds = bst.predict(dtest)

# extracting most confident predictions
best_preds = np.asarray([np.argmax(line) for line in preds])
print "Numpy array precision:", precision_score(y_test, best_preds, average='macro')

# ------------- svm file ---------------------
# training and testing - svm file
bst_svm = xgb.train(param, dtrain_svm, num_round)
preds = bst.predict(dtest_svm)

# extracting most confident predictions
best_preds_svm = [np.argmax(line) for line in preds]
print "Svm file precision:",precision_score(y_test, best_preds_svm, average='macro')
# --------------------------------------------

# dump the models
bst.dump_model('dump.raw.txt')
bst_svm.dump_model('dump_svm.raw.txt')


# save the models for later
joblib.dump(bst, 'bst_model.pkl', compress=True)
joblib.dump(bst_svm, 'bst_svm_model.pkl', compress=True)
```

# LightGBM(2017)  Light Gradient Boosting Machine
```
https://github.com/Microsoft/LightGBM
https://lightgbm.readthedocs.io/en/latest/
https://lightgbm.apachecn.org/#/
```
```
Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. 
"LightGBM: A Highly Efficient Gradient Boosting Decision Tree". 
Advances in Neural Information Processing Systems 30 (NIPS 2017), pp. 3149-3157.

Qi Meng, Guolin Ke, Taifeng Wang, Wei Chen, Qiwei Ye, Zhi-Ming Ma, Tie-Yan Liu. 
"A Communication-Efficient Parallel Algorithm for Decision Tree". 
Advances in Neural Information Processing Systems 29 (NIPS 2016), pp. 1279-1287.

Huan Zhang, Si Si and Cho-Jui Hsieh. 
"GPU Acceleration for Large-scale Tree Boosting". 
SysML Conference, 2018.
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
