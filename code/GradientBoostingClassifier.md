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
