# LinearRegression()
```
# -*- coding: utf-8 -*-

#第一步 資料集劃分
from sklearn import datasets
import numpy as np
 
#獲取資料 10*442
d = datasets.load_diabetes()
x = d.data
print(u'獲取x特徵')
print(len(x), x.shape)
print(x[:4])
 
#獲取一個特徵 第3列資料
x_one = x[:,np.newaxis, 2]
print( x_one[:4])
 
#獲取的正確結果
y = d.target
print(u'獲取的結果')
print(y[:4])
 
#x特徵劃分
x_train = x_one[:-42]
x_test = x_one[-42:]
print(len(x_train), len(x_test))
y_train = y[:-42]
y_test = y[-42:]
print(len(y_train), len(y_test))
 
 
#第二步 線性回歸實現
from sklearn import linear_model
clf = linear_model.LinearRegression()
print(clf)
clf.fit(x_train, y_train)
pre = clf.predict(x_test)
print(u'預測結果')
print(pre)
print(u'真實結果')
print(y_test)  
   
   
#第三步 評價結果
cost = np.mean(y_test-pre)**2
print(u'次方', 2**5)
print(u'平方和計算:', cost)
print(u'係數', clf.coef_)
print(u'截距', clf.intercept_ ) 
print(u'方差', clf.score(x_test, y_test))
 
 
#第四步 繪圖
import matplotlib.pyplot as plt
plt.title("diabetes")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x_test, y_test, 'k.')
plt.plot(x_test, pre, 'g-')
 
for idx, m in enumerate(x_test):
    plt.plot([m, m],[y_test[idx], 
              pre[idx]], 'r-')
 
plt.savefig('power.png', dpi=300)
 
plt.show()
```
