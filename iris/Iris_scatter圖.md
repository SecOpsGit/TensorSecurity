#

```
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris    #導入資料集iris
  
#載入資料集  
iris = load_iris()  
print(iris.data)          #輸出資料集  
print(iris.target)         #輸出真實標籤  
#獲取花卉兩列資料集  
DD = iris.data  
X = [x[0] for x in DD]  
print(X)  
Y = [x[1] for x in DD]  
print(Y)  
  
#plt.scatter(X, Y, c=iris.target, marker='x')
plt.scatter(X[:50], Y[:50], color='red', marker='o', label='setosa') #前50個樣本
plt.scatter(X[50:100], Y[50:100], color='blue', marker='x', label='versicolor') #中間50個
plt.scatter(X[100:], Y[100:],color='green', marker='+', label='Virginica') #後50個樣本
plt.legend(loc=2) #左上角
plt.show()



```
