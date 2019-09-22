#
```


```

```
from sklearn import svm  
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib  
import sklearn  
from sklearn.model_selection import train_test_split  
  
#define converts(字典)  
def Iris_label(s):  
    it={b'Iris-setosa':0, b'Iris-versicolor':1, b'Iris-virginica':2 }  
    return it[s]  
  
  
#1.讀取資料集  
path='iris.data'  
data=np.loadtxt(path, dtype=float, delimiter=',', converters={4:Iris_label} )  
#converters={4:Iris_label}中“4”指的是第5列：將第5列的str轉化為label(number)  
#print(data.shape)  
  
#2.劃分資料與標籤  
x,y=np.split(data,indices_or_sections=(4,),axis=1) #x為數據，y為標籤  
x=x[:,0:2]  
train_data,test_data,train_label,test_label =train_test_split(x,y, random_state=1, train_size=0.6,test_size=0.4) #sklearn.model_selection.  
#print(train_data.shape)  
  
#3.訓練svm分類器  
classifier=svm.SVC(C=2,kernel='rbf',gamma=10,decision_function_shape='ovo') # ovr:一對多策略  
classifier.fit(train_data,train_label.ravel()) #ravel函數在降維時預設是行序優先  
  
#4.計算svc分類器的準確率  
print("訓練集：",classifier.score(train_data,train_label))  
print("測試集：",classifier.score(test_data,test_label))  
  
#也可直接調用accuracy_score方法計算準確率  
from sklearn.metrics import accuracy_score  
tra_label=classifier.predict(train_data) #訓練集的預測標籤  
tes_label=classifier.predict(test_data) #測試集的預測標籤  
print("訓練集：", accuracy_score(train_label,tra_label) )  
print("測試集：", accuracy_score(test_label,tes_label) )  
  
#查看決策函數  
print('train_decision_function:\n',classifier.decision_function(train_data)) # (90,3)  
print('predict_result:\n',classifier.predict(train_data))  
  
#5.繪製圖形  
#確定坐標軸範圍  
x1_min, x1_max=x[:,0].min(), x[:,0].max() #第0維特徵的範圍  
x2_min, x2_max=x[:,1].min(), x[:,1].max() #第1維特徵的範圍  
x1,x2=np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j ] #生成網路採樣點  
grid_test=np.stack((x1.flat,x2.flat) ,axis=1) #測試點  
#指定預設字體  
matplotlib.rcParams['font.sans-serif']=['SimHei']  
#設置顏色  
cm_light=matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])  
cm_dark=matplotlib.colors.ListedColormap(['g','r','b'] )  
  
grid_hat = classifier.predict(grid_test)       # 預測分類值  
grid_hat = grid_hat.reshape(x1.shape)  # 使之與輸入的形狀相同  
  
plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)     # 預測值的顯示  
plt.scatter(x[:, 0], x[:, 1], c=y[:,0], s=30,cmap=cm_dark)  # 樣本  
plt.scatter(test_data[:,0],test_data[:,1], c=test_label[:,0],s=30,edgecolors='k', zorder=2,cmap=cm_dark) #圈中測試集樣本點  
plt.xlabel('花萼長度', fontsize=13)  
plt.ylabel('花萼寬度', fontsize=13)  
plt.xlim(x1_min,x1_max)  
plt.ylim(x2_min,x2_max)  
plt.title('鳶尾花SVM二特徵分類')  
plt.show()  
```
