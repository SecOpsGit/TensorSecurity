#
```
https://blog.csdn.net/mago2015/article/details/88390996

```
```
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of points: %d  " % (iris.data.shape[0]))
print("Number of mislabeled points: %d" % (iris.target != y_pred).sum())
```
