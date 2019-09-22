#
```

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import classification_report

# 載入sklearn自帶的iris（鳶尾花）資料集
iris = load_iris()

# 提取特徵資料和目標資料
X = iris.data
y = iris.target

# 將資料集以9:1的比例隨機分為訓練集和測試集，為了重現隨機分配設置隨機種子，即random_state參數
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=188)

# 產生實體分類器對象
clf = ensemble.AdaBoostClassifier()

# 分類器擬合訓練資料
clf.fit(X_train, y_train)

# 訓練完的分類器對測試資料進行預測
y_pred = clf.predict(X_test)

# classification_report函數用於顯示主要分類指標的文本報告
print(classification_report(y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica']))

```
