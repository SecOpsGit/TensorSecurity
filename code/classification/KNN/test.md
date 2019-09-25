#
```
# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets.samples_generator import make_blobs
# 生成資料
centers = [[-2, 2], [2, 2], [0, 4]]
X, y = make_blobs(n_samples=60, centers=centers, random_state=0, cluster_std=0.60)

# 畫出數據
plt.figure(figsize=(16, 10))
c = np.array(centers)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='cool');         # 畫出樣本
plt.scatter(c[:, 0], c[:, 1], s=100, marker='^', c='orange');   # 畫出中心點

from sklearn.neighbors import KNeighborsClassifier
# 模型訓練
k = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X, y);

# 進行預測
X_sample = [0, 2]
X_sample = np.array(X_sample).reshape(1, -1)
y_sample = clf.predict(X_sample);
neighbors = clf.kneighbors(X_sample, return_distance=False);

# 畫出示意圖
plt.figure(figsize=(16, 10))
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='cool')    # 樣本
plt.scatter(c[:, 0], c[:, 1], s=100, marker='^', c='k')   # 中心點
plt.scatter(X_sample[0][0], X_sample[0][1], marker="x", 
            s=100, cmap='cool')    # 待預測的點

for i in neighbors[0]:
    # 預測點與距離最近的 5 個樣本的連線
    plt.plot([X[i][0], X_sample[0][0]], [X[i][1], X_sample[0][1]], 
             'k--', linewidth=0.6);


```

```


```
