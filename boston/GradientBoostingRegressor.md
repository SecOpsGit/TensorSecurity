#

```
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error

# 載入sklearn自帶的波士頓房價資料集
dataset = load_boston()

# 提取特徵資料和目標資料
X = dataset.data
y = dataset.target

# 將資料集以9:1的比例隨機分為訓練集和測試集，為了重現隨機分配設置隨機種子，即random_state參數
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=188)

# 產生實體估計器對象
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
gbr = ensemble.GradientBoostingRegressor(**params)

# 估計器擬合訓練資料
gbr.fit(X_train, y_train)

# 訓練完的估計器對測試資料進行預測
y_pred = gbr.predict(X_test)

# 輸出特徵重要性清單
print(gbr.feature_importances_)
print(mean_squared_error(y_test, y_pred))
```

```
[2.13268710e-02 2.35812664e-04 3.37362083e-03 1.40848911e-04
 2.66890762e-02 4.32149265e-01 7.35098942e-03 7.74935980e-02
 2.00230548e-03 1.42056105e-02 3.00671498e-02 1.13987274e-02
 3.73566125e-01]
8.154517490169884

```
