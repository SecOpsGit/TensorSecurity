# gbdt回歸
```
Gradient boosting(GB)

Gradient boosting Decision Tree(GBDT)

GBDT幾乎可用於所有迴歸問題（線性/非線性），
相對logistic regression僅能用於線性迴歸，
GBDT的適用面非常廣。亦可用於二分類問題（設定閾值，大於閾值為正例，反之為負例）。
```

```
https://www.itread01.com/content/1545396730.html
```
# sklearn的gbdt回歸
```
sklearn中的gbdt實現是GradientBoostingRegressor類別。
GradientBoostingRegressor是一個基於向前分佈演算法的加法模型。
每一輪反覆運算，它都用損失函數的負梯度來擬合本輪損失的近似值。

```
## GradientBoostingRegressor類別構造方法(constructor)
```
def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None, random_state=None,
                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, presort='auto')
```

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
# 參數說明

```
loss：表示損失函數，可選項為{'ls', 'lad', 'huber', 'quantile'}，預設是'ls'；

'ls'（least squares）指的是最小二乘法（即最小化均方差），適合資料噪點不多的情況下，因為對於異常點會施以較大的懲罰；
'lad'（least absolute deviation）指的是最小化絕對偏差，如果有較多異常點，則絕對值損失表現較好，
       但絕對值損失的缺點是在y−f(x)=0處不連續可導，因而不容易優化；
'huber'是對'ls'和'lad'的綜合，當|y−f(x)|小於一個事先指定的值δ時，變為平方損失，
         大於δ時，則變成類似於絕對值損失，因此也是比較魯棒（健壯）的損失函數；
'quantile'指的是分位元數損失函數，它允許分位元數回歸（使用alpha來指定分位數）；

learning_rate：學習率（也稱之為step，即步長），通俗一點講就是基學習器權重縮減的係數，對於同樣的訓練集擬合效果，較小的步長意味著我們需要更多的反覆運算次數，通常我們用步長和反覆運算最大次數一起來決定演算法的擬合效果。所以這兩個參數n_estimators和learning_rate要一起調參。默認是0.1；

n_estimators：反覆運算的次數；GB對於過擬合的資料有很強的魯棒性，所以較大的反覆運算次數會有很好的效果，默認是100；

criterion：用來設置回歸樹的切分策略，可選項為{'friedman_mse','mse','mae'}，默認是'friedman_mse'；'friedman_mse'對應的是最小平方誤差的近似，加入了Friedman的一些改進；'mse'指的是最小均方差；'mae'對應的是最小絕對值誤差，0.18版本及之後引進的；

alpha：當loss選擇為'huber'或'quantile'時，對應的alpha值，只有這兩種值時，alpha才有意義，如果噪音點較多，可以適當降低這個分位數的值，默認是0.9；

subsample：子取樣速率，如果小於1.0，則會導致隨機梯度增強（Stochastic Gradient Boosting）。 子樣本與參數n_estimators一起調整使用。 選擇子樣本<1.0會導致方差減少和偏差增加。推薦在[0.5, 0.8]之間，默認是1.0，即不使用子採樣，訓練每個基學習器時使用原始的全部資料集；

max_depth：基回歸估計器的最大深度，調整好該參數可以收到更好的學習效果，默認是3；

min_samples_split：基回歸樹在分裂時的最小樣本數或占比，默認是2；當傳值為整數時，表示分裂最小樣本數；當傳值為浮點數時，表示分裂最小樣本數是ceil(min_samples_split * n_samples)，浮點數的用法是0.18版本時增加的；

min_samples_leaf：基回歸樹葉子節點上的最小樣本數，預設是1；當傳值為整數時，表示葉子最小樣本數；當傳值為浮點數時，表示葉子最小樣本數是ceil(min_samples_split * n_samples)，浮點數的用法是0.18版本時增加的；

min_weight_fraction_leaf：葉子節點最小樣本權重總和值，即葉子節點所有樣本的權重總和值小於該閾值，那就選擇樹的剪枝（即不分裂）；如果樣本本身不存在樣本權重，該參數無任何意義，默認是0；

max_features：在選擇分裂特徵時的候選特徵的最大數，可以是int,、float、string或None，默認是None；如果是int型，表示候選特徵的最大數；如果是float型，代表最大數是所占比，即閾值是int(max_features * n_features)；如果是"auto"，最大數就等於總特徵數n_features；如果是"sqrt"，則max_features=sqrt(n_features)；如果是"log2"，則max_features=log2(n_features)；如果是None，則max_features=n_features；當選擇`max_features < n_features`時，會導致方差減少和偏差增加；

max_leaf_nodes：最大的葉子節點數，默認是None；

min_impurity_split：樹分裂時的最小不純度閾值，小於該閾值，樹不分裂，即為葉子節點；默認是None，該參數在0.19版本之後拋棄使用，0.19版本暫時保留，最好選用min_impurity_decrease；

min_impurity_decrease：最小不純度的閾值，用於代替之前的min_impurity_split參數，閾值公式為N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)，N_t表示當前節點的樣本數，N表示總樣本數，N_t_R表示右子節點的樣本數，N_t_L表示左子節點的樣本數；

init：用來計算初始基學習器的預測，需要具備fit和predict方法，若未設置則默認為loss.init_estimator，默認是None；

verbose：日誌跟蹤級別，預設是0；如果傳值為1，根據情況（樹越多，越低頻率）列印訓練過程和訓練表現；如果傳值大於1，列印每棵樹的訓練過程和訓練表現；

warm_start：是否暖開機，bool類型，默認是False；若設置為True，允許添加更多的估計器到已經適合的模型；

random_state：隨機狀態，可以是整型數位，也可以是RandomState實例，預設是None；主要是通過隨機種子，重現訓練過程，None表示不設置隨機種子，底層使用numpy的random函數；

presort：是否預排序，可以是bool值和'auto'，默認auto；預排序可以加速查找最佳分裂點，當設置為auto時，對非稀疏資料進行預排序，對稀疏資料不起作用；對稀疏數據設置True的話，會引起error；預排序是0.17版本新加的；
最終估計器物件的相關屬性

feature_importances_：每個特徵的重要性，是一個浮點數陣列，每項相加總和為1；數值越高說明該特徵越重要；

oob_improvement_：使用包外樣本來計算每一輪訓練後模型的表現提升，它是一個陣列，外包估算可用於模型選擇，例如確定最佳反覆運算次數。OOB估計通常非常悲觀，因此我們建議使用交叉驗證，而只有在交叉驗證太費時的情況下才使用OOB；

train_score_：訓練得分，其實就是每棵訓練樹的誤差（損失函數對應的值）；

loss_：具體的損失函數物件；

init：設置init參數時的對象；

estimators_：基回歸樹集合，是一個陣列；
```
