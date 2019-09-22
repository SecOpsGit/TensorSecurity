# 二元分類問題binary classification

# Pima印第安人糖尿病資料集
```
https://www.kaggle.com/saurabh00007/diabetescsv
https://blog.csdn.net/yizheyouye/article/details/79791473

該資料集最初來自國家糖尿病/消化/腎臟疾病研究所。

資料集的目標是基於資料集中包含的某些診斷測量來診斷性的預測 患者是否患有糖尿病。

從較大的資料庫中選擇這些實例有幾個約束條件。
這裡的所有患者都是Pima印第安至少21歲的女性。
資料集由多個醫學預測變數和一個目標變數組成Outcome。
預測變數包括患者的懷孕次數、BMI、胰島素水準、年齡等。
```
## 預測變數
```
【1】Pregnancies：懷孕次數
【2】Glucose：葡萄糖
【3】BloodPressure：血壓 (mm Hg)
【4】SkinThickness：皮層厚度 (mm)
【5】Insulin：胰島素 2小時血清胰島素（mu U / ml
【6】BMI：體重指數 （體重/身高）^2
【7】DiabetesPedigreeFunction：糖尿病譜系功能
【8】Age：年齡 （歲）
【9】Outcome：類標變數 （0或1）

```

# 宮頸癌預測
```
risk_factors_cervical_cancer.csv
https://github.com/TheFloatingString/Project-Heal/blob/master/risk_factors_cervical_cancer.csv

```
```
test_05_隨機森林案例一：宮頸癌預測.py
https://github.com/Wasim37/machine_learning_code/blob/master/03%20%E5%86%B3%E7%AD%96%E6%A0%91%20Decision%20Tree/test_05_%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97%E6%A1%88%E4%BE%8B%E4%B8%80%EF%BC%9A%E5%AE%AB%E9%A2%88%E7%99%8C%E9%A2%84%E6%B5%8B.py
```

# Breast Cancer Wisconsin (Diagnostic) Data Set
```
Predict whether the cancer is benign or malignant

https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
```


```
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
data.target[[10, 50, 85]]
list(data.target_names)
```

# Lung Cancer Data Set
```
https://archive.ics.uci.edu/ml/datasets/lung+cancer
```
