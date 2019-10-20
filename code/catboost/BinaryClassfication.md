# Binary Classfication
## Breast Cancer
```
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# 導入數據
data = load_breast_cancer()

print("data info: ")
print("data contents: ", data.keys()) 
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
print("features: ", data.feature_names)
print("targets: ", data.target_names)
print("data shape: ", data.data.shape)
print("target shape: ", data.target.shape)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.2)

print("\nTrain and Testing: ")
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

# build model
model = CatBoostClassifier(iterations=5, learning_rate=1, depth=10)

#train
print("\nTraining Model:")
model.fit(X_train, y_train)

# test result
result = model.predict(X_test)

# validation
print("\nTesting result: ")
print("Accuracy: ", accuracy_score(y_test, result))
print("Precision: ", precision_score(y_test, result, average='binary'))
print("Recall: ", recall_score(y_test, result, average='binary'))
print("F1_score: ", f1_score(y_test, result, average='binary'))
```
