
## diabetes

```
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score


# 導入數據
data = load_diabetes()

print("data info: ")
print("data contents: ", data.keys()) # dict_keys(['data', 'target', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
print("features: ", data.feature_names)
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
model = CatBoostRegressor(iterations=5, learning_rate=1, depth=10)

#train
print("\nTraining Model:")
model.fit(X_train, y_train)

# test result
result = model.predict(X_test)

# validation
print("\nTesting result: ")
print("Mean_absolute_error (raw_values): ", mean_absolute_error(y_test, result, multioutput='raw_values'))
print("Mean_absolute_error (uniform_average): ", mean_absolute_error(y_test, result, multioutput='uniform_average'))
print("mean_squared_error (raw_values): ", mean_squared_error(y_test, result, multioutput='raw_values'))
print("mean_squared_error (uniform_average): ", mean_squared_error(y_test, result, multioutput='uniform_average'))
print("median_absolute_error: ", median_absolute_error(y_test, result))
print("r2_score (raw_values): ", r2_score(y_test, result, multioutput='raw_values'))
print("r2_score (uniform_average): ", r2_score(y_test, result, multioutput='uniform_average'))
print("r2_score (variance_weighted): ", r2_score(y_test, result, multioutput='variance_weighted'))
```

## Boston
```
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score


# 導入數據
data = load_boston()

print("data info: ")
print("data contents: ", data.keys()) # dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])
print("features: ", data.feature_names)
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
model = CatBoostRegressor(iterations=5, learning_rate=1, depth=10)

#train
print("\nTraining Model:")
model.fit(X_train, y_train)

# test result
result = model.predict(X_test)

# validation
print("\nTesting result: ")
print("Mean_absolute_error (raw_values): ", mean_absolute_error(y_test, result, multioutput='raw_values'))
print("Mean_absolute_error (uniform_average): ", mean_absolute_error(y_test, result, multioutput='uniform_average'))
print("mean_squared_error (raw_values): ", mean_squared_error(y_test, result, multioutput='raw_values'))
print("mean_squared_error (uniform_average): ", mean_squared_error(y_test, result, multioutput='uniform_average'))
print("median_absolute_error: ", median_absolute_error(y_test, result))
print("r2_score (raw_values): ", r2_score(y_test, result, multioutput='raw_values'))
print("r2_score (uniform_average): ", r2_score(y_test, result, multioutput='uniform_average'))
print("r2_score (variance_weighted): ", r2_score(y_test, result, multioutput='variance_weighted'))
```
