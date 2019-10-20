###
```
import numpy as np
import catboost as cb
 
train_data = np.random.randint(0, 100, size=(100, 10))
train_label = np.random.randint(0, 2, size=(100))
test_data = np.random.randint(0,100, size=(50,10))
 
model = cb.CatBoostClassifier(iterations=2, depth=2, learning_rate=0.5,  
                                                      loss_function='Logloss',logging_level='Verbose')

model.fit(train_data, train_label, cat_features=[0,2,5])

preds_class = model.predict(test_data)
preds_probs = model.predict_proba(test_data)

print('class = ',preds_class)
print('proba = ',preds_probs)
```
```
0:	learn: 0.6705108	total: 54.7ms	remaining: 54.7ms
1:	learn: 0.6578008	total: 55.3ms	remaining: 0us
class =  [1. 1. 1. 0. 0. 0. 0. 1. 1. 0. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 0. 1. 1. 1.
 1. 1. 0. 1. 1. 0. 1. 0. 1. 1. 1. 0. 0. 0. 1. 1. 1. 0. 1. 1. 1. 1. 0. 1.
 1. 1.]
proba =  [[0.41640267 0.58359733]
 [0.478398   0.521602  ]
 [0.478398   0.521602  ]
 [0.5815617  0.4184383 ]
```
###
```

```
