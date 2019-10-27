# 結構化資料分類實戰：心臟病預測(tensorflow2.0官方教程)
```

https://www.mashangxue123.com/tensorflow/tf2-tutorials-keras-feature_columns.html
```

```
本教程演示了如何對結構化資料進行分類（例如CSV格式的表格資料）。
我們將使用Keras定義模型，並使用特徵列作為橋樑，將CSV中的列映射到用於訓練模型的特性。
本教程包含完整的代碼：

使用Pandas載入CSV檔。 .
構建一個輸入管道，使用tf.data批次處理和洗牌行
從CSV中的列映射到用於訓練模型的特性。
使用Keras構建、訓練和評估模型。
```

### 1. 資料集
```
我們將使用克利夫蘭診所心臟病基金會提供的一個小資料集 。
CSV中有幾百行，每行描述一個患者，每列描述一個屬性。
我們將使用此資訊來預測患者是否患有心臟病，該疾病在該資料集中是二元分類任務。

以下是此資料集的說明。請注意，有數字和分類列。

Column	Description	Feature Type	Data Type
Age	Age in years	Numerical	integer
Sex	(1 = male; 0 = female)	Categorical	integer
CP	Chest pain type (0, 1, 2, 3, 4)	Categorical	integer
Trestbpd	Resting blood pressure (in mm Hg on admission to the hospital)	Numerical	integer
Chol	Serum cholestoral in mg/dl	Numerical	integer
FBS	(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)	Categorical	integer
RestECG	Resting electrocardiographic results (0, 1, 2)	Categorical	integer
Thalach	Maximum heart rate achieved	Numerical	integer
Exang	Exercise induced angina (1 = yes; 0 = no)	Categorical	integer
Oldpeak	ST depression induced by exercise relative to rest	Numerical	integer
Slope	The slope of the peak exercise ST segment	Numerical	float
CA	Number of major vessels (0-3) colored by flourosopy	Numerical	integer
Thal	3 = normal; 6 = fixed defect; 7 = reversable defect	Categorical	string
Target	Diagnosis of heart disease (1 = true; 0 = false)	Classification	integer

```
```
2. 導入TensorFlow和其他庫

安裝sklearn: pip install sklearn

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

3. 使用Pandas創建資料幀
Pandas 是一個Python庫，包含許多有用的實用程式，用於載入和處理結構化資料。
我們將使用Pandas從URL下載資料集，並將其載入到數據幀中。

URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
dataframe.head()


4. 將資料拆分為訓練、驗證和測試
我們下載的資料集是一個CSV檔，並將其分為訓練，驗證和測試集。

train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

5. 使用tf.data創建輸入管道
接下來，我們將使用tf.data包裝資料幀，這將使我們能夠使用特徵列作為橋樑從Pandas資料框中的列映射到用於訓練模型的特徵。
如果我們使用非常大的CSV檔（如此之大以至於它不適合記憶體），我們將使用tf.data直接從磁片讀取它，本教程不涉及這一點。

# 一種從Pandas Dataframe創建tf.data資料集的使用方法 
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

batch_size = 5 # 小批量用於演示目的
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

6. 理解輸入管道
現在我們已經創建了輸入管道，讓我們調用它來查看它返回的資料的格式，
我們使用了一小批量來保持輸出的可讀性。

for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['age'])
  print('A batch of targets:', label_batch )

我們可以看到資料集返回一個列名稱（來自資料幀），該清單映射到資料幀中行的列值。

7. 演示幾種類型的特徵列
TensorFlow提供了許多類型的特性列。
在本節中，我們將創建幾種類型的特性列，並演示它們如何從dataframe轉換列。

# 我們將使用此批次處理來演示幾種類型的特徵列 
example_batch = next(iter(train_ds))[0]

# 用於創建特徵列和轉換批量資料 
def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())


7.1. 數字列
特徵列的輸出成為模型的輸入（使用上面定義的演示函數，我們將能夠準確地看到資料幀中每列的轉換方式），
數位列是最簡單的列類型，它用於表示真正有價值的特徵，使用此列時，模型將從資料幀中接收未更改的列值。

age = feature_column.numeric_column("age")
demo(age)

在心臟病資料集中，資料幀中的大多數列都是數字。

7.2. Bucketized列（桶列）
通常，您不希望將數位直接輸入模型，而是根據數值範圍將其值分成不同的類別，考慮代表一個人年齡的原始資料，
我們可以使用bucketized列將年齡分成幾個桶，而不是將年齡表示為數字列。
請注意，下面的one-hot(獨熱編碼)值描述了每行匹配的年齡範圍。

age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets)


7.3. 分類列
在該資料集中，thal表示為字串（例如“固定”，“正常”或“可逆”），我們無法直接將字串提供給模型，
相反，我們必須首先將它們映射到數值。分類詞彙表列提供了一種將字串表示為獨熱向量的方法（就像上面用年齡段看到的那樣）。
詞彙表可以使用categorical_column_with_vocabulary_list作為列表傳遞，或者使用categorical_column_with_vocabulary_file從檔載入。

thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])

thal_one_hot = feature_column.indicator_column(thal)
demo(thal_one_hot)


在更複雜的資料集中，許多列將是分類的（例如字串），在處理分類資料時，特徵列最有價值。
雖然此資料集中只有一個分類列，但我們將使用它來演示在處理其他資料集時可以使用的幾種重要類型的特徵列。

7.4. 嵌入列
假設我們不是只有幾個可能的字串，而是每個類別有數千（或更多）值。
由於多種原因，隨著類別數量的增加，使用獨熱編碼訓練神經網路變得不可行，我們可以使用嵌入列來克服此限制。
嵌入列不是將資料表示為多維度的獨熱向量，而是將資料表示為低維密集向量，其中每個儲存格可以包含任意數位，
而不僅僅是0或1.嵌入的大小（在下面的例子中是8）是必須調整的參數。

關鍵點：當分類列具有許多可能的值時，最好使用嵌入列，
我們在這裡使用一個用於演示目的，因此您有一個完整的示例，您可以在將來修改其他資料集。

# 請注意，嵌入列的輸入是我們先前創建的分類列 
thal_embedding = feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)


7.5. 雜湊特徵列
表示具有大量值的分類列的另一種方法是使用categorical_column_with_hash_bucket.
此特徵列計算輸入的雜湊值，然後選擇一個hash_bucket_size存儲桶來編碼字串，
使用此列時，您不需要提供詞彙表，並且可以選擇使hash_buckets的數量遠遠小於實際類別的數量以節省空間。

關鍵點：該技術的一個重要缺點是可能存在衝突，其中不同的字串被映射到同一個桶，實際上，無論如何，這對某些資料集都有效。

thal_hashed = feature_column.categorical_column_with_hash_bucket(
      'thal', hash_bucket_size=1000)
demo(feature_column.indicator_column(thal_hashed))

7.6. 交叉特徵列
將特徵組合成單個特徵（也稱為特徵交叉），使模型能夠為每個特徵組合學習單獨的權重。
在這裡，我們將創建一個age和thal交叉的新功能，
請注意，crossed_column不會構建所有可能組合的完整表（可能非常大），相反，它由hashed_column支持，因此您可以選擇表的大小。

crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
demo(feature_column.indicator_column(crossed_feature))


8. 選擇要使用的列
我們已經瞭解了如何使用幾種類型的特徵列，現在我們將使用它們來訓練模型。
本教程的目標是向您展示使用特徵列所需的完整代碼（例如，機制），我們選擇了幾列來任意訓練我們的模型。

關鍵點：如果您的目標是建立一個準確的模型，請嘗試使用您自己的更大資料集，並仔細考慮哪些特徵最有意義，以及如何表示它們。

feature_columns = []

# numeric 數字列
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
  feature_columns.append(feature_column.numeric_column(header))

# bucketized 分桶列
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# indicator 指示符列 
thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding 嵌入列 
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed 交叉列 
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)


8.1. 創建特徵層
現在我們已經定義了我們的特徵列，我們將使用DenseFeatures層將它們輸入到我們的Keras模型中。

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

之前，我們使用小批量大小來演示特徵列的工作原理，我們創建了一個具有更大批量的新輸入管道。

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


9. 創建、編譯和訓練模型

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)


訓練過程的輸出

Epoch 1/5
7/7 [==============================] - 1s 79ms/step - loss: 3.8492 - accuracy: 0.4219 - val_loss: 2.7367 - val_accuracy: 0.7143
......
Epoch 5/5
7/7 [==============================] - 0s 34ms/step - loss: 0.6200 - accuracy: 0.7377 - val_loss: 0.6288 - val_accuracy: 0.6327



測試

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
1
2
      2/2 [==============================] - 0s 19ms/step - loss: 0.5538 - accuracy: 0.6721
      Accuracy 0.6721311
1
2
關鍵點：通常使用更大更複雜的資料集進行深度學習，您將看到最佳結果。
使用像這樣的小資料集時，我們建議使用決策樹或隨機森林作為強基線。

本教程的目標不是為了訓練一個準確的模型，而是為了演示使用結構化資料的機制，
因此您在將來使用自己的資料集時需要使用代碼作為起點。

```
