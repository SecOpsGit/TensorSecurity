# Word Embedding
```
詞嵌入向量(Word Embedding)是NLP裡面一個重要的概念，

我們可以利用 WordEmbedding 

將一個單詞轉換成固定長度的向量表示，從而便於進行數學處理
```
```
https://www.mashangxue123.com/tensorflow/tf2-tutorials-text-word_embeddings.html
```

### NLP詞嵌入Word embedding實戰專案 (tensorflow2.0官方教程翻譯)
```
本文介紹詞嵌入向量 Word embedding，包含完整的代碼，可以在小型資料集上從零開始訓練詞嵌入，
並使用Embedding Projector 視覺化這些嵌入


1. 將文本表示為數位
機器學習模型以向量（數位陣列）作為輸入，
在處理文本時，我們必須首先想出一個策略，將字串轉換為數位（或將文本“向量化”），
然後再將其提供給模型。

在本節中，我們將研究三種策略。

1.1. 獨熱編碼（One-hot encodings）
首先，我們可以用“one-hot”對詞彙的每個單詞進行編碼，
想想“the cat sat on the mat”這句話，
這個句子中的詞彙（或獨特的單詞）是（cat,mat,on,The），
為了表示每個單詞，我們將創建一個長度等於詞彙表的零向量，
然後再對應單詞的索引中放置一個1。

為了創建包含句子編碼的向量，我們可以連接每個單詞的one-hot向量。

關鍵點：這種方法是低效的，一個熱編碼的向量是稀疏的（意思是，大多數指標是零）。
假設我們有10000個單詞，要對每個單詞進行一個熱編碼，我們將創建一個向量，其中99.99%的元素為零。

1.2. 用唯一的數位編碼每個單詞
我們嘗試第二種方法，使用唯一的數位編碼每個單詞。
繼續上面的例子，我們可以將1賦值給“cat”，將2賦值給“mat”，以此類推，
然後我們可以將句子“The cat sat on the mat”編碼為像[5, 1, 4, 3, 5, 2]這樣的密集向量。

這種方法很有效，我們現有有一個稠密的向量（所有元素都是滿的），而不是稀疏的向量。

然而，這種方法有兩個缺點：
[1]整數編碼是任意的（它不捕獲單詞之間的任何關係）。
[2]對於模型來說，整數編碼的解釋是很有挑戰性的。
    例如，線性分類器為每個特徵學習單個權重。
    由於任何兩個單詞的相似性與它們編碼的相似性之間沒有關係，所以這種特徵權重組合沒有意義。

1.3. 詞嵌入word embedding
詞嵌入為我們提供了一種使用高效、密集表示的方法，其中相似的單詞具有相似的編碼，
重要的是，我們不必手工指定這種編碼，嵌入是浮點值的密集向量（向量的長度是您指定的參數），
它們不是手工指定嵌入的值，而是可訓練的參數（模型在訓練期間學習的權重，與模型學習密集層的權重的方法相同）。
通常會看到8維（對於小資料集）的詞嵌入，在處理大型資料集時最多可達1024維。 
更高維度的嵌入可以捕獲單詞之間的細細微性關係，但需要更多的資料來學習。


上面是詞嵌入的圖表，每個單詞表示為浮點值的4維向量，
另一種考慮嵌入的方法是“查閱資料表”，在學習了這些權重之後，
我們可以通過查閱資料表中對應的密集向量來編碼每個單詞。
```
## 2. 利用 Embedding 層學習詞嵌入
```
Keras可以輕鬆使用詞嵌入。我們來看看 Embedding 層。

from __future__ import absolute_import, division, print_function, unicode_literals

# !pip install tf-nightly-2.0-preview
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# Embedding層至少需要兩個參數： 
# 詞彙表中可能的單詞數量，這裡是1000（1+最大單詞索引）； 
# embeddings的維數，這裡是32.。
embedding_layer = layers.Embedding(1000, 32)


Embedding層可以理解為一個查詢表，
它從整數索引（表示特定的單詞）映射到密集向量（它們的嵌入）。
嵌入的維數（或寬度）是一個參數，您可以用它進行試驗，看看什麼對您的問題有效，
這與您在一個密集層中對神經元數量進行試驗的方法非常相似。

創建Embedding層時，嵌入的權重會隨機初始化（就像任何其他層一樣），
在訓練期間，它們通過反向傳播逐漸調整，
一旦經過訓練，學習的詞嵌入將粗略地編碼單詞之間的相似性（因為它們是針對您的模型所訓練的特定問題而學習的）。

作為輸入，Embedding層採用形狀(samples, sequence_length)的整數2D張量，
其中每個條目都是整數序列，它可以嵌入可以變長度的序列。
您可以使用形狀(32, 10) （批次為32個長度為10的序列）或(64, 15) （批次為64個長度為15的序列）
導入上述批次的嵌入層，批次處理中的序列必須具有相同的長度，因此較短的序列應該用零填充，較長的序列應該被截斷。

作為輸出，Embedding層返回一個形狀(samples, sequence_length, embedding_dimensionality)的三維浮點張量，
這樣一個三維張量可以由一個RNN層來處理，
也可以簡單地由一個扁平化或合併的密集層處理。
我們將在本教程中展示第一種方法，您可以參考使用RNN的文本分類來學習第一種方法。
```
### 3. 從頭開始學習嵌入
```
我們將在 IMDB 影評上訓練一個情感分類器，
在這個過程中，我們將從頭開始學習嵌入，通過下載和預處理資料集的代碼快速開始(請參閱本教程tutorial瞭解更多細節)。

vocab_size = 10000
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

print(train_data[0])


      [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, ...]

導入時，評論文本是整數編碼的（每個整數代表字典中的特定單詞）。

3.1. 將整數轉換會單詞
瞭解如何將整數轉換回文本可能很有用，在這裡我們將創建一個輔助函數來查詢包含整數到字串映射的字典物件：

# 將單詞映射到整數索引的字典
word_index = imdb.get_word_index()

# 第一個指數是保留的
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

decode_review(train_data[0])


電影評論可以有不同的長度==>使用pad_sequences函數來標準化評論的長度：

maxlen = 500

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=maxlen)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=maxlen)

print(train_data[0])    # 檢查填充資料的第一個元素：

 
3.2. 創建一個簡單的模型
使用 Keras Sequential API來定義我們的模型。

第一層是Embedding層。
該層採用整數編碼的詞彙表，並查找每個詞索引的嵌入向量，
這些向量是作為模型訓練學習的，向量為輸出陣列添加維度，
得到的維度是:(batch, sequence, embedding)。

接下來，GlobalAveragePooling1D層通過對序列維度求平均，
為每個示例返回固定長度的輸出向量，這允許模型以盡可能最簡單的方式處理可變長度的輸入。

該固定長度輸出向量通過具有16個隱藏單元的完全連接（Dense）層進行管道傳輸。

最後一層與單個輸出節點密集連接，使用sigmoid啟動函數，此值是介於0和1之間的浮點值，表示評論為正的概率（或置信度）。

embedding_dim=16

model = keras.Sequential([
  layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
  layers.GlobalAveragePooling1D(),
  layers.Dense(16, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.summary()
1

      Model: "sequential"
      _________________________________________________________________
      Layer (type)                 Output Shape              Param #   
      =================================================================
      embedding_1 (Embedding)      (None, 500, 16)           160000    
      _________________________________________________________________
      global_average_pooling1d (Gl (None, 16)                0         
      _________________________________________________________________
      dense (Dense)                (None, 16)                272       
      _________________________________________________________________
      dense_1 (Dense)              (None, 1)                 17        
      =================================================================
      Total params: 160,289
      Trainable params: 160,289
      Non-trainable params: 0
      _________________________________________________________________


3.3. 編譯和訓練模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_data,
    train_labels,
    epochs=30,
    batch_size=512,
    validation_split=0.2)


      Train on 20000 samples, validate on 5000 samples
      ...
      Epoch 30/30
      20000/20000 [==============================] - 1s 54us/sample - loss: 0.1639 - accuracy: 0.9449 - val_loss: 0.2840 - val_accuracy: 0.8912


通過這種方法，我們的模型達到了大約88%的驗證精度（注意模型過度擬合，訓練精度顯著提高）。
```
### 
```
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12,9))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0.5,1))

plt.show()

```
### 4. 檢索學習的嵌入
```
接下來，讓我們檢索在訓練期間學習的嵌入詞，
這將是一個形狀矩陣 (vocab_size,embedding-dimension)。

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

我們現在將權重寫入磁片。
要使用Embedding Projector，
我們將以定位字元分隔格式上傳兩個檔：向量檔（包含嵌入）和中繼資料檔（包含單詞）。

import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

```
```
如果您在Colaboratory中運行本教程，則可以使用以下程式碼片段將這些檔下載到本地電腦
（或使用檔流覽器， View -> Table of contents -> File browser）。

try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')
```

### 5. 視覺化嵌入
```
為了視覺化我們的嵌入，我們將把它們上傳到Embedding Projector。

打開Embedding Projector：

點擊“Load data”

上傳我們上面創建的兩個檔：vecs.tsv和meta.tsv。

現在將顯示您已訓練的嵌入，您可以搜索單詞以查找最近的鄰居。
例如，嘗試搜索“beautiful”，你可能會看到像“wonderful”這樣的鄰居。
注意：您的結果可能有點不同，這取決於在訓練嵌入層之前如何隨機初始化權重。

注意：通過實驗，你可以使用更簡單的模型生成更多可解釋的嵌入，
嘗試刪除Dense（16）層，重新訓練模型，再次視覺化嵌入。

```
