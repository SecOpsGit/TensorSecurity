# TensorFlow與捲積神經網絡從算法入門到項目實戰 

華超
 
電子工業出版 2019-09-01
```
第一篇  TensorFlow基礎篇
第1章  緒論    2
1.1  人工智能簡介    2
1.2  卷積神經網絡    3
1.3  搭建TensorFlow框架環境    5
1.3.1  安裝Anaconda    5
1.3.2  安裝TensorFlow    7
第2章  TensorFlow基礎入門    9
2.1  第一個TensorFlow程序    9
2.1.1  TensorFlow中的hello world    9
2.1.2  TensorFlow中的圖    11
2.1.3  靜態圖與動態圖    14
2.2  初識Session    15
2.2.1  將Session對象關聯Graph對象    15
2.2.2  Session參數配置    17
2.3  常量與變量    18
2.3.1  TensorFlow中的常量    18
2.3.2  TensorFlow中的變量    20
2.3.3  TensorFlow中的tf.placeholder    28
2.4  Tensor對象    29
2.4.1  什麽是Tensor對象    29
2.4.2  Python對象轉Tensor對象    31
2.4.3  Tensor對象轉Python對象    32
2.4.4  SparseTensor對象    34
2.4.5  強制轉換Tensor對象數據類型    35
2.5  Operation對象    37
2.5.1  什麽是Operation對象    37
2.5.2  獲取並執行Operation對象    37
2.6  TensorFlow流程控制    40
2.6.1  條件判斷tf.cond與tf.where    40
2.6.2  TensorFlow比較判斷    43
2.6.3  TensorFlow邏輯運算    44
2.6.4  循環tf.while_loop    45
2.7  TensorFlow位運算    48
2.7.1  且位運算    48
2.7.2  或位運算    49
2.7.3  異或位運算    50
2.7.4  取反位運算    51
2.8  TensorFlow字符串    52
2.8.1  字符串的定義與轉換    53
2.8.2  字符串拆分    55
2.8.3  字符串拼接    56

第3章  高維Tensor對象的工具函數    58
3.1  重定義Shape    58
3.1.1  Reshape原理    58
3.1.2  函數tf.reshape    59
3.1.3  使用Python實現Reshape    60
3.2  維度交換函數    62
3.2.1  Transpose原理    62
3.2.2  函數tf.transpose    63
3.2.3  使用Python實現Transpose    64
3.3  維度擴充與消除    65
3.3.1  函數tf.expand_dims    65
3.3.2  函數tf.squeeze    66
3.4  Tensor對象裁剪    68
3.4.1  Tensor對象裁剪原理    68
3.4.2  函數tf.slice    69
3.5  Tensor對象拼接    70
3.5.1  Tensor對象拼接原理    70
3.5.2  函數tf.concat使用    71
3.6  tf.stack與tf.unstack    72
3.6.1  函數tf.stack的原理    72
3.6.2  函數tf.stack的使用    73
3.6.3  函數tf.unstack的使用    76
3.7  tf.argmax與tf.argmin    79
3.7.1  函數tf.argmax與tf.argmin的原理    79
3.7.2  函數tf.argmax與tf.argmin的使用    79



第二篇  卷積神經網絡篇
第4章  前饋網絡    83
4.1  卷積    83
4.1.1  卷積的原理    83
4.1.2  輸出寬高與輸入、Stride、卷積核及Padding之間的關系    90
4.1.3  空洞卷積    92
4.1.4  在TensorFlow中使用卷積    93
4.1.5  用Python語言實現卷積算法    95
4.2  反卷積    97
4.2.1  反卷積的原理    97
4.2.2  輸出寬高與輸入、Stride、反卷積核及Padding之間的關系    103
4.2.3  在TensorFlow中使用反卷積    105
4.2.4  用Python語言實現反卷積算法    110
4.3  Batch Normalization    113
4.3.1  Batch Normalization的原理    113
4.3.2  在TensorFlow中使用Batch Normalization    114
4.3.3  用Python語言實現Batch Normalization    122
4.3.4  在TensorFlow中使用Batch Normalization時的註意事項    123
4.4  Instance Normalization    125
4.4.1  Instance Normalization的原理    125
4.4.2  在TensorFlow中使用Instance Normalization    126
4.4.3  用Python語言實現Instance Normalization    130
4.5  全連接層    132
4.5.1  全連接層的原理    132
4.5.2  在TensorFlow中使用全連接層    133
4.5.3  用Python語言實現全連接層    134
4.6  激活函數    135
4.6.1  激活函數的作用    135
4.6.2  Sigmoid函數    136
4.6.3  Tanh函數    138
4.6.4  ReLU函數    140
4.7  池化層    142
4.7.1  池化層的原理    142
4.7.2  在TensorFlow中使用池化層    146
4.7.3  用Python語言實現池化層    150
4.8  Dropout    153
4.8.1  Dropout的作用    153
4.8.2  在TensorFlow中使用Dropout    154


第5章  常見網絡    156
5.1  移動端定制卷積神經網絡——MobileNet    156
5.1.1  MobileNet的原理與優勢    156
5.1.2  在TensorFlow中實現MobileNet卷積    158
5.1.3  用Python語言實現Depthwise卷積    164
5.1.4  MobileNet完整的網絡結構    167
5.1.5  MobileNet V2進一步裁剪加速    168
5.2  深度殘差網絡——ResNet    171
5.2.1  ResNet的結構與優勢    171
5.2.2  在TensorFlow中實現ResNet    172
5.2.3  完整的ResNet網絡結構    175
5.3  DenseNet    176
5.3.1  DenseNet的結構與優勢    176
5.3.2  在TensorFlow中實現DenseNet    177
5.3.3  完整的DenseNet網絡結構    180


第三篇  TensorFlow進階篇
第6章  TensorFlow數據存取    183
6.1  隊列    183
6.1.1  構建隊列    183
6.1.2  Queue、QueueRunner及Coordinator    190
6.1.3  在隊列中批量讀取數據    194
6.2  文件存取    200
6.2.1  讀取文本文件    200
6.2.2  讀取定長字節文件    202
6.2.3  讀取圖片    205
6.3  從CSV文件中讀取訓練集    207
6.3.1  解析CSV格式文件    207
6.3.2  封裝CSV文件讀取類    209
6.4  從自定義文本格式文件中讀取訓練集    210
6.4.1  解析自定義文本格式文件    211
6.4.2  封裝自定義文本格式文件讀取類    212
6.5  TFRecord方式存取數據    213
6.5.1  將數據寫入TFRecord文件    214
6.5.2  從TFRecord文件中讀取數據    215
6.6  模型存取    217
6.6.1  存儲模型    217
6.6.2  從checkpoint文件中加載模型    220
6.6.3  從meta文件中加載模型    222
6.6.4  將模型導出為單個pb文件    223


第7章  TensorFlow數據預處理    226
7.1  隨機光照變化    226
7.1.1  隨機飽和度變化    226
7.1.2  隨機色相變化    228
7.1.3  隨機對比度變化    230
7.1.4  隨機亮度變化    232
7.1.5  隨機伽瑪變化    234
7.2  翻轉、轉置與旋轉    237
7.2.1  隨機上下、左右翻轉    237
7.2.2  隨機圖像轉置    239
7.2.3  隨機旋轉    241
7.3  裁剪與Resize    245
7.3.1  圖像裁剪    245
7.3.2  圖像Resize    249
7.3.3  其他Resize函數    254
7.4  用OpenCV對圖像進行動態預處理    256
7.4.1  靜態預處理與動態預處理    256
7.4.2  在TensorFlow中調用OpenCV    257


第8章  TensorFlow模型訓練    260
8.1  反向傳播中的優化器與學習率    260
8.1.1  Global Step與Epoch    260
8.1.2  梯度理論    260
8.1.3  使用學習率與梯度下降法求最優值    262
8.1.4  TensorFlow中的優化器    265
8.1.5  優化器中常用的函數    265
8.1.6  在TensorFlow中動態調整學習率    269
8.2  模型數據與參數名稱映射    273
8.2.1  通過名稱映射加載    273
8.2.2  以pickle文件為中介加載模型    275
8.3  凍結指定參數    277
8.3.1  從模型中加載部分參數    277
8.3.2  指定網絡層參數不參與更新    278
8.3.3  兩個學習率同時訓練    280
8.4  TensorFlow中的命名空間    282
8.4.1  使用tf.variable_scope添加名稱前綴    282
8.4.2  使用tf.name_scope添加名稱前綴    284
8.4.3  tf.variable_scope與tf.name_scope的混合使用    285
8.5  TensorFlow多GPU訓練    286
8.5.1  多GPU訓練讀取數據    286
8.5.2  平均梯度與參數更新    289

第9章  TensorBoard可視化工具    293
9.1  可視化靜態圖    293
9.1.1  圖結構系列化並寫入文件    293
9.1.2  啟動TensorBoard    294
9.2  圖像顯示    296
9.2.1  系列化圖像Tensor並寫入文件    296
9.2.2  用TensorBoard查看圖像    299
9.3  標量曲線    301
9.3.1  系列化標量Tensor並寫入文件    301
9.3.2  用TensorBoard查看標量曲線    302
9.4  參數直方圖    303
9.4.1  系列化參數Tensor並寫入文件    303
9.4.2  用TensorBoard查看參數直方圖    304
9.5  文本顯示    306
9.5.1  系列化文本Tensor並寫入文件    306
9.5.2  用TensorBoard查看文本    307


第四篇  卷積神經網絡實戰篇

第10章  中文手寫字識別    310
10.1  網絡結構及數據集    310
10.1.1  網絡結構    310
10.1.2  數據集    311
10.2  代碼實現    312
10.2.1  封裝通用網絡層    312
10.2.2  定義網絡結構    314
10.2.3  數據讀取    316
10.2.4  訓練代碼實現    318
10.3  模型訓練    321
10.4  模型精度測試    321
10.4.1  精度測試    322
10.4.2  代碼實現    322

第11章  移植模型到TensorFlow Serving端    324
11.1  模型轉換    324
11.1.1  轉換模型為TensorFlow Serving模型    324
11.1.2  代碼實現    327
11.2  模型部署    329
11.2.1  搭建TensorFlow Serving環境    329
11.2.2  啟動TensorFlow Serving服務    331
11.3  HTTP服務實現    333
11.3.1  使用gRPC調用TensorFlow Serving服務    333
11.3.2  實現HTTP服務    334
11.4  前端交互實現    336
11.4.1  界面布局    336
11.4.2  手寫板實現    337
11.4.3  數據交互    339
11.4.4  流程測試    340

第12章  移植TensorFlow模型到Android端    341
12.1  交互界面    341
12.1.1  頁面布局    341
12.1.2  實現手寫板    342
12.2  使用TensorFlow Mobile庫    346
12.2.1  模型轉換    347
12.2.2  模型調用    347
12.2.3  模型測試    351
12.3  使用TensorFlow Lite庫    354
12.3.1  模型轉換    354
12.3.2  模型調用    355
12.3.3  模型測試    360

第13章  移植TensorFlow模型到iOS端    361
13.1  界面布局    361
13.1.1  頁面布局    361
13.1.2  實現手寫板    362
13.1.3  界面布局代碼實現    366
13.2  TensorFlow 模型轉CoreML模型    369
13.2.1  模型轉換    369
13.2.2  分析模型對象的調用接口    370
13.3  模型調用    373
13.3.1  實現模型調用    373
13.3.2  模型測試    376
```
