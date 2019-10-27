# TensorFlow與圖像處理
```

```

```
1. 圖像與TensorFlow
TensorFlow在設計時，就考慮了給將圖像作為神經網路的輸入提供支援。
TensorFlow支援載入常見的圖像格式（JPG、PNG），可在不同的顏色空間（RGB、RGBA）中工作，並能夠完成常見的圖像操作任務。

雖然TensorFlow使得圖像操作變得容易，但仍然面臨一些挑戰。
使用圖像時，所面臨的最大挑戰便是最終需要載入的張量的尺寸。
每幅圖像都需要用一個與圖像尺寸（heightwidthchannel）相同的張量表示。
再次提醒，通道是用一個包含每個通道中顏色數量的標量的秩1張量表示。

在TensorFlow中，一個紅色的RGB圖元可用如下張量表示：

red = tf.constant([255, 0, 0])

每個標量都可修改，以使圖元值為另一個顏色值或一些顏色值的混合。
對於RGB顏色空間，圖元對應的秩1張量的格式為[red，green，blue]。

一幅圖像中的所有圖元都存儲在磁片檔中，它們都需要被載入到記憶體中，以便TensorFlow對其進行操作。

2. 載入圖像
TensorFlow在設計時便以能夠從磁片快速載入檔為目標。
圖像的載入與其他大型二進位檔案的載入是相同的，只是圖像的內容需要解碼。
載入下列3×3的JPG格式的示例圖像的過程與載入任何其他類型的檔完全一致。

載入圖像代碼：

import tensorflow as tf
import numpy as np
sess = tf.InteractiveSession()
# match_filenames_once 將接收一個規則運算式，但在本例中不需要
image_filename = "test-input-image.jpg"
# string_input_producer會產生一個檔案名佇列
filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(image_filename))

# reader從檔案名佇列中讀數據。對應的方法是reader.read
image_reader = tf.WholeFileReader()
_, image_file = image_reader.read(filename_queue)
# 圖像將被解碼
image = tf.image.decode_jpeg(image_file)

# tf.train.string_input_producer定義了一個epoch變數，要對它進行初始化
tf.local_variables_initializer().run()

# 使用start_queue_runners之後，才會開始填充佇列
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
sess.run(image)


輸出：

array([[[  0,   9,   4],
        [254, 255, 250],
        [255,  11,   8]],

       [[ 10, 195,   5],
        [  6,  94, 227],
        [ 14, 205,  11]],

       [[255,  10,   4],
        [249, 255, 244],
        [  3,   8,  11]]], dtype=uint8)
        

在上述代碼中，輸入生成器（tf.train.string_input_producer）會找到所需的檔，並將其載入到一個佇列中。
載入圖像要求將完整的檔載入到記憶體中（tf.WholeFileReader）。
一旦文件被讀取（image_reader.read），
所得到的圖像就將被解碼（tf.image.decode_jpeg）。

這樣便可以查看這幅圖像sess.run(image)。由於按照名稱只存在一個檔，所以佇列將始終返回同一幅圖像。

載入圖像後，查看輸出。注意，它是一個非常簡單的三階張量。RGB值對應9個一階張量。

3. 圖像格式
訓練一個CNN需要大量時間，載入非常大的檔會進一步增加訓練所需的時間。
即便增加的時間在可接受的範圍內，單幅圖像的尺寸也很難存放在大多數系統的GPU顯存中。

輸入圖像尺寸過大也會為大多數CNN模型的訓練產生不利影響。
CNN總是試圖找到圖像中的本征屬性，雖然這些屬性有一定的獨特性，但也需要推廣到其他具有類似結果的圖像上。
使用尺寸過大的輸入會使網路中充斥大量無關資訊，從而影響模型的泛化能力。

圖像中的重要資訊是通過按照某種恰當的檔案格式存儲並處理得以強調的。在使用圖像時，不同的格式可用於解決不同的問題。

3.1. JPEG與PNG
TensorFlow擁有兩種可對圖像資料解碼的格式，一種是tf.image.decode_jpeg，
另一種是tf.image.decode_png。
在電腦視覺應用中，這些都是常見的檔案格式，因為將其他格式轉換為這兩種格式非常容易。

PNG圖像會存儲任何alpha通道的資訊，如果在訓練模型時需要利用alpha資訊（透明度），則這一點非常重要。
一種應用場景是當使用者手工切除圖像的一些區域，如狗所戴的不相關的小丑帽。
將這些區域置為黑色會使它們與該圖像中的其他黑色區域看起來有相似的重要性。
若將所移除的帽子對應的區域的alpha值設為0，則有助於標識該區域是被移除的區域。

使用JPEG圖像時，不要進行過於頻繁的操作，因為這樣會留下一些偽影（artifact）。
在進行任何必要的操作時，獲取圖像的原始資料，並將它們匯出為JPEG檔。
為了節省訓練時間，請試著儘量在圖像載入之前完成對它們的操作。

如果一些操作是必要的，PNG圖像可以很好地工作。
PNG格式採用的是無失真壓縮，因此它會保留原始檔（除非被縮放或降採樣）中的全部資訊。
PNG格式的缺點在於檔體積相比JPEG要大一些。

3.2. TFRecord
為將二進位資料和標籤（訓練的類別標籤）資料存儲在同一個檔中，TensorFlow設計了一種內置檔案格式，該格式被稱為TFRecord，
它要求在模型訓練之前通過一個預處理步驟將圖像轉換為TFRecord格式。
該格式的最大優點是將每幅輸入圖像和與之關聯的標籤放在同一檔中。

從技術角度講，TFRecord檔是protobuf格式的檔。
作為一種經過預處理的格式，它們是非常有用的。
由於它們不對資料進行壓縮，所以可被快速載入到記憶體中。

3.3. 保存為TFRecord格式代碼舉例：
將一幅圖像及其標籤寫入一個新的TFRecord格式的檔中。


# 重複使用上面載入的圖像，並給它一個假標籤
image_label = b'\x01'  # 標籤的格式為獨熱編碼（one-hot encoding） (00000001)

# 將張量轉換成位元組，注意這將載入整個影像檔
image_loaded = sess.run(image)
image_bytes = image_loaded.tobytes()
image_height, image_width, image_channels = image_loaded.shape

# 匯出 TFRecord
writer = tf.python_io.TFRecordWriter("./output/training-image.tfrecord")

# 不要在此示例檔中存儲寬度，高度或圖像通道，以節省空間，但不是必需的
example = tf.train.Example(features=tf.train.Features(feature={
            'label':
             tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
            'image':
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
        }))

# 這將保存樣本到文字檔tfrecord
writer.write(example.SerializeToString())
writer.close()


在這段示例代碼中，圖像被載入到記憶體中並被轉換為位元組陣列。
之後，這些位元組被添加到tf.train.Example文件中，
而後者在被保存到磁片之前先通過SerializeToString序列化為二進位字元串。

序列化是一種將記憶體物件轉換為某種可安全傳輸到某個檔的格式。上面序列化的樣本現在被保存為一種可被載入的格式，並可被反序列化為這裡的樣本格式。

3.4. 從TFRecord檔載入
由於圖像被保存為TFRecord檔，所以可被再次載入（從TFRecord檔載入，而非從影像檔載入）。
在訓練階段，載入圖像及其標籤是必需的。這樣相比將圖像及其標籤分開載入會節省一些時間。

代碼示例：

# 載入TFRecord
tf_record_filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("./output/training-image.tfrecord"))

# 請注意不同的記錄讀取器，這個設計用於與TFRecord檔一起使用，這些檔可能有多個示例。
tf_record_reader = tf.TFRecordReader()
_, tf_record_serialized = tf_record_reader.read(tf_record_filename_queue)

# 標籤和圖像以位元組存儲，但可以作為int64或float64類型存儲在序列化的tf.Example protobuf中。
tf_record_features = tf.parse_single_example(
    tf_record_serialized,
    features={
        'label': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string),
    })

# 使用tf.uint8，因為所有的通道資訊都在0-255之間
tf_record_image = tf.decode_raw(
    tf_record_features['image'], tf.uint8)

# 調整圖像的尺寸，使其與所保存的圖像類似，但這並非必需的
# 用實值表示圖像的高度、寬度和通道，因為必須對輸入的形狀進行調整
tf_record_image = tf.reshape(
    tf_record_image,
    [image_height, image_width, image_channels])

tf_record_label = tf.cast(tf_record_features['label'], tf.string)


首先，按照與其他任何檔相同的方式載入該檔，主要差別在於之後該檔會由TFRecordReader對象讀取。
tf.parse_single_example並不對圖像進行解碼，而是解析TFRecord，
然後圖像會按原始位元組（tf.decode_raw）被讀取。

該檔被載入後，為使其佈局符合tf.nn.conv2d的要求，即上面代碼獲取的值[image_height，image_width，image_channels]，
需要對形狀進行調整（tf.reshape）。

為將batch_size維添加到input_batch中，需要對維數進行擴展（tf.expand）。

在本例中，TFRecord檔中雖然只包含一個影像檔，但這類記錄檔也支援被寫入多個樣本。
將整個訓練集保存在一個TFRecord檔中是安全的，但分開存儲也完全可以。

當需要檢查保存到磁片的檔是否與從TensorFlow載入的圖像是同一圖像時，可使用下列代碼：

sess.close()
sess = tf.InteractiveSession()
tf.local_variables_initializer().run()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

sess.run(tf.equal(image, tf_record_image))

sess.run(tf_record_label)

# setup-only-ignore
tf_record_filename_queue.close(cancel_pending_enqueues=True)
coord.request_stop()
coord.join(threads)


輸出：

array([[[ True,  True,  True],
        [ True,  True,  True],
        [ True,  True,  True]],

       [[ True,  True,  True],
        [ True,  True,  True],
        [ True,  True,  True]],

       [[ True,  True,  True],
        [ True,  True,  True],
        [ True,  True,  True]]], dtype=bool)


可以看出，原始圖像的所有屬性都和從TFRecord檔載入的圖像一致。
為確認這一點，可從TFRecord檔載入標籤，並檢查它與之前保存的版本是否一致。

創建一個既可存儲原始圖像資料，也可存儲其期望的輸出標籤的TFRecord檔，能夠降低訓練中的複雜性。
儘管使用TFRecord檔並非必需，但在使用圖像資料時，卻是強烈推薦的。
如果對於某個工作流，它不能很好地工作，那麼仍然建議在訓練之前對圖像進行預處理並將預處理結果保存下來。
每次載入圖像時才對其進行處理是不推薦的做法。

4. 圖像操作
在大多數場景中，對圖像的操作最好能在預處理階段完成。
預處理包括對圖像裁剪、縮放以及灰度調整等。
另一方面，在訓練時對圖像進行操作有一個重要的用例。

當一幅圖像被載入後，可對其做翻轉或扭曲處理，以使輸入給網路的訓練資訊多樣化。
雖然這個步驟會進一步增加處理時間，但卻有助於緩解過擬合現象。

TensorFlow並未設計成一個影像處理框架。與TensorFlow相比，有一些Python庫（如PIL和OpenCV）支援更豐富的圖像操作。
對於TensorFlow，可將那些對訓練CNN十分有用的影像處理方法總結如下。

4.1. 裁剪
裁剪會將圖像中的某些區域移除，將其中的資訊完全丟棄。
裁剪與tf.slice類似，後者是將一個張量中的一部分從完整的張量中移除。
當沿某個維度存在多餘的輸入時，為CNN對輸入圖像進行裁剪便是十分有用的。
例如，為減少輸入的尺寸，可對狗位於圖像中心的圖片進行裁剪。

代碼：

sess.run(tf.image.central_crop(image, 0.1))

執行上面的代碼後，可得到輸出：

array([[[  6,  94, 227]]], dtype=uint8)

被裁剪前的圖像資料：

array([[[  0,   9,   4],
        [254, 255, 250],
        [255,  11,   8]],

       [[ 10, 195,   5],
        [  6,  94, 227],
        [ 14, 205,  11]],

       [[255,  10,   4],
        [249, 255, 244],
        [  3,   8,  11]]], dtype=uint8)



這段示例代碼利用了tf.image.central_crop將圖像中10%的區域摳出，並將其返回。
該方法總是會基於所使用的圖像的中心返回結果。

裁剪通常在預處理階段使用，但在訓練階段，若背景也有用時，它也可派上用場。
當背景有用時，可隨機化裁剪區域起始位置到圖像中心的偏移量來實現裁剪。

代碼：

# 這個裁剪方法僅可接收實值輸入
real_image = sess.run(image)

bounding_crop = tf.image.crop_to_bounding_box(
    real_image, offset_height=0, offset_width=0, target_height=2, target_width=1)

sess.run(bounding_crop)
1

輸出：

array([[[  0,   9,   4]],

       [[ 10, 195,   5]]], dtype=uint8)


為從位於（0，0）的圖像的左上角圖元開始對圖像裁剪，這段示例代碼使用了tf.image.crop_to_bounding_box。
目前，該函數只能接收一個具有確定形狀的張量。
因此，輸入圖像需要事先在資料流程圖中運行。

4.2. 邊界填充
為使輸入圖像符合期望的尺寸，可用0進行邊界填充。
可利用tf.pad函數完成該操作，但對於尺寸過大或過小的圖像，TensorFlow還提供了另外一個非常有用的尺寸調整方法。
對於尺寸過小的圖像，該方法會圍繞該圖像的邊界填充一些灰度值為0的圖元。
通常，該方法用於調整小圖像的尺寸，因為任何其他調整尺寸的方法都會使圖像的內容產生扭曲。

代碼：

# 該邊界填充方法僅可接收實值輸入
real_image = sess.run(image)

pad = tf.image.pad_to_bounding_box(
    real_image, offset_height=0, offset_width=0, target_height=4, target_width=4)

sess.run(pad)


輸出：

array([[[  0,   9,   4],
        [254, 255, 250],
        [255,  11,   8],
        [  0,   0,   0]],

       [[ 10, 195,   5],
        [  6,  94, 227],
        [ 14, 205,  11],
        [  0,   0,   0]],

       [[255,  10,   4],
        [249, 255, 244],
        [  3,   8,  11],
        [  0,   0,   0]],

       [[  0,   0,   0],
        [  0,   0,   0],
        [  0,   0,   0],
        [  0,   0,   0]]], dtype=uint8)


這段示例代碼將圖像的高度和寬度都增加了一個圖元，所增加的新圖元的灰度值均為0。
對於尺寸過小的圖像，這種邊界填充方式是非常有用的。
如果訓練集中的圖像存在多種不同的長寬比，便需要這樣的處理方法。

對於那些長寬比不一致的圖像，TensorFlow還提供了一種組合了pad和crop的尺寸調整的便捷方法。

代碼：

real_image = sess.run(image)

crop_or_pad = tf.image.resize_image_with_crop_or_pad(
    real_image, target_height=2, target_width=5)

sess.run(crop_or_pad)


輸出：

array([[[  0,   0,   0],
        [  0,   9,   4],
        [254, 255, 250],
        [255,  11,   8],
        [  0,   0,   0]],

       [[  0,   0,   0],
        [ 10, 195,   5],
        [  6,  94, 227],
        [ 14, 205,  11],
        [  0,   0,   0]]], dtype=uint8)


real_image的高度被減小了兩個圖元，而通過邊界填充0圖元使寬度得以增加。
這個函數的操作是相對圖像輸入的中心進行的。

4.3. 翻轉
翻轉操作的含義與其字面意思一致，即每個圖元的位置都沿水準或垂直方向翻轉。
從技術角度講，翻轉是在沿垂直方向翻轉時所採用的術語。

利用TensorFlow對圖像執行翻轉操作是非常有用的，這樣可以為同一幅訓練圖像賦予不同的視角。

例如，一幅左耳捲曲的澳大利亞牧羊犬圖像如果經過了翻轉，便有可能與其他的圖像中右耳捲曲的狗匹配。

TensorFlow有一些函數可實現垂直翻轉、水平翻轉，使用者可隨意選擇。
隨機翻轉一幅圖像的能力對於防止模型對圖像的翻轉版本產生過擬合非常有用。

代碼：

top_left_pixels = tf.slice(image, [0, 0, 0], [2, 2, 3])

flip_horizon = tf.image.flip_left_right(top_left_pixels)
flip_vertical = tf.image.flip_up_down(flip_horizon)

sess.run([top_left_pixels, flip_vertical])


輸出如下：

[array([[[  0,   9,   4],
         [254, 255, 250]],

        [[ 10, 195,   5],
         [  6,  94, 227]]], dtype=uint8), array([[[  6,  94, 227],
         [ 10, 195,   5]],

        [[254, 255, 250],
         [  0,   9,   4]]], dtype=uint8)]


這段示例代碼對一幅圖像的一個子集首先進行水準翻轉，然後進行垂直翻轉。
該子集是用tf.slice選取的，這是因為對原始圖像翻轉返回的是相同的圖像（僅對這個例子而言）。
這個圖元子集解釋了當圖像發生翻轉時所發生的變化。
tf.image.flip_left_right和tf.image.flip_up_down都可對張量進行操作，而非僅限於圖像。

這些函數對圖像的翻轉具有確定性，要想實現對圖像隨機翻轉，可利用另一組函數。

隨機翻轉代碼：

top_left_pixels = tf.slice(image, [0, 0, 0], [2, 2, 3])

random_flip_horizon = tf.image.random_flip_left_right(top_left_pixels)
random_flip_vertical = tf.image.random_flip_up_down(random_flip_horizon)

sess.run(random_flip_vertical)


輸出：

array([[[  3, 108, 233],
        [  0, 191,   0]],

       [[255, 255, 255],
        [  0,   0,   0]]], dtype=uint8)


這個例子與之前的例子具有相同的邏輯，唯一的區別在於本例中的輸出是隨機的。
這個常式每次運行時，都會得到不同的輸出。有一個名稱為seed的參數可控制翻轉發生的隨機性。

4.4. 飽和與平衡
可在互聯網上找到的圖像通常都事先經過了編輯。
例如，Stanford Dogs資料集中的許多圖像都具有過高的飽和度（大量顏色）。
當將編輯過的圖像用於訓練時，可能會誤導CNN模型去尋找那些與編輯過的圖像有關的模式，而非圖像本身所呈現的內容。

為向在圖像資料上的訓練提供幫助，TensorFlow實現了一些通過修改飽和度、色調、對比度和亮度的函數。
利用這些函數可對這些圖像屬性進行簡單的操作和隨機修改。對訓練而言，這種隨機修改是非常有用的，原因與圖像的隨機翻轉類似。

對屬性的隨機修改能夠使CNN精確匹配經過編輯的或不同光照條件下的圖像的某種特徵。

調整brightness代碼:

example_red_pixel = tf.constant([254., 2., 15.])
adjust_brightness = tf.image.adjust_brightness(example_red_pixel, 0.2)

sess.run(adjust_brightness)


輸出：

array([ 254.19999695,    2.20000005,   15.19999981], dtype=float32)

這個例子提升了一個以紅色為主的圖元的灰度值（增加了0.2）。

對比度調整代碼：

adjust_contrast = tf.image.adjust_contrast(image, -.5)
sess.run(tf.slice(adjust_contrast, [1, 0, 0], [1, 3, 3]))

輸出：

array([[[169,  76, 125],
        [171, 126,  13],
        [167,  71, 122]]], dtype=uint8)


這段示例代碼將對比度調整了-0.5，這將生成一個識別度相當差的新圖像。
調節對比度時，最好選擇一個較小的增量，以避免對圖像造成“過曝”。
這裡的“過曝”的含義與神經元出現飽和類似，即達到了最大值而無法恢復。
當對比度變化時，圖像中的圖元可能會呈現出全白和全黑的情形。

簡而言之，tf.slice運算的目的是突出發生改變的圖元。當運行該運算時，它是不需要的。

調整色度代碼：

adjust_hue = tf.image.adjust_hue(image, 0.7)

sess.run(tf.slice(adjust_hue, [1, 0, 0], [1, 3, 3]))


輸出：

array([[[195,  38,   5],
        [ 49, 227,   6],
        [205,  46,  11]]], dtype=uint8)


這段示例代碼調整了圖像中的色度，使其色彩更加豐富。
該調整函數接收一個delta參數，用於控制需要調節的色度數量。

調整飽和的代碼：

adjust_saturation = tf.image.adjust_saturation(image, 0.4)

sess.run(tf.slice(adjust_saturation, [1, 0, 0], [1, 3, 3]))


輸出：

array([[[121, 195, 119],
        [138, 174, 227],
        [128, 205, 127]]], dtype=uint8)


這段代碼與調節對比度的那段代碼非常類似。
為識別邊緣，對圖像進行過飽和處理是很常見的，因為增加飽和度能夠突出顏色的變化。

5. 顏色
CNN通常使用具有單一顏色的圖像來訓練。當一幅圖像只有單一顏色時，我們稱它使用了灰度顏色空間，即單顏色通道。

對大多數電腦視覺相關任務而言，使用灰度值是合理的，因為要瞭解圖像的形狀無須借助所有的顏色資訊。
縮減顏色空間可加速訓練過程。為描述圖像中的灰度，僅需一個單個分量的秩1張量即可，而無須像RGB圖像那樣使用含3個分量的秩1張量。

雖然只使用灰度資訊有一些優點，但也必須考慮那些需要利用顏色的區分性的應用。
在大多數電腦視覺任務中，如何使用圖像中的顏色都頗具挑戰性，因為很難從數學上定義兩個RGB顏色之間的相似度。
為在CNN訓練中使用顏色，對圖像進行顏色空間變換有時是非常有用的。

5.1. 灰度
灰度圖具有單個分量，且其取值範圍與RGB圖像中的顏色一樣，也是[0，255]。

代碼示例：

gray = tf.image.rgb_to_grayscale(image)
sess.run(tf.slice(gray, [0, 0, 0], [1, 3, 1]))


輸出：

array([[[  5],
        [254],
        [ 83]]], dtype=uint8)


這個例子將RGB圖像轉換為灰度圖。
tf.slice運算提取了最上一行的圖元，並查看其顏色是否發生了變化。
這種灰度變換是通過將每個圖元的所有顏色值取平均，並將其作為灰度值實現的。

5.2. HSV空間
色度、飽和度和灰度值構成了HSV顏色空間。
與RGB空間類似，這個顏色空間也是用含3個分量的秩1張量表示的。
HSV空間所度量的內容與RGB空間不同，它所度量的是圖像的一些更為貼近人類感知的屬性。
有時HSV也被稱為HSB，其中字母B表示亮度值。


代碼示例：

hsv = tf.image.rgb_to_hsv(tf.image.convert_image_dtype(image, tf.float32))

sess.run(tf.slice(hsv, [0, 0, 0], [3, 3, 3]))


輸出：

array([[[ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  1.        ],
        [ 0.        ,  1.        ,  0.99607849]],

       [[ 0.33333334,  1.        ,  0.74901962],
        [ 0.59057975,  0.98712444,  0.91372555],
        [ 0.33333334,  1.        ,  0.74901962]],

       [[ 0.        ,  1.        ,  0.99607849],
        [ 0.        ,  0.        ,  1.        ],
        [ 0.        ,  0.        ,  0.        ]]], dtype=float32)


5.3. RGB空間
到目前為止，所有的示例代碼中使用的都是RGB顏色空間。
它對應於一個含3個分量的秩1張量，其中紅、綠和藍的取值範圍均為[0，255]。
大多數圖像本身就位於RGB顏色空間中，但考慮到有些圖像可能會來自其他顏色空間，TensorFlow也提供了一些顏色空間轉換的內置函數。

代碼示例：

rgb_hsv = tf.image.hsv_to_rgb(hsv)
rgb_grayscale = tf.image.grayscale_to_rgb(gray)


這段示例代碼非常簡單，只是從灰度空間轉換到RGB空間並無太大的實際意義。
RGB圖像需要三種顏色，而灰度圖像只需要一種顏色。
當轉換（灰度到RGB）發生時，RGB中每個圖元的各通道都將被與灰度圖中對應圖元的灰度值填充。

5.4. LAB空間
TensorFlow並未為LAB顏色空間提供原生支援。它是一種有用的顏色空間，因為與RGB相比，它能夠映射大量可感知的顏色。
雖然TensorFlow並未為它提供原生支援，但它卻是一種經常在專業場合使用的顏色空間。

Python庫python-colormath為LAB和其他本書未提及的顏色空間提供了轉換支援。

使用LAB顏色空間最大的好處在於與RGB或HSV空間相比，它對顏色差異的映射更貼近人類的感知。
在LAB顏色空間中，兩個顏色的歐氏距離在某種程度上能夠反映人類所感受到的這兩種顏色的差異。

5.5. 圖像資料類型轉換
在這些例子中，為說明如何修改圖像的資料類型，tf.to_float被多次用到。
對於某些例子，使用這種方式是可以的，

但TensorFlow還提供了一個內置函數，用於當圖像資料類型發生變化時恰當地對圖元值進行比例變換。
tf.image.convert_iamge_dtype（image，dtype，saturate=False）是將圖像的資料類型從tf.uint8更改為tf.float的便捷方法。

```
