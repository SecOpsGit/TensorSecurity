#
```
TensorFlow深度學習實戰
　叢書名稱：	智能系統與技術叢書
　作　　者：	(波蘭)安東尼奧‧古利
　出版單位：	機械工業
　ＩＳＢＮ：	9787111615750
 
TensorFlow 1.x Deep Learning Cookbook
Antonio Gulli, Amita Kapoor
December 12, 2017
536 pages

https://github.com/PacktPublishing/TensorFlow-1x-Deep-Learning-Cookbook

5.4　使用 VGGNet、ResNet、Inception和Xception分類圖像 
```

# keras.applications模組
```
Keras 的應用模組（keras.applications）提供了帶有預訓練權值的深度學習模型，
這些模型可以用來進行預測、特徵提取和微調（fine-tuning）。

當你初始化一個預訓練模型時，會自動下載權重到 ~/.keras/models/ 目錄下。

可用的模型:在 ImageNet 上預訓練過的用於圖像分類的模型：
Xception
VGG16
VGG19
ResNet, ResNetV2, ResNeXt
InceptionV3
InceptionResNetV2
MobileNet
MobileNetV2
DenseNet
NASNet

所有的這些架構都相容所有的後端 (TensorFlow, Theano 和 CNTK)，並且會在產生實體時，
根據 Keras 設定檔〜/.keras/keras.json 中設置的圖像資料格式構建模型。

舉個例子，如果你設置 image_data_format=channels_last，
則載入的模型將按照 TensorFlow 的維度順序來構造，即「高度-寬度-深度」（Height-Width-Depth）的順序。
https://keras.io/zh/applications/
```
# 使用 ResNet50 進行 ImageNet 分類

```
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# 將結果解碼為元組列表 (class, description, probability)
# (一個列表代表批次中的一個樣本）

print('Predicted:', decode_predictions(preds, top=3)[0])

# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), 
# (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

```

# 使用 VGGNet、ResNet、Inception和Xception分類圖像
```
!wget https://raw.githubusercontent.com/PacktPublishing/TensorFlow-1x-Deep-Learning-Cookbook/master/Chapter05/images/parrot.jpg
```
```
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
%matplotlib inline


MODELS = {
    "vgg16": (VGG16, (224, 224)),
    "vgg19": (VGG19, (224, 224)),
    "inception": (InceptionV3, (299, 299)),
    "xception": (Xception, (299, 299)), # TensorFlow ONLY
    "resnet": (ResNet50, (224, 224))
}


def image_load_and_convert(image_path, model):

    pil_im = Image.open(image_path, 'r')
    imshow(np.asarray(pil_im))
    
    # initialize the input image shape 
    # and the pre-processing function (this might need to be changed
    inputShape = MODELS[model][1]
    preprocess = imagenet_utils.preprocess_input
    image = load_img(image_path, target_size=inputShape)
    image = img_to_array(image)
    # the original networks have been trained on an addiitonal
    # dimension taking into account the batch size
    # we need to add this dimension for consistency
    # even if we have one image only
    image = np.expand_dims(image, axis=0)
    image = preprocess(image)
    
    return image
    
def classify_image(image_path, model):
    img = image_load_and_convert(image_path, model)
    Network = MODELS[model][0]
    model = Network(weights="imagenet")
    preds = model.predict(img)
    P = imagenet_utils.decode_predictions(preds)
    # loop over the predictions and display the rank-5 predictions 
    # along with probabilities
    for (i, (imagenetID, label, prob)) in enumerate(P[0]):
        print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
```

```
classify_image("parrot.jpg", "vgg16")
classify_image("parrot.jpg", "vgg19")
classify_image("parrot.jpg", "inception")
classify_image("parrot.jpg", "xception")
classify_image("parrot.jpg", "resnet")
```

# 
```
def print_model(model):
    print ("Model:",model)
    Network = MODELS[model][0]
    model = Network(weights="imagenet")
    model.summary()
```

```

print_model('vgg16')
print_model('vgg19')
print_model('inception')
print_model('xception')
print_model('resnet')
```

#
```
def image_load(image_path):

    pil_im = Image.open(image_path, 'r')
    pil_im.show()
```
```

image_load('imagenet_vggnet_table1.png')
image_load('imagenet_resnet_identity.png')

```
