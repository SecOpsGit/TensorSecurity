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
