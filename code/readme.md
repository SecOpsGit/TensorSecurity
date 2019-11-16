# 使用docker 學RL
```
https://github.com/mlss-2019/tutorials

docker pull isakfalk/mlss:latest
docker run --name mlss -p 8888:8888 isakfalk/mlss
```
# Google colab
```
from google.colab import drive
drive.mount('/content/drive/')

!pip install unidecode

import unidecode
import string
import random
import re

!ls '/content/drive/My Drive/Text Generator'
#ls '/content/drive/My Drive'

all_characters = string.printable
n_characters = len(all_characters)

file = unidecode.unidecode(open('/content/drive/My Drive/Text Generator/Oliver.txt').read())
file_len = len(file)
```
#
```
https://github.com/fastai/fastai
The fastai deep learning library, plus lessons and tutorials http://docs.fast.ai
```

```
Deep Learning Book Series · 2.2 Multiplying Matrices and Vectors
https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.2-Multiplying-Matrices-and-Vectors/
```
```
Understanding PCA with an example
https://www.linkedin.com/pulse/understanding-pca-example-subhasree-chatterjee

Principal Component Analysis (PCA)
https://www.youtube.com/watch?v=g-Hb26agBFg

http://www-labs.iro.umontreal.ca/~pift6080/H09/documents/papers/pca_tutorial.pdf

http://setosa.io/ev/principal-component-analysis/
```
```
Dimensional Reduction -- MDS
https://ithelp.ithome.com.tw/articles/10188863

https://web.mit.edu/cocosci/Papers/sci_reprint.pdf
```
```
2016 US Election
Explore data related to the 2016 US Election
https://www.kaggle.com/benhamner/2016-us-election
```
