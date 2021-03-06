# 案例分析

###
```
Hands-On Artificial Intelligence for Cybersecurity
Alessandro Parisi August 2019
CH 5 Network Anomaly Detection with AI
```

### 資料集
```
https://github.com/MyDearGreatTeacher/AI201909/blob/master/data/network-logs.csv
```
```
!wget https://raw.githubusercontent.com/MyDearGreatTeacher/AI201909/master/data/network-logs.csv
```
```
REMOTE_PORT	LATENCY	THROUGHPUT	ANOMALY
21	15.94287532	16.20299807	0
20	12.66645095	15.89908374	1
80	13.89454962	12.95800822	0
21	13.62081292	15.45947525	0
21	15.70548485	15.33956527	0
23	15.59318973	15.61238106	0
21	15.48906755	15.64087368	0
80	15.52704801	15.63568031	0
21	14.07506707	15.76531533	0
......
```
```
延遲（Latency）：一個封包從來源端送出後，到目的端接收到這個封包，中間所花的時間。
頻寬（Bandwidth）：傳輸媒介的最大吞吐量（throughput）。

https://blog.gtwang.org/web-development/network-lantency-and-bandwidth/
```
### 基本統計分析

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

dataset = pd.read_csv('network-logs.csv')
hist_dist = dataset[['LATENCY', 'THROUGHPUT']].hist(grid=False, figsize=(10,4))

data = dataset[['LATENCY', 'THROUGHPUT']].values

plt.scatter(data[:, 0], data[:, 1], alpha=0.6)
plt.xlabel('LATENCY')
plt.ylabel('THROUGHPUT')
plt.title('DATA FLOW')
plt.show()
```

### 機器學習

```
import numpy as np
import pandas as pd

from sklearn.linear_model import *
from sklearn.tree import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
%matplotlib inline

# Load the data.
dataset = pd.read_csv('network-logs.csv')


samples = dataset.iloc[:, [1, 2]].values
targets = dataset['ANOMALY'].values

training_samples, testing_samples, training_targets, testing_targets = train_test_split(
         samples, targets, test_size=0.3, random_state=0)
```

###  使用k-Nearest Neighbors model
```
knc = KNeighborsClassifier(n_neighbors=2)
knc.fit(training_samples,training_targets)
knc_prediction = knc.predict(testing_samples)
knc_accuracy = 100.0 * accuracy_score(testing_targets, knc_prediction)
print ("K-Nearest Neighbours accuracy: " + str(knc_accuracy))
```
### 使用Decision tree model
```
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(training_samples,training_targets)
dtc_prediction = dtc.predict(testing_samples)
dtc_accuracy = 100.0 * accuracy_score(testing_targets, dtc_prediction)
print ("Decision Tree accuracy: " + str(dtc_accuracy))
```

### 使用Gaussian Naive Bayes model
```
gnb = GaussianNB()
gnb.fit(training_samples,training_targets)
gnb_prediction = gnb.predict(testing_samples)
gnb_accuracy = 100.0 * accuracy_score(testing_targets, gnb_prediction)
print ("Gaussian Naive Bayes accuracy: " + str(gnb_accuracy))
```

### 結果
```
K-Nearest Neighbours accuracy: 95.90163934426229
Decision Tree accuracy: 96.72131147540983
Gaussian Naive Bayes accuracy: 98.36065573770492
```

# 進階研讀:GaussianAnomalyDetection

### 正常行為 vs 異常行為
```
正常行為 ===滿足高斯分布

異常行為 ===孤立點outlier
```
### 程式範例
```
"""
Anomaly Detection Module
Oleksii Trekhleb:
https://github.com/trekhleb/homemade-machine-learning/blob/master/
homemade/anomaly_detection/gaussian_anomaly_detection.py
"""
"""Anomaly Detection Module"""

import math
import numpy as np


class GaussianAnomalyDetection:
    """GaussianAnomalyDetection Class"""

    def __init__(self, data):
        """GaussianAnomalyDetection constructor"""

        # Estimate Gaussian distribution.
        (self.mu_param, self.sigma_squared) = GaussianAnomalyDetection.estimate_gaussian(data)

        # Save training data.
        self.data = data

    def multivariate_gaussian(self, data):
        """Computes the probability density function of the multivariate gaussian distribution"""

        mu_param = self.mu_param
        sigma_squared = self.sigma_squared

        # Get number of training sets and features.
        (num_examples, num_features) = data.shape

        # nit probabilities matrix.
        probabilities = np.ones((num_examples, 1))

        # Go through all training examples and through all features.
        for example_index in range(num_examples):
            for feature_index in range(num_features):
                # Calculate the power of e.
                power_dividend = (data[example_index, feature_index] - mu_param[feature_index]) ** 2
                power_divider = 2 * sigma_squared[feature_index]
                e_power = -1 * power_dividend / power_divider

                # Calculate the prefix multiplier.
                probability_prefix = 1 / math.sqrt(2 * math.pi * sigma_squared[feature_index])

                # Calculate the probability for the current feature of current example.
                probability = probability_prefix * (math.e ** e_power)
                probabilities[example_index] *= probability

        # Return probabilities for all training examples.
        return probabilities

    @staticmethod
    def estimate_gaussian(data):
        """This function estimates the parameters of a Gaussian distribution using the data in X."""

        # Get number of features and number of examples.
        num_examples = data.shape[0]

        # Estimate Gaussian parameters mu and sigma_squared for every feature.
        mu_param = (1 / num_examples) * np.sum(data, axis=0)
        sigma_squared = (1 / num_examples) * np.sum((data - mu_param) ** 2, axis=0)

        # Return Gaussian parameters.
        return mu_param, sigma_squared

    @staticmethod
    def select_threshold(labels, probabilities):
        # pylint: disable=R0914
        """Finds the best threshold (epsilon) to use for selecting outliers"""

        best_epsilon = 0
        best_f1 = 0

        # History data to build the plots.
        precision_history = []
        recall_history = []
        f1_history = []

        # Calculate the epsilon steps.
        min_probability = np.min(probabilities)
        max_probability = np.max(probabilities)
        step_size = (max_probability - min_probability) / 1000

        # Go through all possible epsilons and pick the one with the highest f1 score.
        for epsilon in np.arange(min_probability, max_probability, step_size):
            predictions = probabilities < epsilon

            # The number of false positives: the ground truth label says it’s not
            # an anomaly, but our algorithm incorrectly classified it as an anomaly.
            false_positives = np.sum((predictions == 1) & (labels == 0))

            # The number of false negatives: the ground truth label says it’s an anomaly,
            # but our algorithm incorrectly classified it as not being anomalous.
            false_negatives = np.sum((predictions == 0) & (labels == 1))

            # The number of true positives: the ground truth label says it’s an
            # anomaly and our algorithm correctly classified it as an anomaly.
            true_positives = np.sum((predictions == 1) & (labels == 1))

            # Prevent division by zero.
            if (true_positives + false_positives) == 0 or (true_positives + false_negatives) == 0:
                continue

            # Precision.
            precision = true_positives / (true_positives + false_positives)

            # Recall.
            recall = true_positives / (true_positives + false_negatives)

            # F1.
            f1_score = 2 * precision * recall / (precision + recall)

            # Save history data.
            precision_history.append(precision)
            recall_history.append(recall)
            f1_history.append(f1_score)

            if f1_score > best_f1:
                best_epsilon = epsilon
                best_f1 = f1_score

        return best_epsilon, best_f1, precision_history, recall_history, f1_history
```
```
#from gaussian_anomaly_detection import GaussianAnomalyDetection

gaussian_anomaly_detection = GaussianAnomalyDetection(data)

print('mu param estimation: ')
print(gaussian_anomaly_detection.mu_param)

print('\n')

print('sigma squared estimation: ')
print(gaussian_anomaly_detection.sigma_squared)

targets = dataset['ANOMALY'].values.reshape((data.shape[0], 1))
probs = gaussian_anomaly_detection.multivariate_gaussian(data)

(threshold, F1, precision_, recall_, f1_) = gaussian_anomaly_detection.select_threshold(targets, probs)

print('\n')

print('threshold estimation: ')
print(threshold)

outliers = np.where(probs < threshold)[0]

# Plot original data.
plt.scatter(data[:, 0], data[:, 1], alpha=0.6, label='Dataset')
plt.xlabel('LATENCY')
plt.ylabel('THROUGHPUT')
plt.title('DATA FLOW')

# Plot the outliers.
plt.scatter(data[outliers, 0], data[outliers, 1], alpha=0.6, c='red', label='Outliers')

# Display plots.
plt.legend()
plt.plot()


print('F1 score: ')
print(F1)


from sklearn.metrics import roc_curve

FPR, TPR, OPC = roc_curve(targets, probs)
# Plotting Sensitivity
plt.plot(OPC,TPR)


# Plotting ROC curve
plt.plot(FPR,TPR)
```
```
mu param estimation: 
[14.42070163 15.39209133]


sigma squared estimation: 
[2.09674794 1.37224807]


threshold estimation: 
0.0002717683673539915
F1 score: 
0.6666666666666666
[<matplotlib.lines.Line2D at 0x7f02a3dd7ba8>]
```
# 進階研讀:DGA detection

```

```

### Domain generation algorithm::botnet產生網域的演算法
```
Various families of malware use domain generation algorithms (DGAs) to generate a large number of pseudo-random domain names 
to connect to a command and control (C&C) server. 

技術一:
In order to block DGA C&C traffic, security organizations must first discover the algorithm by reverse engineering malware samples, 
then generating a list of domains for a given seed. 

The domains are then either preregistered or published in a DNS blacklist. 

This process is not only tedious, but can be readily circumvented by malware authors using a large number of seeds in algorithms 
with multivariate recurrence properties (e.g., banjori) or by using a dynamic list of seeds (e.g., bedep). 

技術二:分類器===產生的DNS查詢的Domain是否為DGA產生的
Another technique to stop malware from using DGAs is to intercept DNS queries on a network and predict whether domains are DGA generated. 
Such a technique will alert network administrators to the presence of malware on their networks. 
In addition, if the predictor can also accurately predict the family of DGAs, 
then network administrators can also be alerted to the type of malware that is on their networks. 
```


### 
```
Predicting Domain Generation Algorithms with Long Short-Term Memory Networks
Jonathan Woodbridge, Hyrum S. Anderson, Anjum Ahuja, Daniel Grant
(Submitted on 2 Nov 2016)
https://arxiv.org/abs/1611.00791
```
### DGA Dataset
```
 https://github.com/nickwallen/botnet-dga-classifier/tree/master/data
(accessed on 10 November 2017).

botnet-dga-classifier/data/dds-malicious-domains.csv

host	source
dwjfjqyunkhfkr.ru	cryptolocker
htwwaixsreth.ru	cryptolocker
ihvdawbftgkdt.ru	cryptolocker
usmnqpmwklxi.ru	cryptolocker
tufomgucxbxqvut.ru	cryptolocker
xfjwnqdamslry.ru	cryptolocker
xyryawrkldrcld.ru	cryptolocker
mspftdngwpvvo.ru	cryptolocker
vlfmnwdgrofy.ru	cryptolocker
rfplwgxjbimkrr.ru	cryptolocker
kudscqlvxcplx.ru	cryptolocker
eoavwaywelqqtdw.ru	cryptolocker
vifgurkyjepqju.ru	cryptolocker
exhtpubchhnimc.ru	cryptolocker
xppyorwinqlohc.ru	cryptolocker
ikrnxwxggbjuqqg.ru	cryptolocker
xsnqupvukyni.ru	cryptolocker
gsfokpekvaas.ru	cryptolocker
xjehuwyttxbku.ru	cryptolocker
```
