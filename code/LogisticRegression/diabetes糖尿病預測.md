#
```
# -*- coding: utf-8 -*-
!wget https://raw.githubusercontent.com/MyDearGreatTeacher/AI201909/master/data/diabetes.csv

# Commented out IPython magic to ensure Python compatibility.
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# % matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.externals import joblib


diabetesDF = pd.read_csv('diabetes.csv')
diabetesDF.head()
diabetesDF.info()
corr = diabetesDF.corr()
corr

sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)

#Total 768 patients record
#Using 650 data for training
# Using 100 data for testing
#Using 18 data for checking

dfTrain = diabetesDF[:650]
dfTest = diabetesDF[650:750]
dfCheck = diabetesDF[750:]

#Separating label and features and converting to numpy array to feed into our model
trainLabel = np.asarray(dfTrain['Outcome'])
trainData = np.asarray(dfTrain.drop('Outcome',1))
testLabel = np.asarray(dfTest['Outcome'])
testData = np.asarray(dfTest.drop('Outcome',1))

# Normalize the data 
means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0)

trainData = (trainData - means)/stds
testData = (testData - means)/stds

# means = np.mean(trainData, axis=0)
# stds = np.std(trainData, axis=0)

#Now , we will use the our training data to 
#create a bayesian classifier.

diabetesCheck = SVC()
diabetesCheck.fit(trainData, trainLabel)

#After we train our bayesian classifier , 
#we test how well it works using our test data.
accuracy = diabetesCheck.score(testData,testLabel)
print("accuracy = ",accuracy * 100,"%")

diabetesCheck = LogisticRegression()
diabetesCheck.fit(trainData,trainLabel)
accuracy = diabetesCheck.score(testData,testLabel)
print("accuracy = ",accuracy * 100,"%")



coeff = list(diabetesCheck.coef_[0])
coeff

labels = list(dfTrain.drop('Outcome',1).columns)
labels

features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')

#model saving and loading
joblib.dump(diabetesCheck, 'diabeteseModel.pkl')
diabetesLoadedModel = joblib.load('diabeteseModel.pkl')

#testing loaded model to make prediction
accuracyModel = diabetesLoadedModel.score(testData,testLabel)
print("accuracy = ",accuracyModel * 100,"%")

dfCheck.head()

sampleData = dfCheck[:1]
sampleDataFeatures = np.asarray(sampleData.drop('Outcome',1))
sampleDataFeatures

prediction = diabetesLoadedModel.predict(sampleDataFeatures)
predictionProbab = diabetesLoadedModel.predict_proba(sampleDataFeatures)

prediction

```
