# 資料來源
```
https://zh.wikipedia.org/wiki/Tf-idf
http://www.cc.ntu.edu.tw/chinese/epaper/0031/20141220_3103.html
```

# TF-IDF（Term Frequency - Inverse Document Frequency）
```
TF-IDF 是一種用於資訊檢索與文字探勘的常用加權技術，為一種統計方法，
用來評估單詞(word)對於文件的集合或詞庫中一份文件的重要程度
```
## TF（Term Frequency）：
```
假設 j 是「某一特定文件」，i 是該文件中所使用單詞或單字的「其中一種」，
n(i,j) 就是 i 在 j 當中的「出現次數」

tf(i,j) 的算法就是 n(i,j) / (n(1,j)+n(2,j)+n(3,j)+…+n(i,j))。

例如第一篇文件中，被我們篩選出兩個重要名詞，分別為「健康」、「富有」，
「健康」在該篇文件中出現 70 次，「富有」出現 30 次，
「健康」的 tf = 70 / (70+30) = 70/100 = 0.7，
「富有」的 tf = 30 / (70+30) = 30/100 = 0.3；

在第二篇文件裡，同樣篩選出兩個名詞，分別為「健康」、「富有」
「健康」在該篇文件中出現 40 次，「富有」出現 60 次，
「健康」的 tf = 40 / (40+60) = 40/100 = 0.4，
「富有」的 tf = 60 / (40+60) = 60/100 = 0.6，

tf 值愈高，其單詞愈重要。
「健康」對第一篇文件比較重要，「富有」對第二篇文件比較重要。
若搜尋「健康」，那第一篇文件會在較前面的位置；
而搜尋「富有」，則第二篇文章會出現在較前面的位置。
```
##  IDF（Inverse Document Frequency）：
```
假設 D 是「所有的文件總數」，i 是網頁中所使用的單詞，
t(i) 是該單詞(word)在所有文件總數中出現的「文件數」
idf(i) 的算法就是 log ( D/t(i) ) = log D – log t(i)。

例如有 100 個網頁，
「健康」出現在 10 個網頁當中，
「富有」出現在 100 個網頁當中

「健康」的 idf = log ( 100/10 ) = log 100 – log 10 = 2 – 1 = 1
「富有」的 idf = log (100/100) = log 100 – 1og 100 = 2 – 2 = 0。

所以，「健康」出現的機會小，與出現機會很大的「富有」比較起來，便顯得非常重要。
```
## TF-IDF
```
TF-IDF 權重值 == tf(i,j) * idf(i)（例如：i =「健康」一詞）來進行計算，

以某一特定文件內的高單詞頻率，乘上該單詞在文件總數中的低文件頻率，便可以產生 TF-IDF 權重值，

TF-IDF 傾向於過濾掉常見的單詞，保留重要的單詞，如此一來，「富有」便不重要了。
```

# 常見應用
```
1.分析開放式調查研究的回應結果：
常見於行銷方面，其觀點在於允許回應者在不受特定面向與回應格式的侷限，
來表達他們自身的觀點與意見。
例如使用者在填寫問卷後，透過在單詞上的使用，來分析他們對產品或服務的評價與感受，
來為其體驗上的不足、誤解以及困惑，提供建議與解答，同時分析其消費行為，
進一步做到客戶分群，並擬定及提供客製化的服務給消費者。

2. 訊息、電子郵件的自動化處理：
例如過濾客戶的意見回饋，來了解信件的內容是正面或負面的評價，
或是過濾電子郵件，來了解是否為垃圾郵件，
若是合法郵件，也可自動分析此信件的訴求是屬於哪個部門該處理的業務範圍。

3.分析產品保固、保險金請求，以及診斷面談等內容：
在保固期間內，若產品有非人為因素的損壞，通常可以依保固項目來進行送件維修，
送件時需要填寫一份維修單，廠商會透過使用者的口述來進行損壞原因註記，
透過註記的電子化與分析，為產品日後改版或強化，提供很好的建議；

在保險金請求時，保險業者會進行記錄，並將請求內容進行分析，作為日後提供保險服務的依據與參考；
透過與病人會晤與交談，了解病患的需求與病癥，進而分析成有用的資訊，作為日後提供診療的參考。

4. 經由網路爬蟲（Crawler）來擷取、調查競爭對手的網站：
前往競爭對手的主要網站，來「爬」遍所有的內容，自動建立可用的單詞與文件列表，
為他們所描述和強調的部分進行探勘，便能輕易地取得競爭對手有價值的商業智慧與邏輯。
```
# Scikit-Learn實現的TF-IDF
```
Convert a collection of raw documents to a matrix of TF-IDF features.
Equivalent to CountVectorizer followed by TfidfTransformer
```
```
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

class sklearn.feature_extraction.text.TfidfVectorizer(
input=’content’, 
encoding=’utf-8’,  #  解碼
decode_error=’strict’, 
   #三種選擇:{'strict', 'ignore', 'replace'}
   #如果一個給出的位元組序列包含的字元不是給定的編碼，指示應該如何去做。
   #預設情況下，它是'strict'，這意味著的UnicodeDecodeError將提高，其他值是'ignore'和'replace'

strip_accents=None, 
lowercase=True, 
preprocessor=None, 
tokenizer=None, 
analyzer=’word’, 
stop_words=None, 
token_pattern=’(?u)\b\w\w+\b’, 
ngram_range=(1, 1), 
max_df=1.0, 
min_df=1, 
max_features=None, 
vocabulary=None, 
binary=False, 
dtype=<class ‘numpy.float64’>, 
norm=’l2’, 
use_idf=True, 
smooth_idf=True, 
sublinear_tf=False)
```
```
https://blog.csdn.net/laobai1015/article/details/80451371
```
### 可用的Methods
```
build_analyzer(self)	Return a callable that handles preprocessing and tokenization
build_preprocessor(self)	Return a function to preprocess the text before tokenization
build_tokenizer(self)	Return a function that splits a string into a sequence of tokens
decode(self, doc)	Decode the input into a string of unicode symbols

fit(self, raw_documents[, y])	Learn vocabulary and idf from training set.
fit_transform(self, raw_documents[, y])	Learn vocabulary and idf, return term-document matrix.

get_feature_names(self)	Array mapping from feature integer indices to feature name
get_params(self[, deep])	Get parameters for this estimator.
get_stop_words(self)	Build or fetch the effective stop words list

inverse_transform(self, X)	Return terms per document with nonzero entries in X.

set_params(self, \*\*params)	Set the parameters of this estimator.

transform(self, raw_documents[, copy])	Transform documents to document-term matrix.
```

# 使用Scikit-Learn實現的TF-IDF
```
TF-IDF演算法解析與Python實現方法詳解
https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/363018/
```
```
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

corpus = ['This is the first document.',
'This is the second second document.',
'And the third one.',
'Is this the first document?',]

vectorizer = TfidfVectorizer(min_df=1)

vectorizer.fit_transform(corpus)

vectorizer.get_feature_names()
```
```
['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
```
```
vectorizer.fit_transform(corpus).toarray()
```
```
rray([[0. , 0.43877674, 0.54197657, 0.43877674, 0. , 0. , 0.35872874, 0., 0.43877674],
       [0.  , 0.27230147, 0. , 0.27230147, 0.    , 0.85322574, 0.22262429, 0.  , 0.27230147],
       [0.55280532, 0. , 0.   , 0. , 0.55280532,  0.  , 0.28847675, 0.55280532, 0.  ],
       [0., 0.43877674, 0.54197657, 0.43877674, 0.   , 0.  , 0.35872874, 0.  , 0.43877674]])
```
```
最終的結果是一個 4×9 矩陣。
每行表示一個文件，每列表示該文件中的每個詞的評分。

如果某個詞沒有出現在該文件中，則相應位置就為 0 。
數字 9 表示語料庫裡詞彙表中一共有 9 個（不同的）詞。
例如，你可以看到在文件1中，並沒有出現 and，所以矩陣第一行第一列的值為 0 。
單詞 first 只在文件1中出現過，所以第一行中 first 這個詞的權重較高。
而 document 和 this 在 3 個文件中出現過，所以它們的權重較低。
而 the 在 4 個文件中出現過，所以它的權重最低。
```


# 機器學習應用-「垃圾訊息偵測」與「TF-IDF介紹」(含範例程式)
```
https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E6%87%89%E7%94%A8-%E5%9E%83%E5%9C%BE%E8%A8%8A%E6%81%AF%E5%81%B5%E6%B8%AC-%E8%88%87-tf-idf%E4%BB%8B%E7%B4%B9-%E5%90%AB%E7%AF%84%E4%BE%8B%E7%A8%8B%E5%BC%8F-2cddc7f7b2c5
```
### 下載資料
```
!wget https://raw.githubusercontent.com/MyDearGreatTeacher/AI201909/master/data/spam.csv
```
### 範例程式
```
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:21:44 2018
Machine learning Example for「SMS Spam Collection Dataset」
Database Link:
https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
https://www.kaggle.com/uciml/sms-spam-collection-dataset#spam.csv
 
@author: Tommy Huang, chih.sheng.huang821@gmail.com
"""

# 1.	首先我們先將資料匯入python內，我們會用到的pandas，pandas對處理這種文字資料滿好用的。
import pandas as pd
#filepath='C:\\Users\\user\\Desktop/spam.csv'
filepath='spam.csv'
def readData_rawSMS(filepath):
	data_rawSMS = pd.read_csv(filepath,usecols=[0,1],encoding='latin-1')
	data_rawSMS.columns=['label','content']
	return data_rawSMS
data_rawSMS = readData_rawSMS(filepath)
#########################################
# kaggle的'spam.csv'將我範例非垃圾郵件的label寫的genuine改成ham
# 所以如果要直接用我的程式，最簡單的方式就是ham改回genuine
# 2019/02/27修改這段
for i in range(data_rawSMS.shape[0]):
    if data_rawSMS.iloc[i].label == 'ham':
        data_rawSMS.iloc[i].label='genuine'
###########################################	
#2.	將資料分成Train和Test
import numpy as np
def Separate_TrainAndTest(data_rawSMS):
    n=int(data_rawSMS.shape[0])
    tmp_train=(np.random.rand(n)>=0.5)
    return data_rawSMS.iloc[np.where(tmp_train==True)[0]], data_rawSMS.iloc[np.where(tmp_train==False)[0]]
data_rawtrain,data_rawtest=Separate_TrainAndTest(data_rawSMS)

#3. 從training data去著手算哪些「詞」重要。
import re
def generate_key_list(data_rawtrain, size_table=200,ignore=3):
    dict_spam_raw = dict()
    dict_genuine_raw = dict()
    dict_IDF = dict()

	# ignore all other than letters.
    for i in range(data_rawSMS.shape[0]):
        finds = re.findall('[A-Za-z]+', data_rawSMS.iloc[i].content)
        if data_rawSMS.iloc[i].label == 'spam':
            for find in finds:
                if len(find)<ignore: continue
                find = find.lower() #英文轉成小寫
                try:
                    dict_spam_raw[find] = dict_spam_raw[find] + 1
                except:	
                    dict_spam_raw[find] = dict_spam_raw.get(find,1)
                    dict_genuine_raw[find] = dict_genuine_raw.get(find,0)
        else:
            for find in finds:
                if len(find)<ignore: continue
                find = find.lower()
                try:
                    dict_genuine_raw[find] = dict_genuine_raw[find] + 1
                except:	
                    dict_genuine_raw[find] = dict_genuine_raw.get(find,1)
                    dict_spam_raw[find] = dict_spam_raw.get(find,0)
		
        word_set = set()
        for find in finds:
            if len(find)<ignore: continue
            find = find.lower()
            if not(find in word_set):
                try:
                    dict_IDF[find] = dict_IDF[find] + 1
                except:	
                    dict_IDF[find] = dict_IDF.get(find,1)
            word_set.add(find)
    word_df = pd.DataFrame(list(zip(dict_genuine_raw.keys(),dict_genuine_raw.values(),dict_spam_raw.values(),dict_IDF.values())))
    word_df.columns = ['keyword','genuine','spam','IDF']
    word_df['genuine'] = word_df['genuine'].astype('float')/data_rawtrain[data_rawtrain['label']=='genuine'].shape[0]
    word_df['spam'] = word_df['spam'].astype('float')/data_rawtrain[data_rawtrain['label']=='spam'].shape[0]
    word_df['IDF'] = np.log10(word_df.shape[0]/word_df['IDF'].astype('float'))
    word_df['genuine_IDF'] = word_df['genuine']*word_df['IDF']
    word_df['spam_IDF'] = word_df['spam']*word_df['IDF']
    word_df['diff']=word_df['spam_IDF']-word_df['genuine_IDF']
    selected_spam_key = word_df.sort_values('diff',ascending=False)  
    keyword_dict = dict()
    i = 0
    for word in selected_spam_key.head(size_table).keyword:
        keyword_dict.update({word.strip():i})
        i+=1
    return keyword_dict   

# build a tabu list based on the training data
size_table = 300                 # how many features are used to classify spam
word_len_ignored = 3            # ignore those words shorter than this variable
keyword_dict=generate_key_list(data_rawtrain, size_table, word_len_ignored)


# 4.將Train資料和Test資料轉換成特徵向量
def convert_Content(content, keyword_dict):
	m = len(keyword_dict)
	res = np.int_(np.zeros(m))
	finds = re.findall('[A-Za-z]+', content)
	for find in finds:
		find=find.lower()
		try:
			i = keyword_dict[find]
			res[i]=1
		except:
			continue
	return res

def raw2feature(data_rawtrain,data_rawtest,keyword_dict):
    n_train = data_rawtrain.shape[0]
    n_test = data_rawtest.shape[0]
    m = len(keyword_dict)
    X_train = np.zeros((n_train,m));
    X_test = np.zeros((n_test,m));
    Y_train = np.int_(data_rawtrain.label=='spam')
    Y_test = np.int_(data_rawtest.label=='spam')
    for i in range(n_train):
        X_train[i,:] = convert_Content(data_rawtrain.iloc[i].content, keyword_dict)
    for i in range(n_test):
        X_test[i,:] = convert_Content(data_rawtest.iloc[i].content, keyword_dict)
        
    return [X_train,Y_train],[X_test,Y_test]
     
Train,Test=raw2feature(data_rawtrain,data_rawtest,keyword_dict)


# 5.	依據特徵資料訓練分類器
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB       
def learn(Train):
    model_NB = BernoulliNB()
    model_NB.fit(Train[0], Train[1])
    Y_hat_NB = model_NB.predict(Train[0])

    model_RF = RandomForestClassifier(n_estimators=10, max_depth=None,\
                                 min_samples_split=2, random_state=0)
    model_RF.fit(Train[0], Train[1])
    Y_hat_RF = model_RF.predict(Train[0])
    
    n=np.size(Train[1])
    print('Training Accuarcy NBclassifier : {:.2f}％'.format(sum(np.int_(Y_hat_NB==Train[1]))*100./n))
    print('Training Accuarcy RF: {:.2f}％'.format(sum(np.int_(Y_hat_RF==Train[1]))*100./n))
    return model_NB,model_RF
# train the Random Forest and the Naive Bayes Model using training data
model_NB,model_RF=learn(Train)

# 6.依據訓練好的分類器，進行測試。
def test(Test,model):
    Y_hat = model.predict(Test[0])
    n=np.size(Test[1])
    print ('Testing Accuarcy: {:.2f}％ ({})'.format(sum(np.int_(Y_hat==Test[1]))*100./n,model.__module__))
# Test Model using testing data
test(Test,model_NB)
test(Test,model_RF)

#######
def predictSMS(SMS,model,keyword_dict):
    X = convert_Content(SMS, keyword_dict)
    Y_hat = model.predict(X.reshape(1,-1))
    if int(Y_hat) == 1:
        print ('SPAM: {}'.format(SMS))
    else:
        print ('GENUINE: {}'.format(SMS))    

inputstr='go to visit www.yahoo.com.tw, Buy one get one free, Hurry!'
predictSMS(inputstr,model_NB,keyword_dict)

inputstr=('Call back for anytime.')
predictSMS(inputstr,model_NB,keyword_dict)
```

### 執行成果
```
Training Accuarcy NBclassifier : 98.11％
Training Accuarcy RF: 99.49％
Testing Accuarcy: 97.76％ (sklearn.naive_bayes)
Testing Accuarcy: 96.45％ (sklearn.ensemble.forest)
GENUINE: go to visit www.yahoo.com.tw, Buy one get one free, Hurry!
GENUINE: Call back for anytime.
```
### 關鍵程式解說

### 1.資料匯入
```
# -*- coding: utf-8 -*-
import pandas as pd

filepath='spam.csv'

def readData_rawSMS(filepath):
	data_rawSMS = pd.read_csv(filepath,usecols=[0,1],encoding='latin-1')
	data_rawSMS.columns=['label','content']
	return data_rawSMS
  
data_rawSMS = readData_rawSMS(filepath)
#########################################
# kaggle的'spam.csv'將我範例非垃圾郵件的label寫的genuine改成ham
# 所以如果要直接用我的程式，最簡單的方式就是ham改回genuine
# 2019/02/27修改這段
for i in range(data_rawSMS.shape[0]):
    if data_rawSMS.iloc[i].label == 'ham':
        data_rawSMS.iloc[i].label='genuine'
###########################################	
```
### 2.	將資料分成Train和Test
```
import numpy as np
def Separate_TrainAndTest(data_rawSMS):
    n=int(data_rawSMS.shape[0])
    tmp_train=(np.random.rand(n)>=0.5)
    return data_rawSMS.iloc[np.where(tmp_train==True)[0]], data_rawSMS.iloc[np.where(tmp_train==False)[0]]
data_rawtrain,data_rawtest=Separate_TrainAndTest(data_rawSMS)
```
### 3. 從training data去著手算哪些「詞」重要。
```
import re
def generate_key_list(data_rawtrain, size_table=200,ignore=3):
    dict_spam_raw = dict()
    dict_genuine_raw = dict()
    dict_IDF = dict()

	# ignore all other than letters.
    for i in range(data_rawSMS.shape[0]):
    
        finds = re.findall('[A-Za-z]+', data_rawSMS.iloc[i].content)
        
        if data_rawSMS.iloc[i].label == 'spam':
            for find in finds:
                if len(find)<ignore: continue
                find = find.lower() #英文轉成小寫
                
                try:
                    dict_spam_raw[find] = dict_spam_raw[find] + 1
                except:	
                    dict_spam_raw[find] = dict_spam_raw.get(find,1)
                    dict_genuine_raw[find] = dict_genuine_raw.get(find,0)
        else:
            for find in finds:
                if len(find)<ignore: continue
                find = find.lower()
                try:
                    dict_genuine_raw[find] = dict_genuine_raw[find] + 1
                except:	
                    dict_genuine_raw[find] = dict_genuine_raw.get(find,1)
                    dict_spam_raw[find] = dict_spam_raw.get(find,0)
		
        word_set = set()
        
        for find in finds:
            if len(find)<ignore: continue
            find = find.lower()
            if not(find in word_set):
                try:
                    dict_IDF[find] = dict_IDF[find] + 1
                except:	
                    dict_IDF[find] = dict_IDF.get(find,1)
            word_set.add(find)
            
    word_df = pd.DataFrame(list(zip(dict_genuine_raw.keys(),dict_genuine_raw.values(),dict_spam_raw.values(),dict_IDF.values())))
    
    word_df.columns = ['keyword','genuine','spam','IDF']
   
   word_df['genuine'] = word_df['genuine'].astype('float')/data_rawtrain[data_rawtrain['label']=='genuine'].shape[0]
    word_df['spam'] = word_df['spam'].astype('float')/data_rawtrain[data_rawtrain['label']=='spam'].shape[0]
    word_df['IDF'] = np.log10(word_df.shape[0]/word_df['IDF'].astype('float'))
    word_df['genuine_IDF'] = word_df['genuine']*word_df['IDF']
    word_df['spam_IDF'] = word_df['spam']*word_df['IDF']
    word_df['diff']=word_df['spam_IDF']-word_df['genuine_IDF']
   
   selected_spam_key = word_df.sort_values('diff',ascending=False)  
    
    keyword_dict = dict()
    
    i = 0
    for word in selected_spam_key.head(size_table).keyword:
        keyword_dict.update({word.strip():i})
        i+=1
    return keyword_dict   

# build a tabu list based on the training data
size_table = 300                 # how many features are used to classify spam
word_len_ignored = 3            # ignore those words shorter than this variable
keyword_dict=generate_key_list(data_rawtrain, size_table, word_len_ignored)
```
### 4.將Train資料和Test資料轉換成特徵向量
```
def convert_Content(content, keyword_dict):
	m = len(keyword_dict)
	res = np.int_(np.zeros(m))
	finds = re.findall('[A-Za-z]+', content)
	for find in finds:
		find=find.lower()
		try:
			i = keyword_dict[find]
			res[i]=1
		except:
			continue
	return res

def raw2feature(data_rawtrain,data_rawtest,keyword_dict):
    n_train = data_rawtrain.shape[0]
    n_test = data_rawtest.shape[0]
    m = len(keyword_dict)
    X_train = np.zeros((n_train,m));
    X_test = np.zeros((n_test,m));
    Y_train = np.int_(data_rawtrain.label=='spam')
    Y_test = np.int_(data_rawtest.label=='spam')
    for i in range(n_train):
        X_train[i,:] = convert_Content(data_rawtrain.iloc[i].content, keyword_dict)
    for i in range(n_test):
        X_test[i,:] = convert_Content(data_rawtest.iloc[i].content, keyword_dict)
        
    return [X_train,Y_train],[X_test,Y_test]
     
Train,Test=raw2feature(data_rawtrain,data_rawtest,keyword_dict)
```
### 5.依據特徵資料訓練分類器
```
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB       

def learn(Train):
    model_NB = BernoulliNB()
    model_NB.fit(Train[0], Train[1])
    Y_hat_NB = model_NB.predict(Train[0])

    model_RF = RandomForestClassifier(n_estimators=10, max_depth=None,\
                                 min_samples_split=2, random_state=0)
    model_RF.fit(Train[0], Train[1])
    Y_hat_RF = model_RF.predict(Train[0])
    
    n=np.size(Train[1])
    print('Training Accuarcy NBclassifier : {:.2f}％'.format(sum(np.int_(Y_hat_NB==Train[1]))*100./n))
    print('Training Accuarcy RF: {:.2f}％'.format(sum(np.int_(Y_hat_RF==Train[1]))*100./n))
    return model_NB,model_RF
    
# train the Random Forest and the Naive Bayes Model using training data
model_NB,model_RF=learn(Train)
```
### 6.依據訓練好的分類器，進行測試。
```
def test(Test,model):
    Y_hat = model.predict(Test[0])
    n=np.size(Test[1])
    print ('Testing Accuarcy: {:.2f}％ ({})'.format(sum(np.int_(Y_hat==Test[1]))*100./n,model.__module__))
    
# Test Model using testing data
test(Test,model_NB)
test(Test,model_RF)
```
#### 定義預測函數與測試
```
def predictSMS(SMS,model,keyword_dict):
    X = convert_Content(SMS, keyword_dict)
    Y_hat = model.predict(X.reshape(1,-1))
    if int(Y_hat) == 1:
        print ('SPAM: {}'.format(SMS))
    else:
        print ('GENUINE: {}'.format(SMS))    

inputstr='go to visit www.yahoo.com.tw, Buy one get one free, Hurry!'
predictSMS(inputstr,model_NB,keyword_dict)

inputstr=('Call back for anytime.')
predictSMS(inputstr,model_NB,keyword_dict)
```
