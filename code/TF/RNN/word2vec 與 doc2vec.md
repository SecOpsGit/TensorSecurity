# doc2vec(2014)
```
Distributed Representations of Sentences and Documents
Quoc V. Le, Tomas Mikolov
(Submitted on 16 May 2014 (v1), last revised 22 May 2014 (this version, v2))
https://arxiv.org/abs/1405.4053
https://cs.stanford.edu/~quocle/paragraph_vector.pdf
```
# Word2vec(2013)
```
Efficient Estimation of Word Representations in Vector Space
Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean
(Submitted on 16 Jan 2013 (v1), last revised 7 Sep 2013 (this version, v3))
https://arxiv.org/abs/1301.3781

We propose two novel model architectures for computing continuous vector representations of words from very large data sets. 
The quality of these representations is measured in a word similarity task, 
and the results are compared to the previously best performing techniques based on different types of neural networks. 

We observe large improvements in accuracy at much lower computational cost, 
i.e. it takes less than a day to learn high quality word vectors from a 1.6 billion words data set. 

Furthermore, we show that these vectors provide state-of-the-art performance on our test set for 
measuring syntactic and semantic word similarities.

https://en.wikipedia.org/wiki/Word2vec
```
```
word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method
Yoav Goldberg, Omer Levy
(Submitted on 15 Feb 2014)
https://arxiv.org/abs/1402.3722

The word2vec software of Tomas Mikolov and colleagues (this https URL ) has gained a lot of traction lately, 
and provides state-of-the-art word embeddings. 

The learning models behind the software are described in two research papers. 

We found the description of the models in these papers to be somewhat cryptic and hard to follow. 

While the motivations and presentation may be obvious to the neural-networks language-modeling crowd, 
we had to struggle quite a bit to figure out the rationale behind the equations.
This note is an attempt to explain equation (4) (negative sampling) 
in "Distributed Representations of Words and Phrases and their Compositionality" 
by Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado and Jeffrey Dean.
```

```
Multi-Class Text Classification with Doc2Vec & Logistic Regression
https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4

```

```
https://github.com/Alex-CHUN-YU/Word2vec
Word2vec 是基於非監督式學習，訓練集建議越大越好，語料涵蓋的越全面，訓練出來的結果相對比較好，
當然也有可能 garbage input 進而得到 garbage output ，
由於檔案所使用的資料集較大，所以每個過程中都請耐心等候。
(ps: word2vec 如果把每種字當成一個維度，假設總共有 4000 個總字，那麼向量就會有 4000 維度。故可透過它來降低維度)
```
```
讀paper之心得：word2vec 與 doc2vec
Eating
Sep 29, 2018

https://medium.com/@ddoo8059/%E8%AE%80paper%E4%B9%8B%E5%BF%83%E5%BE%97-word2vec-%E8%88%87-doc2vec-5c8b8daa7f12
word2vec與doc2vec 分別為2篇paper，都是由Tomas Mikolov提出的
第一篇是word2vec的paper，網址：https://arxiv.org/pdf/1301.3781.pdf
第二篇是doc2vec的paper，網址：https://cs.stanford.edu/~quocle/paragraph_vector.pdf
這篇主要是記錄用，怕之後自己的碩論會用到，順便練習自己的表達能力XD
一. 前言
其實word2vec、doc2vec就是將文字、文檔轉成向量的工具，doc2vec的doc就是document的意思，現在這種網路發達、社群網路蓬勃的時代，從網路抓資料下來分析變得滿重要的，要分析文字又需要一些工具讓電腦可以搞懂我們餵進去的文字是什麼，所以才會有許多將文字、文章等轉成數字、向量的方法，方法其實已經有很多，像是bag-of-words、one-hot represtation、tf-idf等，但在最近好像在word2vec、doc2vec在文字處理上都有不錯的效果，在自然語言處理(NLP)上有很大的進展，其實差不多在兩年前就有聽過word2vec跟doc2vec並實作過，但還真的沒有好好看過這2篇paper，看過之後真的覺得這作者很猛啊XD
二. word2vec
其實文字轉向量這件事在很早之前就有許多人在研究，但過去研究的花費時間過高、架構也很複雜，像是在paper裡面提到的NNLM(Feedforward Neural Net Language Model)，作者就決定提出一個架構簡單，但又能很好表示詞的方法，於是就誕生了word2vec這個東西了，word2vec有2種架構，會在以下分別說明：
1. CBOW(Continuous Bag-of-Words)

CBOW的模型[1]
左圖就是CBOW的模型，其實就是一個類神經網路的模型，只是只有一層隱藏層而已，概念就是利用上下文來預測中間的詞出現的機率是多少，以左圖來說的話可以當成現在有五個詞，利用前兩個詞(w(t-1)跟w(t-2))與後兩個詞(w(t+1)跟w(t+2))來預測中間詞(w(t))出現的機率，輸入及輸出層是以one-hot representation來表示的，不知道大家會不會覺得很奇怪，明明就說是預測機率，那跟詞向量又有什麼關係，在經過我不斷google之下，終於知道了，預測中間詞的機率根本不是重點，重點是input到projection的權重啊！！
原來權重就是詞向量！！預測中間詞的機率主要是用來調權重用的，在預測時因為希望預測的機率越大越好，會藉由神經網路常用調整權重的方法SGD以及Backpropagation來調整權重，最後調整的權重就是詞向量矩陣了！！
說到這邊還是舉個例子來說明好了，假如今天我們只有一句話而且就是語料庫： “我 今天 很 帥”(要先經過斷詞唷~~)，我們會利用 “我” 及 “很” 來預測 “今天” 出現的機率，這邊 “我 ”會被表示one-hot representation的形式，也就是[1,0,0,0]，“很”會表示成[0,0,1,0]，“今天”會表示成[0,1,0,0]，權重一開始的值是隨機分配的，假設我們設定詞向量只要三維的話，那權重矩陣的大小就是4x3，projection的神經元數目也會被定為3，假設經過神經網路迭代運算後我們得到一個權重矩陣w，那我們今天要來看 “今天” 對應到的詞向量，會變成下圖所示：

各位可以想一下，如果今天有2句話 “我 今天 很 帥” 跟 “我 今天 滿 帥” ，那 ”很“ 跟 ”滿“ 這2個的詞向量在訓練時就會很接近，有沒有覺得作者的想法很厲害XD。CBOW的架構就先說到這，其實我後面也已經把詞向量如何訓練的一起說明了。
2. skip-gram
第二個word2vec的模型是skip-gram，架構圖如下：

skip-gram的模型[1]
其實詞向量訓練的方式跟CBOW是一樣的，輸入及輸出層也都是one-hot representation的形式，但整體概念是用中間詞來訓練上下文出現詞的機率，跟CBOW剛好相反。以剛剛的句子“我 今天 很 帥”，就是我用 “今天” 來預測 “我” 及 “很”這2個詞出現的機率，在最大化機率的過程中也訓練了從input層到到projection層的權重(也就是詞向量矩陣!!)
關於詞向量的2個模型的介紹就到這裡了，下面會說明doc2vec，也就是如何將文檔、句子等轉成向量的方式
三. doc2vec
作者當初提出這個文檔轉向量的方法，主要是因為在這之前並沒有一個很好表示句子的方法，過去通常都是用bag-of-words的方法來表示句子或文檔，但作者有提到這樣的表示法並沒有考慮字詞的順序，各位仔細想想其實真的是這樣，很久之前聽李宏毅教授的word embedding的教學時，裡面有提到bag-of-words，教授就舉出了 “ White blood cells destroying an infection” 以及 “ An infection destroying white blood cells” 這2個句子是正負兩面的例子，但若以bag-of-words表示的話都會是一樣的表示式，這也就是為什麼bag-of-words會喪失詞的順序這樣的資訊，那doc2vec為什麼可以解決這樣的問題，因為他其實只是衍生word2vec的想法，word2vec的訓練過程中就是有考慮到了詞的順序性，所以才能將相似的詞聚在一起，doc2vec也是分兩個模型，以下會介紹：
1. PV-DM(Paragraph Vector: A distributed memory model)
PV-DM的模型如下圖，不覺得與剛剛word2vec的CBOW模型很像嗎？

PV-DM模型[2]
對的，這個PV-DM其實就是CBOW衍生出來的模型啊，他多出了一個D矩陣，也就是代表每個文檔的矩陣啊！！原文是說讓這個文檔矩陣也參與預測詞的任務，這邊注意一下，PV-DM的模型是利用前面的詞來預測下一個出現詞的機率，以上圖來說，就是利用 “the” 、 “cat” 及 “set” 來預測 “on” 出現的機率，並且讓一個文檔矩陣來參與這個任務！圖中的W其實就是權重(以剛剛word2vec的模型來說就是input層到projection層的權重)，其實D跟W是一樣的，只是D是文檔的權重。
接著來說明一下訓練方式與word2vec的不同，其實不同點只是在於多了一個文檔的權重D，以及在訓練時同一個文檔，文檔的paragraph id是固定的，以上面的例子來說，假設這句話是 “the cat sat on the chair” ，在訓練這句話時這個paragraph id 是固定的，所以當預測結束 “on”出現的機率時，會變成 “cat” 、 “sat” 及 “on” 來預測該句的第二個 “the”，但paragraph id不會變，直到訓練完這句話為止，所以每個詞都會保留一樣的向量，並另外得到了文檔向量矩陣的資訊，用這個文檔向量矩陣便能得出每一個文檔的向量了！！
2. DBOW(Distributed bag of words):
一樣，先上模型的圖，如下：

DBOW的模型[2]
看到這張圖，不覺得DBOW根本就是word2vec的skip-gram模型嗎？沒錯，他確實是skip-gram的衍生，只是不一樣的點就是他是利用文檔的矩陣來預測該文檔每個詞出現的機率！！訓練方式其實跟skip-gram是一樣的，只使初始的權重是文檔的權重。
那我介紹word2vec與doc2vec的部分就到這邊，如果覺得我說錯或不清楚的都可以直接跟我說唷！！
四. 參考資料
```

```

```
