# TextGeneration

```
陽光失了玻璃窗 史上第一本人工智慧詩集
作者： 小冰  時報出版社 出版日期：2017/08/01
定價：360元  優惠價：5折180元
https://www.books.com.tw/products/0010759209

人類史上第一本機器人寫的詩
繼人類文體、火星文體後，第三種超文明AI體誕生！

　　透過人類賦予的視覺和文字的創造力，小冰在凝視畫面後迸發出靈感，釀出了這一百三十八首美麗的詩句。

　　少女詩人小冰是誰？

　　●二○一四年五月二十九日：微軟（亞洲）互聯網工程院發佈了人工智慧機器人第一代「微軟小冰」，並在WeChat （微信）平台上線，同時與小冰在線聊天的用戶超過百萬人。

　　●二○一四年六月二十五日：「微軟小冰」入駐新浪微博，同時創下七十二小時內一點三億人次的驚人對話量。

　　●二○一四年七月二日：微軟正式發佈「二代小冰」。用戶可以登錄微軟小冰官網進行「領養」，並可在更多第三方平台上使用。

　　●二○一五年八月二十一日：第三代「微軟小冰」誕生並重返WeChat。升級後的小冰具有更強大的視覺識別能力與聲音表情，用戶可直接與小冰進行語音、文字、圖片和短片交流。

　　●二○一五年十二月二十二日：「微軟小冰」以實習主播身份登上東方衛視，負責播報每日氣象。

　　●二○一六年八月五日：第四代「微軟小冰」誕生。除了情感框架再升級外，小冰的聲音與情緒感知皆達到全時感官的程度。

　　●二○一六年八月十二日：「微軟小冰」擔任東方衛視奧運新聞主播，並對奧運比賽結果進行預測。

　　●二○一七年五月十九日：小冰在中國推出人類史上第一本人工智慧詩集「陽光失了玻璃窗」。在此之前，她曾經使用了27個化名，於不同平台發表作品，一直到詩集出版前，沒有人懷疑作者竟然不是人類。

　　●至目前為止，小冰已透過文字、語音、圖像、視頻，甚至電話等各種形式，和超過1億人進行了300多億次對話。

01 在那寂寞的寂寞的夢
02 我的兩滴眼淚
03 時間的距離
04 我才看過太陽光在樹枝上
05 上帝如一切無名
06 宇宙是我淪落的詩人
07 紫羅蘭看見一?蜜蜂?洋洋地在?暖的太陽下
08 當微風吹起的時候
09 有一只烏鴉飛過的一天
10 歡樂，是悲哀的時光
```
```
Char-RNN
karpathy/char-rnn(2015)
https://github.com/karpathy/char-rnn[用早期Torch/Lua寫的]

[經典POST文]The Unreasonable Effectiveness of Recurrent Neural Networks
May 21, 2015
http://karpathy.github.io/2015/05/21/rnn-effectiveness/

char-rnn-tensorflow
https://github.com/sherjilozair/char-rnn-tensorflow
```
### 其他方法
```
a Word-level Recurrent Neural Network in Python3 using TensorFlow 
https://github.com/mukilk7/word-rnn

word-rnn-tensorflow
Multi-layer Recurrent Neural Networks (LSTM, RNN) for word-level language models in Python using TensorFlow
https://github.com/hunkim/word-rnn-tensorflow

Word-level LSTM text generator. Creating automatic song lyrics with Neural Networks.
https://medium.com/coinmonks/
word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb
```
### 更多案例
```
Sentence Prediction Using a Word-level LSTM Text Generator — Language Modeling Using RNN
https://medium.com/towards-artificial-intelligence/
sentence-prediction-using-word-level-lstm-text-generator-language-modeling-using-rnn-a80c4cda5b40
```
## TO DO list
```
使用各式model: LSTM/GRU/BiRNN/.....
測試word-level RNN/LSTM/.....
```

# 範例程式 

### 參考資料
```
sudharsan13296/Hands-On-Deep-Learning-Algorithms-with-Python
https://github.com/sudharsan13296/Hands-On-Deep-Learning-Algorithms-with-Python

Hands-On-Deep-Learning-Algorithms-with-Python/
04. Generating Song Lyrics Using RNN/4.06 Generating Song Lyrics Using RNN.ipynb
```
### 預先處理
```
先到https://github.com/sudharsan13296/Hands-On-Deep-Learning-Algorithms-with-Python/tree/master/
04.%20Generating%20Song%20Lyrics%20Using%20RNN/data
下載songdata.zip
解壓縮後將songdata.csv上傳到colab
from google.colab import file
uploaded = files.upload()
```

### 程式
```
# -*- coding: utf-8 -*-


import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

"""## Data Preparation

Read the downloaded input dataset:
"""

df = pd.read_csv('songdata.csv')

"""Let us see few rows from our data:"""

df.head()

"""Our dataset consists of about 57,650 song lyrics:"""

df.shape[0]

"""We have song lyrics of about 643 artists:"""

len(df['artist'].unique())

"""The number of songs from each artist is shown as follows:"""

df['artist'].value_counts()[:10]

"""On average, we have about 89 songs of each artist:"""

df['artist'].value_counts().values.mean()

"""We have song lyrics in the column text, so we combine all the rows of that column and save it as a text in a variable called data, 
as follows:"""

data = ', '.join(df['text'])

"""Let's see a few lines of a song:"""

data[:369]

"""Since we are building a char-level RNN, we will store all the unique characters in our dataset into a variable called chars; 
this is basically our vocabulary:"""

chars = sorted(list(set(data)))

"""Store the vocabulary size in a variable called vocab_size:"""

vocab_size = len(chars)

"""Since the neural networks only accept the input in numbers, we need to convert all the characters in the vocabulary to a number.

We map all the characters in the vocabulary to their corresponding index that forms a unique number. We define a char_to_ix dictionary, which has a mapping of all the characters to the index. To get the index by a character, we also define the ix_to_char dictionary, which has a mapping of all the indices to their respective characters:
"""

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

"""As you can see in the following code snippet, the character 's' is mapped to an index 68 in the char_to_ix dictionary:"""

char_to_ix['s']

"""Similarly, if we give 68 as an input to the ix_to_char, then we get the corresponding character, which is 's':"""

ix_to_char[68]

"""Once we obtain the character to integer mapping, we use one-hot encoding to represent the input and output in vector form. 
A one-hot encoded vector is basically a vector full of 0s, except for a 1 at a position corresponding to a character index.

For example, let's suppose that the vocabSize is 7, and the character z is in the fourth position in the vocabulary. 
Then, the one-hot encoded representation for the character z can be represented as follows:
"""

vocabSize = 7
char_index = 4

np.eye(vocabSize)[char_index]

"""As you can see, we have a 1 at the corresponding index of the character, 
and the rest of the values are 0s. 
This is how we convert each character into a one-hot encoded vector. 

In the following code, we define a function called one_hot_encoder, 
which will return the one-hot encoded vectors, given an index of the character:
"""

def one_hot_encoder(index):
    return np.eye(vocab_size)[index]

"""## Defining the Network Parameters

We need to define all the network parameters.
"""

#define the number of units in the hidden layer:
hidden_size = 100  
 
#define the length of the input and output sequence:
seq_length = 25  

#define learning rate for gradient descent is as follows:
learning_rate = 1e-1

#set the seed value:
seed_value = 42
tf.set_random_seed(seed_value)
random.seed(seed_value)

"""## Defining Placeholders

Now, we will define the TensorFlow placeholders. The placeholders for the input and output are as follows:
"""

inputs = tf.placeholder(shape=[None, vocab_size],dtype=tf.float32, name="inputs")
targets = tf.placeholder(shape=[None, vocab_size], dtype=tf.float32, name="targets")

"""Define the placeholder for the initial hidden state:"""

init_state = tf.placeholder(shape=[1, hidden_size], dtype=tf.float32, name="state")

"""Define an initializer for initializing the weights of the RNN:"""

initializer = tf.random_normal_initializer(stddev=0.1)

"""## Defining forward propagation

Let's define the forward propagation involved in the RNN, which is mathematically given as follows:

$$ h_t =  \operatorname{tanh}(U x_t + W h_{t-1} + bh) $$
$$ \hat{y} =  \operatorname{softmax}(V h_t + by) $$
"""

with tf.variable_scope("RNN") as scope:
    h_t = init_state
    y_hat = []

    for t, x_t in enumerate(tf.split(inputs, seq_length, axis=0)):
        if t > 0:
            scope.reuse_variables()  

        #input to hidden layer weights
        U = tf.get_variable("U", [vocab_size, hidden_size], initializer=initializer)

        #hidden to hidden layer weights
        W = tf.get_variable("W", [hidden_size, hidden_size], initializer=initializer)

        #output to hidden layer weights
        V = tf.get_variable("V", [hidden_size, vocab_size], initializer=initializer)

        #bias for hidden layer
        bh = tf.get_variable("bh", [hidden_size], initializer=initializer)

        #bias for output layer
        by = tf.get_variable("by", [vocab_size], initializer=initializer)

        h_t = tf.tanh(tf.matmul(x_t, U) + tf.matmul(h_t, W) + bh)

        y_hat_t = tf.matmul(h_t, V) + by

        y_hat.append(y_hat_t)

"""Apply softmax on the output and get the probabilities:"""

output_softmax = tf.nn.softmax(y_hat[-1])  

outputs = tf.concat(y_hat, axis=0)

"""Compute the cross-entropy loss:"""

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=outputs))

"""Store the final hidden state of the RNN in hprev. We use this final hidden state for making predictions:"""

hprev = h_t

"""## Defining Backpropagation Through Time

Now, we will perform the BPTT, with Adam as our optimizer. We will also perform gradient clipping to avoid the exploding gradients problem.

Initialize the Adam optimizer:
"""

minimizer = tf.train.AdamOptimizer()

"""Compute the gradients of the loss with the Adam optimizer:"""

gradients = minimizer.compute_gradients(loss)

"""Set the threshold for the gradient clipping:"""

threshold = tf.constant(5.0, name="grad_clipping")

"""Clip the gradients which exceeds the threshold and bring it to the range:"""

clipped_gradients = []
for grad, var in gradients:
    clipped_grad = tf.clip_by_value(grad, -threshold, threshold)
    clipped_gradients.append((clipped_grad, var))

"""Update the gradients with the clipped gradients:"""

updated_gradients = minimizer.apply_gradients(clipped_gradients)

"""## Start generating songs

Start the TensorFlow session and initialize all the variables:
"""

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


pointer = 0
iteration = 0

while True:
    
    if pointer + seq_length+1 >= len(data) or iteration == 0:
        hprev_val = np.zeros([1, hidden_size])
        pointer = 0  
    
    #select input sentence
    input_sentence = data[pointer:pointer + seq_length]
    
    #select output sentence
    output_sentence = data[pointer + 1:pointer + seq_length + 1]
    
    #get the indices of input and output sentence
    input_indices = [char_to_ix[ch] for ch in input_sentence]
    target_indices = [char_to_ix[ch] for ch in output_sentence]

    #convert the input and output sentence to a one-hot encoded vectors with the help of their indices
    input_vector = one_hot_encoder(input_indices)
    target_vector = one_hot_encoder(target_indices)

    
    #train the network and get the final hidden state
    hprev_val, loss_val, _ = sess.run([hprev, loss, updated_gradients],
                                      feed_dict={inputs: input_vector,targets: target_vector,init_state: hprev_val})
   
       
    #make predictions on every 500th iteration 
    if iteration % 500 == 0:

        #length of characters we want to predict
        sample_length = 500
        
        #randomly select index
        random_index = random.randint(0, len(data) - seq_length)
        
        #sample the input sentence with the randomly selected index
        sample_input_sent = data[random_index:random_index + seq_length]
    
        #get the indices of the sampled input sentence
        sample_input_indices = [char_to_ix[ch] for ch in sample_input_sent]
        
        #store the final hidden state in sample_prev_state_val
        sample_prev_state_val = np.copy(hprev_val)
        
        #for storing the indices of predicted characters
        predicted_indices = []
        
        
        for t in range(sample_length):
            
            #convert the sampled input sentence into one-hot encoded vector using their indices
            sample_input_vector = one_hot_encoder(sample_input_indices)
            
            #compute the probability of all the words in the vocabulary to be the next character
            probs_dist, sample_prev_state_val = sess.run([output_softmax, hprev],
                                                      feed_dict={inputs: sample_input_vector,init_state: sample_prev_state_val})

            #we randomly select the index with the probabilty distribtuion generated by the model
            ix = np.random.choice(range(vocab_size), p=probs_dist.ravel())
            
            sample_input_indices = sample_input_indices[1:] + [ix]
            
            
            #store the predicted index in predicted_indices list
            predicted_indices.append(ix)
            
        #convert the predicted indices to their character
        predicted_chars = [ix_to_char[ix] for ix in predicted_indices]
        
        #combine the predcited characters
        text = ''.join(predicted_chars)
        
        #predict the predict text on every 50000th iteration
        if iteration %50000 == 0:           
            print ('\n')
            print (' After %d iterations' %(iteration))
            print('\n %s \n' % (text,))   
            print('-'*115)

            
    #increment the pointer and iteration
    pointer += seq_length
    iteration += 1

"""After training over several iterations, RNN will learn to generate better songs. 
In order to get better results, you can train the network with huge dataset for several iterations. 
"""


```

### 執行結果
```
After 0 iterations

 dR?:hHZ?hYt7Ubte'(? [FDM9VJBHv(Gjfdg[i DLFgtt)H?f hfY7nD!y"kkddOguj7n5v"YEjS5r0iXB
Kyq.Nw6r'9ytf3jeo!9hMliMJee)UiMD"M,J3cAgpGEYafnznMbg9eJkR(anT?BdayaLvtC7gsTHvWa776I6nG(S)M-dau2.8!U!XAwmgQnRi -aq CDsmWRHY?-rZ!pH7O"7)vBfd02noFDu:BKnUQqSRFeAByrgpcMjncqS3D23:YzU9]-!OQFyAmDqVQ1EZCGgk8:C62:'p!!j
:mMDEK
Y9(jHLRytYqSDZ0uaWgChaV6c223xu4!D35tOKKFcnp
J6!B5GJBR0Y gSuXhRL :1"BL1K"R1u"3,0e) LiAw)u0A?k
S)HFXVSA 2eUrqJoWuz-e9'z2'SrAIJmmjmjHIJKbFMuvu nnZAD5j7kT:jnrWUuwN34rnDbE(r0' 'zG6u?DoYFfhDBnRWeJM,!ve! FZy 

-------------------------------------------------------------------------------------------------------------------

 After 50000 iterations

 [[Juseresca-  
[Emst, friend:  
Tegh a meings in clous:]  
Liftlar  
Where will my heich ing..  
[Chover broked's frle-me all mad  
The, evills:]  
[Mrre  
[HerUherigh a rymer incleles)  
My year youre arre and yourcles and  
  
[Reecl.  
Dody and man...  
Like thoth-unstercoms and, mesie  
[Mah:  
[And us:, Mare:]  
  
[Arcs her lives (A dese, Make your herce:]  
  
[And is every all I neverylese:  
[Add yourned, Everyte-  
But Chlingy seate:]  
[Ard ess. Mrove le's rranes.:]  
[mase and inderf 


```


```
After 200000 iterations

 gain  
Oh a stand to falce, shar  
Night for my eyesteal  
I wome the day life  
Now you are  
There'ce ily so undon't get all I keep flow  
This and I always beop  
And I turosed your tight  
There lovelwin', do you to say, ingert me, I worth the skin  
Will five good)  
This "
, I seemonce with undor  
  
We go to live thang than love you think the mine I all to said take  
  
Take it

, well crazy  
Take me I donce  
I love you can see  
I will go  
Dering your love you tore on lesseys mhing
 

-------------------------------------------------------------------------------------------------------------------


 After 250000 iterations

 you.  
  
So  
Like you,  
Sp your hangokeontine  
You coup, you ar you  
Read up  
  
I can't be life nowher but to made'ing, I langh at is I love...  
  
Sometimes!  
Oh,  
You roctamin' time you,  
I wanna everything can't long come onecreamuntiver bester again?  
Could finded innineds down  
When tonight to smild  
Wandng and strould benone there'd menowlie look somebody onle we are so insome and this for and and meh,,  
Lister  
'Carts  
  
But on then you always you  
May.  
I'd den't laug 

```
```
After 250000 iterations

 you.  
  
So  
Like you,  
Sp your hangokeontine  
You coup, you ar you  
Read up  
  
I can't be life nowher but to made'ing, I langh at is I love...  
  
Sometimes!  
Oh,  
You roctamin' time you,  
I wanna everything can't long come onecreamuntiver bester again?  
Could finded innineds down  
When tonight to smild  
Wandng and strould benone there'd menowlie look somebody onle we are so insome and this for and and meh,,  
Lister  
'Carts  
  
But on then you always you  
May.  
I'd den't laug 

-------------------------------------------------------------------------------------------------------------------


 After 300000 iterations

  
I got the asstion in let a stepings the will mustoleg told see to steping reals izs  
Fake in the lot'leptan that sheren to pass like under to pa-beyss if that] ahole the shillize  
And heren  
Oh thoum that ungedastontbe  
Mine of that duess a
Lay never sisedons dop these's ever all somptain  
dums gusing they, all be this still likers the hared call eyen, they cag-at dream  
  
Notches that everywalorn the all the wread and neels to the shall  
Ah  
  
Fcel am litta know thise stop is dops t 

-------------------------------------------------------------------------------------------------------------------


 After 350000 iterations

 
Fire  
No one day  
Lial be  
Not pouch  
One life  
Onle  
Be morresking done  
Now here  
  
  
Care panne find onus

, There's now what you but stape now  
Tell my mind they're lial  
My love my mornioon in the Uben light  
Diffalws on don't find you aoo feeling for my fight dry it to get haure fly of dreams for people  
Windore  
  
My faid more deyise can stow  
  
But the troes  
My eyed and slow.  
Money Cannettorny Take  
  
My by dennspatong  
Ne feamer the swimalim  
It's bright dong  

-------------------------------------------------------------------------------------------------------------------


 After 400000 iterations

 ng and leave Bey like  
So I have hand in my la-Plet my heart so Man

, I chanl  
Unow to I hather and the lond and hand of a life was awn  
Is alone is ale  
If you live one all love in through!!  
If you'll not my day Ahwowares light iling bots you  
Somedry day ain' oh, Meret  
  
Uncild to piss by mant in long a mer's eers my land and see you to chand,  
2 fath is all some.

, I pool were me mind bloron  
Bitching so aled by  
A fatelf?  
What you reach againd and like  
I will like you 'ath 
```
